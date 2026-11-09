"""Position-weighted CE for E4; original E1 modules remain unchanged."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import time
import torch
from torch.nn import functional as F
from transformers import Seq2SeqTrainer, TrainerCallback
from transformers.convert_slow_tokenizer import bytes_to_unicode
from .dataset import WhisperSpeechDataset, WhisperCollator
from .error_profile import align_syllables, normalize_for_syllable_alignment
from .profile_builder_v2 import calculate_statistics


def build_weights(rows, joined, processor, profile):
    """Validate stored Train profile and produce A/B weights without truncation."""
    tok = processor.tokenizer
    pairs = [dict(reference=r['text'].strip(), prediction=r['_general_hypothesis']) for r in joined]
    overall, syllables, _ = calculate_statistics(pairs)
    if overall != profile['overall_statistics'] or syllables != profile['syllable_profiles']:
        raise ValueError('Stored profile differs from exact Train cache')
    error_utterances = defaultdict(set)
    for i, r in enumerate(pairs):
        for a, b in align_syllables(normalize_for_syllable_alignment(r['reference']),
                                     normalize_for_syllable_alignment(r['prediction'])):
            if a != '<EMPTY>' and a != b:
                error_utterances[a].add(i)
    p0 = (overall['substitution_count']+overall['deletion_count'])/overall['reference_syllable_count']
    syllable_weights = {}
    for s, d in syllables.items():
        n, e = d['total_count'], d['error_count']
        rate = (e+10*p0)/(n+10)
        syllable_weights[s] = 1+min(.5,max(0.,rate-p0)) if n>=5 and e>=2 and len(error_utterances[s])>=2 else 1.
    byte_decoder = {v:k for k,v in bytes_to_unicode().items()}
    a_rows, b_rows, examples = [], [], []
    counts = Counter()
    for row, hypothesis in zip(rows, joined):
        text = row['text'].strip()
        ids = tok(text,add_special_tokens=False).input_ids
        raw = [bytes(byte_decoder[c] for c in tok.convert_ids_to_tokens(t)) for t in ids]
        full = tok(text,add_special_tokens=True).input_ids
        if b''.join(raw) != text.encode('utf-8') or full != tok.prefix_tokens+ids+[tok.eos_token_id]:
            raise ValueError(f'Token byte alignment failed: {row["audio_member"]}')
        chars=[]; position=0
        for ch in text:
            end=position+len(ch.encode('utf-8')); chars.append((ch,position,end)); position=end
        weights=[]; position=0
        for token, fragment in zip(ids,raw):
            end=position+len(fragment)
            overlap=[(s,min(end,b)-max(position,a)) for s,a,b in chars if min(end,b)>max(position,a)]
            weight=1+sum(n*(syllable_weights.get(s,1.)-1) for s,n in overlap)/len(fragment)
            weights.append(weight)
            if weight>1 and len(examples)<30:
                examples.append(dict(audio=row['audio_member'],text=text,token_id=token,
                    byte_hex=fragment.hex(),overlap=overlap,weight=weight))
            position=end
        # A uses the same eligibility/equality as E3; no silent alternate policy.
        eligible=hypothesis['_correction_reason']=='eligible'
        counts['a_weighted_utterances']+=eligible
        aa=[1.5 if eligible else 1.]*len(full)
        bb=[1.]*len(tok.prefix_tokens)+weights+[1.]
        for values in (aa,bb):
            tensor=torch.tensor(values)
            if not torch.isfinite(tensor).all() or tensor.min()<1 or tensor.max()>1.5:
                raise ValueError('Invalid token weights')
        counts.update(train_utterances=1,content_tokens=len(ids),all_label_tokens=len(full),
                      a_weighted_tokens=sum(w>1 for w in aa),b_weighted_content_tokens=sum(w>1 for w in weights),
                      b_weighted_utterances=int(any(w>1 for w in weights)))
        a_rows.append(dict(row,_token_weights=aa));b_rows.append(dict(row,_token_weights=bb))
    if counts['a_weighted_utterances']!=34:
        raise ValueError('Expected exactly the existing 34 E3 eligible candidates')
    return dict(A=a_rows,B=b_rows),dict(counts=counts,syllable_weights=syllable_weights,
        weighted_syllable_types=sum(w>1 for w in syllable_weights.values()),p0=p0,
        smoothing='(e+10*p0)/(n+10)',syllable_rule='1+min(.5,max(0,smoothed-p0)) gated by n>=5,e>=2,error_utterances>=2',
        reduction='sum(valid_token_weight*CE)/sum(valid_token_weight)',examples=examples)


class WeightedDataset(WhisperSpeechDataset):
    def __getitem__(self,index):
        result=super().__getitem__(index)
        result['token_weights']=self.rows[index]['_token_weights']
        return result


class WeightedCollator(WhisperCollator):
    def __call__(self,features):
        result=super().__call__(features)
        if not any('token_weights' in f for f in features):
            return result
        width=result['labels'].shape[1]
        weights=torch.ones(result['labels'].shape,dtype=torch.float32)
        remove_first=self.processor.tokenizer.bos_token_id is not None and all(
            f['labels'][0]==self.processor.tokenizer.bos_token_id for f in features)
        for i,f in enumerate(features):
            values=f.get('token_weights',[1.]*len(f['labels']))
            if len(values)!=len(f['labels']):raise ValueError('Weight/label length mismatch')
            if remove_first:values=values[1:]
            if len(values)>width:raise ValueError('Weight padding mismatch')
            weights[i,:len(values)]=torch.tensor(values)
        result['token_weights']=weights
        return result


class WeightedTrainer(Seq2SeqTrainer):
    def __init__(self,*args,audit=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.model_accepts_loss_kwargs=True  # preserve historical E1 accumulation
        self.audit=audit

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        inputs=dict(inputs); weights=inputs.pop('token_weights',None)
        output=model(**inputs,return_dict=True)
        unweighted=output.loss
        if weights is not None and model.training:
            labels=inputs['labels'];valid=labels.ne(-100)
            if weights.shape!=labels.shape or not torch.isfinite(weights).all() or weights.min()<1 or weights.max()>1.5:
                raise ValueError('Invalid batch weights')
            ce=F.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),
                               labels.reshape(-1),reduction='none',ignore_index=-100).view_as(weights)
            denominator=(weights*valid).sum()
            loss=(ce*weights).sum()/denominator
        else:
            loss=unweighted
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite E4 loss')
        if self.audit is not None and model.training:
            self.audit.add(float(unweighted.detach()),float(loss.detach()),inputs['labels'].shape[0])
        output.loss=loss
        return (loss,output) if return_outputs else loss

    def prediction_step(self,model,inputs,prediction_loss_only,ignore_keys=None,**gen_kwargs):
        return super().prediction_step(model,{k:v for k,v in inputs.items() if k!='token_weights'},
            prediction_loss_only,ignore_keys=ignore_keys,**gen_kwargs)


class WeightedAudit(TrainerCallback):
    def __init__(self,directory):
        self.directory=Path(directory);self.report=dict(epochs=[],nonfinite_events=[])
        self.sums=[0.,0.,0];self.started=None

    def save(self):
        path=self.directory/'training_audit.json';temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.report,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

    def add(self,asr,weighted,n):
        self.sums[0]+=asr*n;self.sums[1]+=weighted*n;self.sums[2]+=n

    def on_train_begin(self,args,state,control,**kwargs):
        self.started=time.perf_counter();torch.cuda.reset_peak_memory_stats();self.save()

    def on_epoch_end(self,args,state,control,**kwargs):
        a,w,n=self.sums
        self.report['epochs'].append(dict(epoch=state.epoch,step=state.global_step,
            asr_loss=a/n,weighted_loss=w/n,loss_reporting='sample-weighted microbatch loss means'))
        self.sums=[0.,0.,0];self.save()

    def on_evaluate(self,args,state,control,metrics,**kwargs):
        for item in self.report['epochs']:
            if item['step']==state.global_step and 'validation' not in item:
                item['validation']={k:v for k,v in metrics.items() if k in ('eval_cer','eval_wer','eval_loss')}
                self.save();break
        if shutil.disk_usage(self.directory).free<5*1024**3:raise OSError('Less than 5GiB free disk reserve')

    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        if gradients and not torch.isfinite(torch.stack(torch._foreach_norm(gradients))).all():
            self.report['nonfinite_events'].append(dict(step=state.global_step,event='nonfinite gradient'))
            self.save();raise FloatingPointError('Nonfinite E4 gradient')

    def on_log(self,args,state,control,logs=None,**kwargs):
        with (self.directory/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(logs or {},step=state.global_step),allow_nan=False)+'\n')

    def on_train_end(self,args,state,control,**kwargs):
        self.report.update(training_seconds=time.perf_counter()-self.started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            best_checkpoint=state.best_model_checkpoint,best_validation_cer=state.best_metric)
        self.save()
