"""Legacy-E1 controls around unchanged E2 model/loss; Validation diagnostics."""
from collections import Counter
import json,math,shutil,time
from pathlib import Path
import torch
from transformers import TrainerCallback,WhisperForConditionalGeneration
from .ctc_utils import E2SpeechDataset,E2Collator,ID_TO_TOKEN,BLANK_ID
from .e2_trainer import E2Trainer,make_e2_trainer


class IdentifiedDataset(E2SpeechDataset):
    def __getitem__(self,index):
        item=super().__getitem__(index)
        item['_sample_ids']=self.rows[index]['batch']+'/'+self.rows[index]['audio_member']
        return item


class IdentifiedCollator(E2Collator):
    def __call__(self,features):
        output=super().__call__(features)
        output['_sample_ids']=[f['_sample_ids'] for f in features]
        return output


def gradient_norm(parameters):
    grads=[p.grad.detach() for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))) if grads else 0.


def ctc_diagnostics(logits,lengths,ids):
    rows=[]
    for i,t in enumerate(lengths.tolist()):
        values=logits[i,:t].float();prob=values.softmax(-1);path=values.argmax(-1)
        collapsed=torch.unique_consecutive(path);tokens=collapsed[collapsed!=BLANK_ID].tolist()
        rows.append(dict(sample_id=ids[i],frames=t,blank_argmax_frames=int((path==0).sum()),
            nonblank_argmax_frames=int((path!=0).sum()),mean_blank_probability=float(prob[:,0].mean()),
            nonblank_tokens=len(tokens),greedy=''.join(ID_TO_TOKEN[j] for j in tokens)))
    return rows


def summarize_ctc(rows):
    if not rows:return None
    frames=sum(r['frames'] for r in rows)
    return dict(utterances=len(rows),frames=frames,blank_argmax_fraction=sum(r['blank_argmax_frames'] for r in rows)/frames,
        mean_blank_probability=sum(r['mean_blank_probability']*r['frames'] for r in rows)/frames,
        nonblank_argmax_frames=sum(r['nonblank_argmax_frames'] for r in rows),
        nonblank_collapsed_tokens=sum(r['nonblank_tokens'] for r in rows),
        all_blank_utterances=sum(r['nonblank_tokens']==0 for r in rows),
        all_blank_fraction=sum(r['nonblank_tokens']==0 for r in rows)/len(rows))


class ControlAudit(TrainerCallback):
    def __init__(self,directory):
        self.directory=Path(directory);self.report=dict(epochs=[],nonfinite_events=[],optimizer_steps=0,scheduler_steps=0)
        self.losses=Counter();self.micro=[];self.eval_ctc=[];self.gradients=[];self.optimizer_events=0

    def save(self):
        tmp=self.directory/'audit.tmp';tmp.write_text(json.dumps(self.report,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(self.directory/'training_audit.json')

    def event(self,name,row):
        with (self.directory/name).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

    def failure(self,stage,step,error):
        self.report['nonfinite_events'].append(dict(stage=stage,step=step,error=str(error)));self.save()

    def on_train_begin(self,args,state,control,**kwargs):
        self.started=time.perf_counter();torch.cuda.reset_peak_memory_stats();self.save()

    def on_optimizer_step(self,args,state,control,**kwargs):self.optimizer_events+=1

    def on_step_end(self,args,state,control,lr_scheduler=None,**kwargs):
        # Transformers 5.18 invokes create_scheduler directly. Observe its real
        # state after every update rather than a deprecated construction hook.
        if lr_scheduler is None:raise RuntimeError('Scheduler missing from callback')
        self.report['scheduler_steps']=int(lr_scheduler.last_epoch)
        self.event('scheduler_events.jsonl',dict(step=state.global_step,last_epoch=lr_scheduler.last_epoch,
            init_inclusive_step_count=getattr(lr_scheduler,'_step_count',None),learning_rate=lr_scheduler.get_last_lr()))

    def on_epoch_end(self,args,state,control,**kwargs):
        n=self.losses['samples']
        self.report['epochs'].append(dict(epoch=state.epoch,step=state.global_step,
            train={k:self.losses[k]/n for k in ('asr_loss','ctc_loss','weighted_ctc_loss','total_loss')},
            gradient_preclip={k:dict(mean=sum(r[k] for r in self.gradients)/len(self.gradients),max=max(r[k] for r in self.gradients))
                for k in ('total','encoder','decoder','head')},samples=n))
        self.losses.clear();self.gradients.clear();self.save()

    def on_evaluate(self,args,state,control,metrics,**kwargs):
        item=next((r for r in self.report['epochs'] if r['step']==state.global_step and 'validation' not in r),None)
        if item is not None:
            item['validation']={k:v for k,v in metrics.items() if k in ('eval_loss','eval_cer','eval_wer')}
            item['ctc']=summarize_ctc(self.eval_ctc)
            for row in self.eval_ctc:self.event('validation_ctc.jsonl',dict(epoch=state.epoch,step=state.global_step,**row))
        self.eval_ctc.clear();self.save()
        if shutil.disk_usage(self.directory).free<5*1024**3:raise OSError('Disk reserve below 5GiB')

    def on_log(self,args,state,control,logs=None,**kwargs):self.event('training_logs.jsonl',dict(logs or {},step=state.global_step))

    def on_train_end(self,args,state,control,**kwargs):
        self.report.update(training_seconds=time.perf_counter()-self.started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            optimizer_steps=self.optimizer_events,best_checkpoint=state.best_model_checkpoint,best_validation_cer=state.best_metric)
        self.save()


class LegacyE2Trainer(E2Trainer):
    def __init__(self,*args,control_audit,ctc_enabled,**kwargs):
        super().__init__(*args,**kwargs)
        self.model_accepts_loss_kwargs=True
        self.control_audit=control_audit;self.ctc_enabled=ctc_enabled;self.eval_ids=None;self.last_components=None;self.ctc_logits=None
        def observe_head(module,args,output):
            if not self.model.training:self.ctc_logits=output.detach()
        self.model.ctc_head.register_forward_hook(observe_head)

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        clean=dict(inputs);ids=clean.pop('_sample_ids',self.eval_ids)
        if self.ctc_enabled:
            output=model(**clean)
            asr,ctc,weighted,total=output.asr_loss,output.ctc_loss,output.weighted_ctc_loss,output.loss
        else:
            # Keep model.__call__/Accelerate autocast identical to R1/E1.
            # Original E2 forward computes raw CTC, but it is discarded from
            # the optimized scalar. The head receives no backward gradient.
            output=model(**clean)
            asr=total=output.asr_loss;ctc=weighted=asr.new_zeros(())
            output.loss=asr
        if not all(torch.isfinite(x) for x in (asr,ctc,weighted,total)):
            self.control_audit.failure('forward',self.state.global_step,'Nonfinite loss');raise FloatingPointError('Nonfinite loss')
        values=dict(asr_loss=float(asr.detach()),ctc_loss=float(ctc.detach()),weighted_ctc_loss=float(weighted.detach()),total_loss=float(total.detach()))
        if model.training:
            self.last_components=dict(values,sample_ids=ids)
            n=inputs['labels'].shape[0];self.control_audit.losses['samples']+=n
            for k,v in values.items():self.control_audit.losses[k]+=v*n
        elif self.ctc_enabled and self.ctc_logits is not None:
            self.control_audit.eval_ctc.extend(ctc_diagnostics(self.ctc_logits,inputs['ctc_input_lengths'],ids))
        self.ctc_logits=None
        return (total,output) if return_outputs else total

    def training_step(self,model,inputs,num_items_in_batch=None):
        if num_items_in_batch is None:raise RuntimeError('Legacy verification failed: num_items_in_batch missing')
        original=self.accelerator.backward;observed=[]
        def trace(loss,**kwargs):
            observed.append(float(loss.detach()));return original(loss,**kwargs)
        self.accelerator.backward=trace
        try:loss=super().training_step(model,inputs,num_items_in_batch)
        finally:self.accelerator.backward=original
        assert len(observed)==1
        assert abs(observed[0]-self.last_components['total_loss'])<1e-5
        if self.accelerator.gradient_accumulation_steps!=1:raise RuntimeError('Unexpected Accelerate loss scaling')
        row=dict(step=self.state.global_step,epoch=self.state.epoch,**self.last_components,
            backward_loss=observed[0],divided_by_accumulation=False,
            current_accumulation_steps=self.current_gradient_accumulation_steps,num_items_in_batch=int(num_items_in_batch))
        self.control_audit.micro.append(row);self.control_audit.event('backward_audit.jsonl',row)
        return loss

    def _clip_grad_norm(self,model):
        norms=dict(total=gradient_norm(model.parameters()),encoder=gradient_norm(model.model.encoder.parameters()),
            decoder=gradient_norm(model.model.decoder.parameters()),head=gradient_norm(model.ctc_head.parameters()))
        if not all(math.isfinite(v) for v in norms.values()):
            self.control_audit.failure('preclip',self.state.global_step,norms);raise FloatingPointError('Nonfinite gradients')
        row=dict(step=self.state.global_step+1,microbatches=len(self.control_audit.micro),**norms)
        self.control_audit.gradients.append(row);self.control_audit.event('gradients_preclip.jsonl',row)
        self.control_audit.micro.clear()
        return super()._clip_grad_norm(model)

    def prediction_step(self,model,inputs,prediction_loss_only,ignore_keys=None,**gen_kwargs):
        clean=dict(inputs);self.eval_ids=clean.pop('_sample_ids',None)
        return super().prediction_step(model,clean,prediction_loss_only,ignore_keys=ignore_keys,**gen_kwargs)


def make_control_trainer(model,processor,datasets,directory,settings,audit,enabled):
    template=make_e2_trainer(model,processor,datasets,directory,settings)
    from transformers.trainer_utils import SaveStrategy
    template.args.save_strategy=SaveStrategy.BEST;template.args.save_only_model=True;template.args.save_total_limit=1
    return LegacyE2Trainer(model=model,args=template.args,train_dataset=datasets['train'],eval_dataset=datasets['val'],
        data_collator=IdentifiedCollator(processor),processing_class=processor,compute_metrics=template.compute_metrics,
        callbacks=[audit],control_audit=audit,ctc_enabled=enabled)
