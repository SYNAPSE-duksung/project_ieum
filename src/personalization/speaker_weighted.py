"""Speaker-generic adapter for the fixed KSM E4 weighting policy."""
from collections import Counter,defaultdict
import torch
from transformers.convert_slow_tokenizer import bytes_to_unicode
from .profile_builder_v2 import calculate_statistics
from .error_profile import normalize_for_syllable_alignment,align_syllables


def build_speaker_weights(rows,joined,processor,profile):
    tok=processor.tokenizer
    pairs=[dict(reference=r['text'].strip(),prediction=h['_general_hypothesis']) for r,h in zip(rows,joined)]
    overall,syllables,_=calculate_statistics(pairs)
    if overall!=profile['overall_statistics'] or syllables!=profile['syllable_profiles']:
        raise ValueError('Train cache/Profile mismatch')
    errors=defaultdict(set)
    for i,r in enumerate(pairs):
        for a,b in align_syllables(normalize_for_syllable_alignment(r['reference']),normalize_for_syllable_alignment(r['prediction'])):
            if a!='<EMPTY>' and a!=b:errors[a].add(i)
    p0=(overall['substitution_count']+overall['deletion_count'])/overall['reference_syllable_count']
    sw={s:(1+min(.5,max(0,(d['error_count']+10*p0)/(d['total_count']+10)-p0))
        if d['total_count']>=5 and d['error_count']>=2 and len(errors[s])>=2 else 1.) for s,d in syllables.items()}
    decoder={v:k for k,v in bytes_to_unicode().items()}
    variants=dict(A=[],B=[]);counts=Counter();examples=[]
    for row,h in zip(rows,joined):
        text=row['text'].strip();ids=tok(text,add_special_tokens=False).input_ids
        fragments=[bytes(decoder[c] for c in tok.convert_ids_to_tokens(t)) for t in ids]
        full=tok(text,add_special_tokens=True).input_ids
        if b''.join(fragments)!=text.encode('utf-8') or full!=tok.prefix_tokens+ids+[tok.eos_token_id]:
            raise ValueError('Byte/label mapping mismatch: '+row['audio_member'])
        spans=[];pos=0
        for ch in text:
            end=pos+len(ch.encode('utf-8'));spans.append((ch,pos,end));pos=end
        content=[];pos=0
        for t,fragment in zip(ids,fragments):
            end=pos+len(fragment)
            overlap=[(s,min(end,b)-max(pos,a)) for s,a,b in spans if min(end,b)>max(pos,a)]
            w=1+sum(n*(sw.get(s,1)-1) for s,n in overlap)/len(fragment)
            content.append(w);pos=end
            if w>1 and len(examples)<20:examples.append(dict(audio=row['audio_member'],token=t,byte_hex=fragment.hex(),overlap=overlap,weight=w))
        # E4-A is normalized inequality, including empty/long hypotheses; no
        # candidate Decoder is used, so E3 correction exclusions do not apply.
        error=h['_correction_reason']!='same'
        aa=[1.5 if error else 1.]*len(full);bb=[1.]*len(tok.prefix_tokens)+content+[1.]
        for v,weights in [('A',aa),('B',bb)]:
            t=torch.tensor(weights)
            if len(weights)!=len(full) or not torch.isfinite(t).all() or t.min()<1 or t.max()>1.5:
                raise ValueError('Invalid weights')
            variants[v].append(dict(row,_token_weights=weights))
        counts.update(train_utterances=1,a_weighted_utterances=int(error),b_weighted_utterances=int(any(w>1 for w in content)),
            content_tokens=len(ids),all_label_tokens=len(full),a_weighted_tokens=sum(w>1 for w in aa),b_weighted_content_tokens=sum(w>1 for w in content))
    return variants,dict(counts=dict(counts),syllable_weights=sw,weighted_syllable_types=sum(w>1 for w in sw.values()),p0=p0,
        policy='n>=5,e>=2,error utterances>=2; r=(e+10*p0)/(n+10); w=1+min(.5,max(0,r-p0)); byte overlap projection',
        reduction='sum(valid_weight*CE)/sum(valid_weight)',examples=examples)
