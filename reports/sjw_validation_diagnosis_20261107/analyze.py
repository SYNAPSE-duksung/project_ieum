"""Saved SJW Train profiles/Validation predictions and WAV metadata; no inference."""
import csv,json,hashlib,math
from pathlib import Path
from collections import Counter,defaultdict
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
from src.personalization.evaluator import normalize_text,metric_counts,edit_distance
from src.personalization.error_profile import normalize_for_syllable_alignment,align_syllables
OUT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(name,rows):
    if not rows:(OUT/name).write_text('no_rows\n',encoding='utf-8');return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def write(name,data):(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def key(r):return r['batch'],r['audio_member'].replace('\\','/')
def compact(s):return normalize_text(s).replace(' ','')

def main():
    base=ROOT/'reports/sjw_experiment_20261107'
    original=read(base/'train_profile.json')
    excluded=read(ROOT/'outputs/personalization/error_profiles_v2/B_exclude_strong/profiles/SJW_M_29.json')
    predictions=[json.loads(x) for x in (base/'E1_validation_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest=[r for r in csvread(ROOT/'data/splits/personalization_split.csv') if r['speaker_id']=='SJW_M_29' and r['personal_split'] in ('profile_train','personal_val')]
    mp={key(r):r for r in manifest};assert len(manifest)==253 and len(predictions)==63
    candidates=[r for r in csvread(ROOT/'outputs/personalization/error_profiles/predictions.csv') if r['speaker_id']=='SJW_M_29']
    audit=[r for r in csvread(ROOT/'outputs/personalization/error_profiles_v2/suspected_mismatch_audit.csv') if r['speaker_id']=='SJW_M_29']
    flags={key(r):r for r in audit}
    reference_pool=[(r,compact(r['text'])) for r in manifest]
    groups=defaultdict(list);utterances=[];syllable_val=defaultdict(Counter)
    for record in predictions:
        row=mp[key(record)];rid=row['recording_id'];ref=record['reference'];hyp=record['hypothesis']
        groups[rid].append(record)
        for a,b in align_syllables(normalize_for_syllable_alignment(ref),normalize_for_syllable_alignment(hyp)):
            if a!='<EMPTY>':
                syllable_val[a]['N']+=1;syllable_val[a]['C' if a==b else 'D' if b=='<EMPTY>' else 'S']+=1
        path=ROOT.parent/'ieum_datasets'/row['batch']/row['audio_member']
        info=sf.info(path);audio,sr=sf.read(path,dtype='float32',always_2d=True)
        rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2)));peak=float(np.max(np.abs(audio)))
        hc=compact(hyp);rc=compact(ref)
        best=None
        for other,oc in reference_pool:
            if key(other)==key(row):continue
            score=1-edit_distance(hc,oc)/max(len(hc),len(oc),1)
            if best is None or score>best[0]:best=(score,other)
        score,other=best
        own=1-edit_distance(hc,rc)/max(len(hc),len(rc),1)
        exact=bool(hc and hc==compact(other['text']))
        # Descriptive review screen, explicitly separate from historical Strong flags.
        review=record['cer']>=.5 and score>=.9 and score-own>=.3
        utterances.append(dict(audio_member=row['audio_member'],recording_id=rid,reference=ref,hypothesis=hyp,
            ce=record['ce'],nc=record['nc'],cer=record['cer'],we=record['we'],nw=record['nw'],
            duration=info.duration,samplerate=sr,channels=info.channels,rms=rms,peak=peak,
            near_zero_fraction=float(np.mean(np.abs(audio)<1e-4)),near_full_scale_fraction=float(np.mean(np.abs(audio)>=.999)),
            own_similarity=own,best_other_similarity=score,exact_other_match=exact,review_screen=review,
            matched_audio=other['audio_member'],matched_recording=other['recording_id'],matched_split=other['personal_split'],
            matched_reference=other['text'],same_recording=other['recording_id']==rid))
    table('validation_utterance_audit.csv',utterances)
    table('reference_mismatch_review.csv',[r for r in utterances if r['review_screen']])
    group_rows=[]
    for rid,records in groups.items():
        subset=[r for r in utterances if r['recording_id']==rid]
        ce=sum(r['ce'] for r in records);nc=sum(r['nc'] for r in records)
        train=[r for r in candidates if mp[key(r)]['recording_id']==rid]
        tce=tn=0
        for r in train:
            a=normalize_for_syllable_alignment(r['reference']);b=normalize_for_syllable_alignment(r['prediction'])
            tce+=edit_distance(a,b);tn+=len(a)
        group_rows.append(dict(recording_id=rid,val_count=len(records),val_reference_chars=nc,val_errors=ce,
            val_cer=ce/nc,erroneous_utterances=sum(r['ce']>0 for r in records),
            mean_duration=float(np.mean([r['duration'] for r in subset])),mean_rms=float(np.mean([r['rms'] for r in subset])),
            samplerates=sorted({r['samplerate'] for r in subset}),channels=sorted({r['channels'] for r in subset}),
            exact_other_matches=sum(r['exact_other_match'] for r in subset),review_screen_count=sum(r['review_screen'] for r in subset),
            train_utterances=len(train),train_hangul_cer=tce/tn if tn else None,
            train_strong_suspects=sum(flags.get(key(r),{}).get('excluded_in_B')=='1' for r in train)))
    group_rows.sort(key=lambda r:-r['val_errors']);table('recording_groups.csv',group_rows)
    syls=original['syllable_profiles'];weights=read(base/'preflight.json')['weight_info']['syllable_weights'];syllable_rows=[]
    for s,d in syls.items():
        if d['error_count']<2:continue
        v=syllable_val[s]
        syllable_rows.append(dict(syllable=s,train_N=d['total_count'],train_errors=d['error_count'],train_error_rate=d['error_rate'],
            weighted=weights[s]>1,weight=weights[s],val_N=v['N'],val_S=v['S'],val_D=v['D'],val_errors=v['S']+v['D'],
            val_error_rate=(v['S']+v['D'])/v['N'] if v['N'] else None))
    syllable_rows.sort(key=lambda r:(-r['train_errors'],-r['val_N']));table('repeated_syllable_coverage.csv',syllable_rows)
    coverage=[]
    for label,subset in [('all_repeated_reference_syllables',syllable_rows),('weighted_51',[r for r in syllable_rows if r['weighted']])]:
        coverage.append(dict(group=label,types=len(subset),val_present_types=sum(r['val_N']>0 for r in subset),
            val_N_ge5_types=sum(r['val_N']>=5 for r in subset),val_N_ge10_types=sum(r['val_N']>=10 for r in subset),
            val_errored_types=sum(r['val_errors']>0 for r in subset),val_occurrences=sum(r['val_N'] for r in subset),
            val_reference_errors=sum(r['val_errors'] for r in subset)))
    total=sum(r['ce'] for r in utterances);allchars=sum(r['nc'] for r in utterances)
    summary=dict(ranked_cer=excluded['overall_statistics']['cer'],ranked_rows=excluded['utterance_count'],
        original_train_rows=190,strong_excluded=6,strong_excluded_fraction=6/190,
        original_train_hangul_cer=original['overall_statistics']['cer'],ranked_overall=excluded['overall_statistics'],
        e1_validation_cer=total/allchars,val_rows=len(utterances),val_errors=total,val_reference_chars=allchars,
        erroneous_val_utterances=sum(r['ce']>0 for r in utterances),recording_groups=group_rows,
        top3_errors=sum(r['ce'] for r in sorted(utterances,key=lambda r:-r['ce'])[:3]),
        exact_other_matches=sum(r['exact_other_match'] for r in utterances),
        review_screen_count=sum(r['review_screen'] for r in utterances),
        review_screen_error_count=sum(r['ce'] for r in utterances if r['review_screen']),
        review_screen_definition='CER>=0.5; other Train/Val reference similarity>=0.9; exceeds own-reference similarity by>=0.3; review only, not historical Strong',
        coverage=coverage,train_strong_flags=list(flags.values()),personal_test_accessed=False,new_model_inference=False)
    kept=[r for r in candidates if flags.get(key(r),{}).get('excluded_in_B')!='1']
    assert len(kept)==184
    normalization=[]
    for label,subset in [('General Train190',candidates),('General filtered Train184',kept)]:
        totals=[0,0,0,0]
        for r in subset:
            for i,value in enumerate(metric_counts(r['reference'],r['prediction'])):totals[i]+=value
        normalization.append(dict(set=label,utterances=len(subset),cer=totals[0]/totals[1],wer=totals[2]/totals[3],
            character_errors=totals[0],reference_characters=totals[1]))
    summary.update(same_cache_e1_normalization=normalization,
        erroneous_exact_other_matches=sum(r['ce']>0 and r['exact_other_match'] for r in utterances),
        erroneous_exact_same_recording_matches=sum(r['ce']>0 and r['exact_other_match'] and r['same_recording'] for r in utterances),
        erroneous_exact_other_train_matches=sum(r['ce']>0 and r['exact_other_match'] and r['matched_split']=='profile_train' for r in utterances),
        erroneous_exact_other_val_matches=sum(r['ce']>0 and r['exact_other_match'] and r['matched_split']=='personal_val' for r in utterances))
    table('normalization_comparison.csv',normalization)
    write('summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='train_strong_flags'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
