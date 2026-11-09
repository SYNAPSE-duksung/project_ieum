"""Saved SJW best models: identical Validation predictions and error analysis."""
import csv
import gc
import json
import os
from pathlib import Path
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import torch
import yaml
from src.personalization.trainer import build_model_and_processor,make_trainer
from src.personalization.dataset import load_speaker_splits,WhisperSpeechDataset
from src.personalization.evaluator import decode_predictions,metric_counts
from src.personalization.error_profile import align_syllables,normalize_for_syllable_alignment,build_raw_error_profile

REPORT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/personalization/second_speaker/SJW_M_29'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(name,value):(REPORT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(name,rows):
    if not rows:
        (REPORT/name).write_text('no_rows\n',encoding='utf-8');return
    with (REPORT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    cfg=yaml.safe_load((ROOT/'scripts/07_personalization/configs/e1.yaml').read_text(encoding='utf-8'))
    rows=load_speaker_splits(cfg['paths']['manifest_csv'],'SJW_M_29',cfg['columns'],cfg['split_labels'],cfg['paths']['audio_root'],cfg['recording_id_column'])['val']
    assert len(rows)==63
    preflight=read(REPORT/'preflight.json');targets={s:w for s,w in preflight['weight_info']['syllable_weights'].items() if w>1}
    assert len(targets)==51
    results={};summaries=[];charstats={};patternstats={}
    with (ROOT/'outputs/personalization/error_profiles/predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        train_pairs=[(r['reference'],r['prediction']) for r in csv.DictReader(f) if r['speaker_id']=='SJW_M_29']
    assert len(train_pairs)==190
    repeated=[p for p in build_raw_error_profile(train_pairs) if p['error_type']=='substitution' and p['count']>=2]
    repeat_keys={(p['reference_syllable'],p['predicted_syllable']) for p in repeated}
    pair_support={key:set() for key in repeat_keys}
    for i,(ref,hyp) in enumerate(train_pairs):
        for a,b in align_syllables(normalize_for_syllable_alignment(ref),normalize_for_syllable_alignment(hyp)):
            if (a,b) in pair_support:pair_support[(a,b)].add(i)
    for name in ('E1','E4-A','E4-B'):
        trained=read(OUT/name/'summary.json')
        assert read(OUT/name/'status.json')['status']=='completed'
        model,processor=build_model_and_processor(trained['best_checkpoint'])
        ds=WhisperSpeechDataset(rows,processor,'text',max_audio_seconds=30.,max_label_length=160)
        t=make_trainer(model,processor,ds,ds,REPORT/f'{name}_validation_runtime',cfg['training'])
        pred=t.predict(ds,metric_key_prefix='validation')
        refs,hyps=decode_predictions(pred.predictions,pred.label_ids,processor.tokenizer)
        raw=processor.tokenizer.batch_decode(pred.predictions,skip_special_tokens=True)
        records=[];stats={};pair_counts=Counter();total=Counter()
        for row,ref,hyp,rawhyp in zip(rows,refs,hyps,raw):
            ce,nc,we,nw=metric_counts(ref,hyp);sdi=Counter()
            for a,b in align_syllables(list(ref.replace(' ','')),list(hyp.replace(' ',''))):
                if a!=b:sdi['I' if a=='<EMPTY>' else 'D' if b=='<EMPTY>' else 'S']+=1
            assert sum(sdi.values())==ce
            total.update(ce=ce,nc=nc,we=we,nw=nw,**sdi)
            for a,b in align_syllables(normalize_for_syllable_alignment(ref),normalize_for_syllable_alignment(hyp)):
                if a=='<EMPTY>':continue
                v=stats.setdefault(a,Counter());v['N']+=1;v['C' if a==b else 'D' if b=='<EMPTY>' else 'S']+=1
                if (a,b) in repeat_keys:pair_counts[(a,b)]+=1
            records.append(dict(batch=row['batch'],audio_member=row['audio_member'],recording_id=row['recording_id'],reference=ref,
                hypothesis=hyp,raw_hypothesis=rawhyp,ce=ce,nc=nc,we=we,nw=nw,cer=ce/nc if nc else 0,wer=we/nw if nw else 0,
                S=sdi['S'],D=sdi['D'],I=sdi['I']))
        results[name]=records;charstats[name]=stats;patternstats[name]=pair_counts
        recomputed_cer=total['ce']/total['nc'];recomputed_wer=total['we']/total['nw']
        summary=dict(speaker='SJW_M_29',experiment=name,best_epoch=trained['best_epoch'],validation_cer=recomputed_cer,
            validation_wer=recomputed_wer,logged_best_cer=trained['best_validation_cer'],logged_best_wer=trained['best_validation_wer'],
            training_seconds=trained['training_seconds'],best_checkpoint=trained['best_checkpoint'],
            counts=dict(total),best_reload_matches_training_cer=recomputed_cer==trained['best_validation_cer'],
            best_reload_matches_training_wer=recomputed_wer==trained['best_validation_wer'])
        summaries.append(summary)
        with (REPORT/f'{name}_validation_predictions.jsonl').open('w',encoding='utf-8') as f:
            for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        del model,t,pred,ds;gc.collect();torch.cuda.empty_cache()
    error_order=sorted(results['E1'],key=lambda r:-r['ce'])
    table('validation_error_concentration.csv',[dict(rank=i+1,audio_member=r['audio_member'],reference=r['reference'],
        hypothesis=r['hypothesis'],character_errors=r['ce'],reference_characters=r['nc'],cer=r['cer']) for i,r in enumerate(error_order)])
    comparisons=[];changes={}
    for name in ('E4-A','E4-B'):
        c=Counter()
        for a,b in zip(results['E1'],results[name]):
            assert a['audio_member']==b['audio_member'] and a['reference']==b['reference']
            verdict='improved' if b['ce']<a['ce'] else 'worsened' if b['ce']>a['ce'] else 'equal'
            wverdict='improved' if b['we']<a['we'] else 'worsened' if b['we']>a['we'] else 'equal'
            c[verdict]+=1;c['wer_'+wverdict]+=1;c['changed']+=a['hypothesis']!=b['hypothesis'];c['raw_changed']+=a['raw_hypothesis']!=b['raw_hypothesis']
            comparisons.append(dict(experiment=name,audio_member=a['audio_member'],reference=a['reference'],e1_prediction=a['hypothesis'],
                prediction=b['hypothesis'],changed=a['hypothesis']!=b['hypothesis'],e1_cer=a['cer'],cer=b['cer'],cer_delta=b['cer']-a['cer'],
                e1_wer=a['wer'],wer=b['wer'],wer_delta=b['wer']-a['wer'],cer_outcome=verdict,wer_outcome=wverdict,
                e1_S=a['S'],S=b['S'],e1_D=a['D'],D=b['D'],e1_I=a['I'],I=b['I']))
        changes[name]={k:c[k] for k in ('changed','raw_changed','improved','worsened','equal','wer_improved','wer_worsened','wer_equal')}
    table('utterance_comparison.csv',comparisons)
    allchars=set(targets)|set(charstats['E1']);syllable_rows=[];groups=[]
    for s in sorted(allchars):
        item=dict(syllable=s,is_target=s in targets,train_weight=targets.get(s,1.),validation_reference_count=charstats['E1'].get(s,{}).get('N',0))
        for name in results:
            v=charstats[name].get(s,Counter())
            for key in ('C','S','D'):item[name+'_'+key]=v.get(key,0)
            item[name+'_error_rate']=(v.get('S',0)+v.get('D',0))/v['N'] if v.get('N',0) else None
        for name in ('E4-A','E4-B'):item[name+'_error_delta']=item[name+'_S']+item[name+'_D']-item['E1_S']-item['E1_D']
        syllable_rows.append(item)
    table('target_syllables.csv',[r for r in syllable_rows if r['is_target']]);table('non_target_syllables.csv',[r for r in syllable_rows if not r['is_target']])
    for target in (True,False):
        for name in results:
            subset=[r for r in syllable_rows if r['is_target']==target];n=sum(r['validation_reference_count'] for r in subset)
            ss=sum(r[name+'_S'] for r in subset);dd=sum(r[name+'_D'] for r in subset)
            groups.append(dict(group='target' if target else 'non_target',experiment=name,reference_count=n,S=ss,D=dd,
                substitution_rate=ss/n if n else None,deletion_rate=dd/n if n else None,error_rate=(ss+dd)/n if n else None))
    table('syllable_groups.csv',groups)
    patterns=[]
    for p in repeated:
        a,b=p['reference_syllable'],p['predicted_syllable']
        item=dict(reference_syllable=a,predicted_syllable=b,train_pair_count=p['count'],train_pair_ratio=p['ratio'],
            train_pair_utterances=len(pair_support[(a,b)]),
            validation_reference_count=charstats['E1'].get(a,{}).get('N',0))
        for name in results:item[name+'_pair_count']=patternstats[name][(a,b)]
        for name in ('E4-A','E4-B'):item[name+'_pair_delta']=item[name+'_pair_count']-item['E1_pair_count']
        patterns.append(item)
    table('repeated_substitution_patterns.csv',patterns)
    flat=[]
    for r in summaries:
        c=changes.get(r['experiment'],{});flat.append({k:v for k,v in r.items() if k!='counts'}|dict(cer_delta_pp=100*(r['validation_cer']-summaries[0]['validation_cer']),
            wer_delta_pp=100*(r['validation_wer']-summaries[0]['validation_wer']),changed=c.get('changed',0),improved=c.get('improved',0),worsened=c.get('worsened',0),equal=c.get('equal',63)))
    table('validation_summary.csv',flat)
    epochs=[]
    for name in results:
        audit=read(OUT/name/'training_audit.json')
        for item in audit['epochs']:
            epochs.append(dict(experiment=name,epoch=item['epoch'],step=item['step'],asr_loss=item['asr_loss'],
                weighted_loss=item['weighted_loss'],validation_cer=item['validation']['eval_cer'],validation_wer=item['validation']['eval_wer']))
    table('epochs.csv',epochs)
    ksm=[]
    with (ROOT/'reports/e4_experiment_20261107/comparison.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):ksm.append(dict(speaker='KSM_M_45',experiment=r['experiment'],best_epoch=r['best_epoch'],validation_cer=float(r['validation_cer']),validation_wer=float(r['validation_wer'])))
    side=ksm+[{k:r[k] for k in ('speaker','experiment','best_epoch','validation_cer','validation_wer')} for r in summaries]
    table('ksm_sjw_comparison.csv',side)
    write('comparison.json',dict(summary=flat,changes=changes,syllable_groups=groups,repeated_patterns=patterns,
        side_by_side=side,weighted_syllable_types=len(targets),weighted_syllables_present=sum(r['is_target'] and r['validation_reference_count']>0 for r in syllable_rows),
        personal_test_evaluated=False,statistics_76_speakers='deferred at explicit user request'))
    lines=['# SJW_M_29 E1/E4-A/E4-B 개인화 Validation 결과','',
        '사용자 요청에 따라 순위 CSV 검색 및 76명 전체 화자 통계는 보류했다. 이번에는 SJW_M_29 한 명의 세 실험과 Validation만 실행했다. 기존 KSM 결과는 읽기만 했다.','',
        '## 데이터 및 설정','',
        '현재 원본 분할은 Train190/Validation63/Test61이다. 모든 WAV 접근·헤더 길이·토큰 길이 검사를 통과했다. Train 음성은 643초다. 학습은 원본 Train 190개를 그대로 사용했다. Strong 의심 6개를 제외한 184개/13.549%는 화자 선정 통계이며, 이번 원본 Train 기반 실험 지표와 구분한다. Strong mismatch는 확정 라벨 오류가 아닌 불일치 의심 기준이다.',
        'General Small 동일 초기 가중치, AdamW torch/LR1e-5/linear scheduler/warmup0/weight_decay0/clip1/epoch5/batch2/accum4/seed42/data_seed=None/full encoder+decoder/BF16/기존 legacy scaling을 유지했다. 생성은 korean/transcribe/max_length160/beam1이고 동일 E1 평가 함수와 정규화를 사용했다. 저장만 Best 모델 하나로 제한했으며 optimizer resume 상태는 저장하지 않았다.',
        'E4-A는 정규화 기준 오류 Train 21개에 1.5를 적용했다. E4-B는 이 190개의 정답·General 예측에서 새 Profile을 생성했으며 KSM과 같은 n>=5,e>=2,오류발화>=2, prior strength10, 상한1.5, byte 겹침 정책을 사용했다. 새 일반화 adapter는 KSM의 기존 가중치와 정확히 일치했다.','',
        '## Best 및 저장 모델 재평가','',
        '| 모델 | Best Epoch | Validation CER | WER | E1 대비 CER %p | E1 대비 WER %p | 학습 초 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in flat:lines.append(f"| {r['experiment']} | {r['best_epoch']:g} | {100*r['validation_cer']:.4f}% | {100*r['validation_wer']:.4f}% | {r['cer_delta_pp']:+.4f} | {r['wer_delta_pp']:+.4f} | {r['training_seconds']:.2f} |")
    lines += ['', '| E1 대비 | 정규화 예측 변경 | CER 개선/악화/동일 | WER 개선/악화/동일 |','|---|---:|---|---|']
    for name,c in changes.items():lines.append(f"| {name} | {c['changed']} | {c['improved']}/{c['worsened']}/{c['equal']} | {c['wer_improved']}/{c['wer_worsened']}/{c['wer_equal']} |")
    coverage=preflight['weight_info']['counts']
    lines += ['', f"E4-B는 51종 음절을 기준으로 Train {coverage['b_weighted_utterances']}/{coverage['train_utterances']}개 발화, content token {coverage['b_weighted_content_tokens']}/{coverage['content_tokens']}개({100*coverage['b_weighted_content_tokens']/coverage['content_tokens']:.4f}%)에 가중치를 적용했다. 같은 음절의 정상 인식 문맥에도 가중치가 적용된다. 두 가중 방법 모두 유효 토큰 전체에서 sum(w*CE)/sum(w)를 사용하며, 전부 가중치1이면 E1 token-mean CE로 돌아간다."]
    lines += ['', 'Best 선택은 학습 중 Validation CER 기준이다. 표의 CER/WER는 저장된 Best를 같은 63개에 재로딩해 평가한 값이며, 학습 로그와 일치 여부는 validation_summary.csv에 별도 기록했다.','',
        '## 대상/비대상 한글 음절 오류','',
        '| 집단 | 모델 | 정답 등장 | 치환 | 삭제 | (치환+삭제)/등장 |','|---|---|---:|---:|---:|---:|']
    for r in groups:lines.append(f"| {r['group']} | {r['experiment']} | {r['reference_count']} | {r['S']} | {r['D']} | {100*r['error_rate']:.4f}% |")
    lines += ['', '음절 오류는 기존 Profile의 한글-only 정렬이며 숫자를 포함하는 전체 CER와 다르다. 삽입은 대응 정답 음절이 없어 위 참조 음절 오류율에 합치지 않았다. 대상 51종 전체(미등장 포함)는 target_syllables.csv, 비대상은 non_target_syllables.csv에 저장했다.',
        '반복 치환은 이 화자의 원본 Train에서 같은 정답→예측 쌍이 2회 이상 나온 경우로 학습 전에 정의했다. Validation 정답 등장 수와 동일 쌍의 실제 오류 수를 repeated_substitution_patterns.csv에 기록했다. Val에 정답 음절이 없으면 개선 여부를 판단할 수 없다.','',
        '## KSM_M_45와 나란히 비교','',
        '| 화자 | 모델 | Best Epoch | Validation CER | WER |','|---|---|---:|---:|---:|']
    for r in side:lines.append(f"| {r['speaker']} | {r['experiment']} | {r['best_epoch']} | {100*r['validation_cer']:.4f}% | {100*r['validation_wer']:.4f}% |")
    lines += ['', '## 반복 치환 관측 결과','',
        f"Train 반복 치환 쌍은 {len(patterns)}개이고, 그 정답 음절이 Validation에 등장한 쌍은 {sum(p['validation_reference_count']>0 for p in patterns)}개다. 같은 발화 안의 반복도 count에 포함되므로 CSV에 별도 Train 오류 발화 수를 기록했다.",
        '| 모델 | Validation의 동일 반복 치환 오류 합계 |','|---|---:|']
    for name in results:lines.append(f"| {name} | {sum(p[name+'_pair_count'] for p in patterns)} |")
    for name in ('E4-A','E4-B'):
        lines.append(f"- {name}: 쌍별 오류 감소 {sum(p[name+'_pair_delta']<0 for p in patterns)}개, 증가 {sum(p[name+'_pair_delta']>0 for p in patterns)}개, 동일 {sum(p[name+'_pair_delta']==0 for p in patterns)}개. Validation에서 관측되지 않은 쌍의 동일 0회는 개선으로 해석하지 않는다.")
    lines += ['', '화자별 Validation은 서로 다른 데이터이므로 화자 간 절대 지표 차이를 방법의 효과로 해석하지 않는다. 방법 간 효과는 같은 화자의 E1 대비 변화로 비교한다. 한 seed·두 화자의 탐색적 결과이며 일반화 효과를 단정하지 않는다.','',
        '## Best 경로','']
    total_ce=sum(r['ce'] for r in error_order);top_ce=sum(r['ce'] for r in error_order[:3])
    lines += [f"E1 Validation 문자 오류 {total_ce}건 중 오류 수 상위 3개 발화가 {top_ce}건을 차지한다. 발화별 원문·예측·오류 수는 validation_error_concentration.csv에서 확인할 수 있다. 이 관측만으로 전사 오류 여부를 확정하지 않는다.",'']
    for r in flat:lines.append(f"- {r['experiment']}: `{r['best_checkpoint']}`")
    lines += ['', '학습 실패/NaN/Inf·시간·GPU 메모리는 각 실험 training_audit.json 및 console log에 기록했다. Personal Test 추론·평가는 실행하지 않았다. 다른 화자의 추가 학습이나 하이퍼파라미터 변경도 하지 않았다.']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    append_notes()
    print(json.dumps(dict(summary=flat,changes=changes,groups=groups),indent=2))


def append_notes():
    data=read(REPORT/'comparison.json')
    with (REPORT/'utterance_comparison.csv').open(encoding='utf-8-sig',newline='') as f:
        changed=[r for r in csv.DictReader(f) if r['changed']=='True']
    table('changed_prediction_examples.csv',changed)
    heading='## 최종 비교 해석'
    text=(REPORT/'report.md').read_text(encoding='utf-8').split(heading)[0].rstrip()
    notes=['',heading,'',
        f"대상 51종 중 Validation에 실제 등장한 음절은 {data['weighted_syllables_present']}종이다. 대상 정답 315회에서 E1/E4-A의 치환+삭제는 97건, E4-B는 100건으로 3건 늘었다. 비대상 정답 699회에서는 160→161건으로 1건 늘었다. 대상 중 는/야/이의 치환이 각각 1건 증가했고, 나머지 대상 음절의 치환·삭제 수는 같았다.",
        'E1/E4-A는 동일한 63개 예측을 생성했다. E4-B는 1개 발화에서 문자 오류가 11→15건, 단어 오류가 4→5건으로 증가했다. 전체 CER +0.3945%p와 WER +0.2890%p는 이 한 발화의 차이로 설명된다. 개선·악화가 상쇄된 결과가 아니다.','',
        '| 정답 | E1/E4-A 예측 | E4-B 예측 |','|---|---|---|']
    for r in changed:notes.append(f"| {r['reference']} | {r['e1_prediction']} | {r['prediction']} |")
    notes += ['', '위 오류는 제공된 Validation 전사 기준이다. 이번 작업에서는 음성을 직접 청취해 전사를 검증하지 않았으므로, 높은 오류율이나 Strong 플래그를 확정 전사 오류로 해석하지 않는다.',
        'KSM_M_45에서는 세 모델의 Best 지표가 동일했고, SJW_M_29에서는 E4-A가 동일하고 E4-B가 소폭 악화됐다. 이번 두 화자·한 seed 결과에서 E1 대비 Validation 개선은 관측되지 않았다. 다른 화자 추가 학습이나 하이퍼파라미터 탐색은 수행하지 않았다.',
        '보존 확인은 preservation.json에, 세 학습 설정 26개 항목의 일치는 conditions_check.json에 기록했다. 새 구현 파일은 src/personalization/speaker_weighted.py와 scripts/07_personalization/sjw_experiment.py이다. 기존 E1/E2/E3/E4 및 완료된 KSM 결과는 수정하지 않았다.']
    (REPORT/'report.md').write_text(text+'\n'+'\n'.join(notes)+'\n',encoding='utf-8')


if __name__=='__main__':main()
