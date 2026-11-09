"""Read-only saved-model/Validation comparison; all writes stay in this report."""
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import numpy as np
import torch
import yaml
from safetensors import safe_open
from src.personalization.trainer import build_model_and_processor,make_trainer
from src.personalization.dataset import load_speaker_splits,WhisperSpeechDataset
from src.personalization.evaluator import decode_predictions,metric_counts,normalize_text
from src.personalization.error_profile import align_syllables,normalize_for_syllable_alignment

OUT=Path(__file__).resolve().parent
MODELS={
    'E1':ROOT/'outputs/personalization/E1_v2/KSM_M_45/checkpoints/checkpoint-42',
    'E4-A':ROOT/'outputs/personalization/E4/KSM_M_45/E4-A/checkpoints/checkpoint-42',
    'E4-B':ROOT/'outputs/personalization/E4/KSM_M_45/E4-B/checkpoints/checkpoint-42',
}


def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def parameter_comparison():
    summaries=[];details=[]
    for left,right in [('E1','E4-A'),('E1','E4-B'),('E4-A','E4-B')]:
        with safe_open(MODELS[left]/'model.safetensors',framework='pt',device='cpu') as a, \
             safe_open(MODELS[right]/'model.safetensors',framework='pt',device='cpu') as b:
            assert set(a.keys())==set(b.keys())
            sums={name:Counter() for name in ('all','encoder','decoder')}
            for name in a.keys():
                x=a.get_tensor(name).reshape(-1);y=b.get_tensor(name).reshape(-1)
                assert x.shape==y.shape
                delta2=norm2=other2=dot=0.;maxabs=0.;changed=0
                for start in range(0,x.numel(),1048576):
                    xx=x[start:start+1048576].double();yy=y[start:start+1048576].double();diff=yy-xx
                    delta2+=float(diff.square().sum());norm2+=float(xx.square().sum());other2+=float(yy.square().sum())
                    dot+=float((xx*yy).sum());changed+=int(torch.count_nonzero(diff));maxabs=max(maxabs,float(diff.abs().max()))
                row=dict(left=left,right=right,tensor=name,elements=x.numel(),changed_elements=changed,
                    difference_l2=math.sqrt(delta2),relative_l2=math.sqrt(delta2/norm2) if norm2 else None,max_abs_difference=maxabs)
                details.append(row)
                for group in ('all','encoder' if name.startswith('model.encoder.') else 'decoder'):
                    v=sums[group];v.update(elements=x.numel(),changed_elements=changed,tensors=1,changed_tensors=int(changed>0),
                        delta2=delta2,norm2=norm2,other2=other2,dot=dot);v['maxabs']=max(v['maxabs'],maxabs)
                del x,y
            for group,v in sums.items():
                summaries.append(dict(left=left,right=right,group=group,tensors=v['tensors'],changed_tensors=v['changed_tensors'],
                    parameters=v['elements'],changed_parameters=v['changed_elements'],changed_fraction=v['changed_elements']/v['elements'],
                    difference_l2=math.sqrt(v['delta2']),baseline_l2=math.sqrt(v['norm2']),
                    relative_l2=math.sqrt(v['delta2']/v['norm2']),rms_difference=math.sqrt(v['delta2']/v['elements']),
                    max_abs_difference=v['maxabs'],cosine=v['dot']/math.sqrt(v['norm2']*v['other2'])))
        print('PARAMETERS',left,right,flush=True)
    table('parameter_comparison.csv',summaries);table('parameter_tensor_details.csv',details)
    return summaries


def counts_alignment(ref,hyp):
    """Evaluation characters including digits; existing DP tie-break, no Hangul filter."""
    result=Counter()
    for a,b in align_syllables(list(ref.replace(' ','')),list(hyp.replace(' ',''))):
        if a==b:continue
        result['I' if a=='<EMPTY>' else 'D' if b=='<EMPTY>' else 'S']+=1
    return {k:result[k] for k in ('S','D','I')}


def validation_comparison():
    cfg=yaml.safe_load((ROOT/'scripts/07_personalization/configs/e1.yaml').read_text(encoding='utf-8'))
    rows=load_speaker_splits(cfg['paths']['manifest_csv'],'KSM_M_45',cfg['columns'],cfg['split_labels'],cfg['paths']['audio_root'],cfg['recording_id_column'])['val']
    assert len(rows)==65
    results={};model_metrics={}
    for name,path in MODELS.items():
        model,processor=build_model_and_processor(path)
        ds=WhisperSpeechDataset(rows,processor,'text',max_audio_seconds=30.,max_label_length=160)
        # Same E1 prediction pipeline and CUDA BF16 configuration as training validation.
        trainer=make_trainer(model,processor,ds,ds,OUT/f'{name}_runtime',cfg['training'])
        predicted=trainer.predict(ds,metric_key_prefix='validation')
        refs,hyps=decode_predictions(predicted.predictions,predicted.label_ids,processor.tokenizer)
        raw=processor.tokenizer.batch_decode(predicted.predictions,skip_special_tokens=True)
        records=[];totals=Counter()
        for row,ref,hyp,rawhyp,ids in zip(rows,refs,hyps,raw,predicted.predictions):
            assert normalize_for_syllable_alignment(row['text'])==normalize_for_syllable_alignment(ref)
            ce,nc,we,nw=metric_counts(ref,hyp);edits=counts_alignment(ref,hyp)
            assert sum(edits.values())==ce
            totals.update(ce=ce,nc=nc,we=we,nw=nw,**edits)
            records.append(dict(batch=row['batch'],audio_member=row['audio_member'],recording_id=row['recording_id'],
                reference=ref,raw_reference=row['text'],hypothesis=hyp,raw_hypothesis=rawhyp,ce=ce,nc=nc,we=we,nw=nw,
                cer=ce/nc if nc else 0.,wer=we/nw if nw else 0.,**edits,token_ids=ids.tolist()))
        results[name]=records
        model_metrics[name]=dict(cer=totals['ce']/totals['nc'],wer=totals['we']/totals['nw'],counts=dict(totals),
            trainer_metrics=predicted.metrics,precision='E1 Seq2SeqTrainer CUDA BF16 prediction',generation_config=model.generation_config.to_dict())
        assert abs(model_metrics[name]['cer']-predicted.metrics['validation_cer'])<1e-12
        with (OUT/f'{name}_validation_predictions.jsonl').open('w',encoding='utf-8') as f:
            for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print('VALIDATION',name,model_metrics[name]['cer'],model_metrics[name]['wer'],flush=True)
        del model,trainer,predicted,ds;gc.collect();torch.cuda.empty_cache()
    comparisons=[];changes={}
    for name in ('E4-A','E4-B'):
        c=Counter()
        for a,b in zip(results['E1'],results[name]):
            assert (a['batch'],a['audio_member'],a['reference'])==(b['batch'],b['audio_member'],b['reference'])
            outcome='improved' if b['ce']<a['ce'] else 'worsened' if b['ce']>a['ce'] else 'equal'
            word_outcome='improved' if b['we']<a['we'] else 'worsened' if b['we']>a['we'] else 'equal'
            c.update({outcome:1,'wer_'+word_outcome:1,'raw_changed':int(a['raw_hypothesis']!=b['raw_hypothesis']),
                'normalized_changed':int(a['hypothesis']!=b['hypothesis']),'token_ids_changed':int(a['token_ids']!=b['token_ids']),
                'space_removed_changed':int(a['hypothesis'].replace(' ','')!=b['hypothesis'].replace(' ',''))})
            comparisons.append(dict(model=name,batch=a['batch'],audio_member=a['audio_member'],reference=a['reference'],
                e1_prediction=a['hypothesis'],e4_prediction=b['hypothesis'],raw_changed=a['raw_hypothesis']!=b['raw_hypothesis'],
                normalized_changed=a['hypothesis']!=b['hypothesis'],e1_cer=a['cer'],e4_cer=b['cer'],cer_change=b['cer']-a['cer'],
                e1_wer=a['wer'],e4_wer=b['wer'],wer_change=b['wer']-a['wer'],cer_outcome=outcome,wer_outcome=word_outcome,
                e1_S=a['S'],e4_S=b['S'],e1_D=a['D'],e4_D=b['D'],e1_I=a['I'],e4_I=b['I']))
        changes[name]={key:c[key] for key in ('improved','worsened','equal','wer_improved','wer_worsened','wer_equal',
            'raw_changed','normalized_changed','token_ids_changed','space_removed_changed')}
    table('utterance_comparison.csv',comparisons)
    selected={s:w for s,w in read(ROOT/'outputs/personalization/E4/KSM_M_45/E4-B/provenance.json')['syllable_weights'].items() if w>1}
    assert len(selected)==23
    profile={name:{s:Counter() for s in selected} for name in MODELS}
    for name,records in results.items():
        for record in records:
            for a,b in align_syllables(normalize_for_syllable_alignment(record['reference']),normalize_for_syllable_alignment(record['hypothesis'])):
                if a in selected:
                    profile[name][a]['N']+=1
                    profile[name][a]['C' if a==b else 'D' if b=='<EMPTY>' else 'S']+=1
                if a=='<EMPTY>' and b in selected:profile[name][b]['I_predicted']+=1
    syllable_rows=[]
    for s,w in sorted(selected.items()):
        item=dict(syllable=s,train_syllable_weight=w,validation_reference_count=profile['E1'][s]['N'])
        for name in MODELS:
            v=profile[name][s];assert v['N']==item['validation_reference_count']
            for key in ('C','S','D','I_predicted'):item[name+'_'+key]=v[key]
            item[name+'_error_rate']=(v['S']+v['D'])/v['N'] if v['N'] else None
        for name in ('E4-A','E4-B'):
            item[name+'_reference_error_change']=sum(profile[name][s][k]-profile['E1'][s][k] for k in ('S','D'))
        syllable_rows.append(item)
    table('weighted_syllable_validation.csv',syllable_rows)
    return model_metrics,changes,syllable_rows


def report(parameters,metrics,changes,syllables):
    lines=['# KSM_M_45 E1/E4-A/E4-B Best 체크포인트 진단','',
        '저장된 Epoch 2 모델 세 개만 비교했다. 추가 학습·optimizer update·Personal Test 평가는 실행하지 않았다. 모든 산출물은 이 신규 보고서 디렉터리에 저장했다.','',
        '## 1. 모델 파라미터 차이','',
        '| 기준 → 비교 | 차이 L2 | 상대 L2 | 최대 절대 차이 | 달라진 파라미터 비율 |',
        '|---|---:|---:|---:|---:|']
    for r in parameters:
        if r['group']=='all':lines.append(f"| {r['left']} → {r['right']} | {r['difference_l2']:.8f} | {r['relative_l2']:.8g} | {r['max_abs_difference']:.8g} | {100*r['changed_fraction']:.4f}% |")
    lines += ['', 'safetensors에 직렬화된 공통 텐서를 float64 누산으로 원소별 비교했다. 상대 L2는 차이 norm/기준 모델 norm이다. tied proj_out/decoder embedding은 저장된 가중치 하나로 계산해 중복 집계하지 않았다. 정확히 다른 원소 수를 사용하므로 극소 차이도 포함한다. Encoder/Decoder 및 tensor별 세부 결과는 parameter_comparison.csv와 parameter_tensor_details.csv에 있다.','',
        '## 2. 동일 Validation 65개 예측','',
        '| 모델 | CER | WER | 문자 오류 S/D/I | 문자 오류/정답 문자 |',
        '|---|---:|---:|---|---|']
    for name,m in metrics.items():
        c=m['counts'];lines.append(f"| {name} | {100*m['cer']:.4f}% | {100*m['wer']:.4f}% | {c['S']}/{c['D']}/{c['I']} | {c['ce']}/{c['nc']} |")
    lines += ['', '| E1 대비 | 원문 예측 변경 | 정규화 예측 변경 | CER 개선/악화/동일 | WER 개선/악화/동일 |','|---|---:|---:|---|---|']
    for name,c in changes.items():
        lines.append(f"| {name} | {c['raw_changed']} | {c['normalized_changed']} | {c['improved']}/{c['worsened']}/{c['equal']} | {c['wer_improved']}/{c['wer_worsened']}/{c['wer_equal']} |")
    lines += ['', '기존 trainer.make_trainer → Seq2SeqTrainer.predict와 evaluator.decode_predictions/compute_asr_metrics/metric_counts를 그대로 사용했다. 화자 manifest 순서, batch2, CUDA BF16, korean/transcribe, max_length160, beam1을 유지했다. 모델은 eval/no_grad 경로에서만 실행했다. CER는 정규화 후 공백 제거, WER는 정규화 후 공백 단위이며 숫자는 보존된다.','']
    for name,c in changes.items():
        if c['raw_changed']==0:
            reason='65개 발화의 원문 예측 자체가 모두 같다. 개선과 악화의 상쇄가 아니다.'
        elif c['normalized_changed']==0:
            reason='원문 차이는 있으나 평가 정규화 후 예측이 모두 같다. 정규화가 차이를 제거한 경우다.'
        elif c['improved']+c['worsened']>0:
            reason='예측이 달라지고 발화별 오류도 달라진다. 전체 동일 지표는 개선·악화의 합을 함께 확인해야 한다.'
        else:reason='예측 차이는 있으나 발화별 문자 오류 수는 같다. 동일 오류 수의 다른 예측 또는 공백 차이를 확인해야 한다.'
        lines.append(f'- {name}: {reason}')
    lines += ['', '## 3. E4-B 가중 음절 23종의 Validation 오류','',
        '| 음절 | 정답 등장 | E1 S/D | E4-A S/D | E4-B S/D | E4-B 오류 증감 |',
        '|---|---:|---|---|---|---:|']
    for r in syllables:
        lines.append(f"| {r['syllable']} | {r['validation_reference_count']} | {r['E1_S']}/{r['E1_D']} | {r['E4-A_S']}/{r['E4-A_D']} | {r['E4-B_S']}/{r['E4-B_D']} | {r['E4-B_reference_error_change']:+d} |")
    lines += ['', f"23종 중 Validation에 등장한 종류는 {sum(r['validation_reference_count']>0 for r in syllables)}종, 정답 등장 횟수 합계는 {sum(r['validation_reference_count'] for r in syllables)}회다.",
        '이 표는 기존 Error Profile의 normalize_for_syllable_alignment와 align_syllables를 사용한 한글-only 정렬이다. 음절별 오류율=(치환+삭제)/해당 정답 음절 수이며, 미등장 음절의 오류율은 NA다. 삽입은 대응 정답 음절이 없어 분자에 합치지 않고 CSV의 I_predicted에 예측 음절별로 별도 표시했다. 이 부분의 Hangul-only 집계는 숫자를 포함하는 전체 평가 CER와 구분해야 한다. 정렬 동률은 기존 함수의 정책을 유지하며, 이는 음향 정렬이 아니다.','',
        '## 4. 원인 해석과 범위','',
        '파라미터가 다르면 학습 결과가 같았던 것은 아니다. 작은 가중치 변경으로 연속적인 모델 파라미터는 달라져도 동일한 유한 Validation 문장에서 greedy Decoder가 같은 토큰 경로를 선택할 수 있다. 이 비교는 파라미터와 생성 결과에 근거하며, logits나 토큰 margin은 추가 측정하지 않았으므로 결정 경계가 얼마나 이동했는지는 정량화하지 않는다.',
        '새로운 실험 방법·하이퍼파라미터 탐색은 제안하거나 실행하지 않았다. 이 결과는 저장된 Best 모델과 해당 65발화에 한정되며 Test나 다른 화자의 효과로 확대 해석하지 않는다.','',
        '## 산출물 및 보존','',
        '- parameter_comparison.csv / parameter_tensor_details.csv: 모델·블록·텐서별 파라미터 차이.',
        '- E1_validation_predictions.jsonl, E4-A_validation_predictions.jsonl, E4-B_validation_predictions.jsonl: 동일 65발화의 원문/정규화 예측 및 토큰 ID.',
        '- utterance_comparison.csv: 발화별 예측, CER/WER와 S/D/I 변화.',
        '- weighted_syllable_validation.csv: 가중 음절 23종 전체의 관측·정오답·삽입 통계.',
        '- summary.json / input_checksums.json / preservation.json: 요약과 원본 보존 확인.','']
    for name,path in MODELS.items():lines.append(f'- {name} Best: `{path}`')
    # Summarize the targeted positions without treating insertions as reference errors.
    targeted=[]
    for name in MODELS:
        targeted.append(dict(model=name,N=sum(r['validation_reference_count'] for r in syllables),
            C=sum(r[name+'_C'] for r in syllables),S=sum(r[name+'_S'] for r in syllables),
            D=sum(r[name+'_D'] for r in syllables),I_predicted=sum(r[name+'_I_predicted'] for r in syllables)))
    lines += ['', '## 가중 음절 집계 보충','',
        '| 모델 | 정답 등장 | 정인식 | 치환 | 삭제 | 가중 음절이 삽입된 횟수 |','|---|---:|---:|---:|---:|---:|']
    for r in targeted:lines.append(f"| {r['model']} | {r['N']} | {r['C']} | {r['S']} | {r['D']} | {r['I_predicted']} |")
    lines += ['', '세 모델 모두 가중 음절의 치환은 어 2회, 운 2회, 줘 2회, 해 1회이며 삭제는 없다. ‘세’ 음절 삽입은 1회로 별도 집계됐다. 알은 Validation 정답에 없었다. E1이 이미 해당 정답 위치 108회 중 101회를 맞혔다는 관측과 낮은 관측 횟수를 함께 해석해야 한다. 이 집계는 같은 예측 결과의 부분 분석이며 새로운 추론 조건이나 실험이 아니다.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    torch.set_num_threads(4)
    inputs=[]
    for path in MODELS.values():inputs += [p for p in path.rglob('*') if p.is_file()]
    inputs += [ROOT/'data/splits/personalization_split.csv',ROOT/'scripts/07_personalization/configs/e1.yaml',
        ROOT/'src/personalization/trainer.py',ROOT/'src/personalization/dataset.py',ROOT/'src/personalization/evaluator.py',
        ROOT/'src/personalization/error_profile.py',ROOT/'src/personalization/e4_weighted.py',
        ROOT/'outputs/personalization/E4/KSM_M_45/E4-B/provenance.json',
        ROOT/'reports/e4_experiment_20261107/comparison.csv',ROOT/'reports/e4_experiment_20261107/report.md']
    before={str(p):sha(p) for p in inputs};write('input_checksums.json',before)
    parameters=parameter_comparison()
    metrics,changes,syllables=validation_comparison()
    write('summary.json',dict(models={k:str(v) for k,v in MODELS.items()},parameter_comparison=parameters,
        validation_metrics=metrics,prediction_changes=changes,weighted_syllables=syllables,
        training_executed=False,personal_test_evaluated=False))
    report(parameters,metrics,changes,syllables)
    changed=[str(p) for p in inputs if sha(p)!=before[str(p)]]
    assert not changed,changed
    write('preservation.json',dict(all_inputs_unchanged=True,files_verified=len(before),training_executed=False,personal_test_evaluated=False))
    print(json.dumps(dict(metrics={k:dict(cer=v['cer'],wer=v['wer'],counts=v['counts']) for k,v in metrics.items()},changes=changes),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
