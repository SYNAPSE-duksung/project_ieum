"""Saved Best checkpoints only: KSM Test comparison. No train() call."""
import csv,gc,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import torch,yaml
from transformers import set_seed
from src.personalization.dataset import load_speaker_splits,WhisperSpeechDataset
from src.personalization.trainer import build_model_and_processor,make_trainer
from src.personalization.evaluator import compute_asr_metrics,decode_predictions,metric_counts,normalize_text,save_predictions

DEST=ROOT/'outputs/personalization/evaluation/KSM_M_45/E4_test_20261108'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,obj):(DEST/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(name,rows):
    with (DEST/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    if DEST.exists():raise FileExistsError('Refusing overwrite: '+str(DEST))
    config_path=ROOT/'scripts/07_personalization/configs/e1.yaml';cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    rows=load_speaker_splits(cfg['paths']['manifest_csv'],'KSM_M_45',cfg['columns'],cfg['split_labels'],cfg['paths']['audio_root'],cfg['recording_id_column'])
    assert {k:len(v) for k,v in rows.items()}==dict(train=168,val=65,test=67)
    keys={s:{(r['batch'],r['audio_member'].replace('\\','/')) for r in subset} for s,subset in rows.items()}
    wavs={s:{Path(r['_resolved_audio']).resolve() for r in subset} for s,subset in rows.items()}
    assert not keys['test']&(keys['train']|keys['val']) and not wavs['test']&(wavs['train']|wavs['val'])
    audio_hash={s:{sha(r['_resolved_audio']) for r in subset} for s,subset in rows.items()}
    assert not audio_hash['test']&(audio_hash['train']|audio_hash['val']), 'Cross-split byte-identical WAV'
    old_root=ROOT/'outputs/personalization/E1_v2/KSM_M_45'
    e1=read(old_root/'run_summary.json')
    e4={n:read(ROOT/f'outputs/personalization/E4/KSM_M_45/{n}/summary.json') for n in ('E4-A','E4-B')}
    models={'E1':Path(e1['best_checkpoint']),**{n:Path(v['best_checkpoint']) for n,v in e4.items()}}
    assert sha(models['E1']/'model.safetensors')==sha(old_root/'best_model/model.safetensors')
    for n,v in e4.items():
        state=read(ROOT/f'outputs/personalization/E4/KSM_M_45/{n}/trainer_state.json')
        assert Path(state['best_model_checkpoint']).resolve()==models[n].resolve()
        assert state['best_metric']==v['best_validation_cer']
    historical=[json.loads(line) for line in (old_root/'test_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    old_predictions={str(Path(r['audio_path']).resolve()):r for r in historical}
    assert len(old_predictions)==67 and set(old_predictions)=={str(p) for p in wavs['test']}
    assert all(normalize_text(r['text'])==old_predictions[str(Path(r['_resolved_audio']).resolve())]['reference'] for r in rows['test'])
    protected=[config_path,Path(cfg['paths']['manifest_csv']),old_root/'run_summary.json',old_root/'test_metrics.json',old_root/'test_predictions.jsonl']
    protected += [ROOT/f'src/personalization/{n}.py' for n in ('dataset','trainer','evaluator')]
    for folder in [*models.values(),ROOT/'outputs/personalization/E4/KSM_M_45/E4-A',ROOT/'outputs/personalization/E4/KSM_M_45/E4-B']:
        protected += [p for p in folder.rglob('*') if p.is_file()]
    protected += [Path(r['_resolved_audio']) for r in rows['test']]
    before={str(p):sha(p) for p in set(protected)}
    DEST.mkdir(parents=True)
    plan=dict(speaker='KSM_M_45',manifest=str(Path(cfg['paths']['manifest_csv']).resolve()),audio_root=cfg['paths']['audio_root'],
        models={n:str(p) for n,p in models.items()},test_count=67,split_counts={k:len(v) for k,v in rows.items()},
        cross_split_utterance_overlap=0,cross_split_path_overlap=0,cross_split_audio_hash_overlap=0,
        historical_e1_test_ids_and_references_match=True,recording_id_disjoint_required=False,
        note='Original split is at utterance level; shared source recordings do not imply copied Train examples.',
        generation='original E1 factory: korean/transcribe, max_length160, beam1, do_sampleFalse; eval batch2; CUDA precision selected by original factory',
        training_executed=False,test_items=[dict(batch=r['batch'],audio_member=r['audio_member'],wav_path=r['_resolved_audio']) for r in rows['test']])
    write('preflight.json',plan);write('input_checksums.json',before)
    print('EVALUATION CHECKPOINTS AND DATA',json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    results={};metrics={};historical_check={}
    for name,path in models.items():
        set_seed(int(cfg['training']['seed']))
        model,processor=build_model_and_processor(path,cfg['language'],cfg['task'])
        ds=WhisperSpeechDataset(rows['test'],processor,cfg['columns']['text'],max_audio_seconds=30.,max_label_length=160)
        # These dataset arguments are placeholders; train/evaluate are never called.
        # predict receives the explicitly validated Test dataset only.
        trainer=make_trainer(model,processor,ds,ds,DEST/name/'trainer_runtime',cfg['training'])
        write(name+'_evaluation_arguments.json',trainer.args.to_dict())
        prediction=trainer.predict(ds,metric_key_prefix='test')
        m=compute_asr_metrics(prediction.predictions,prediction.label_ids,processor.tokenizer)
        refs,hyps=decode_predictions(prediction.predictions,prediction.label_ids,processor.tokenizer)
        raw=processor.tokenizer.batch_decode(prediction.predictions,skip_special_tokens=True)
        save_predictions(DEST/name,rows['test'],refs,hyps,m)
        records=[];counts=[0,0,0,0]
        for row,ref,hyp,rhyp in zip(rows['test'],refs,hyps,raw):
            assert ref==normalize_text(row['text'])
            ce,nc,we,nw=metric_counts(ref,hyp)
            counts=[a+b for a,b in zip(counts,[ce,nc,we,nw])]
            records.append(dict(reference=ref,prediction=hyp,raw_prediction=rhyp,cer=ce/nc if nc else 0.,wer=we/nw if nw else 0.,ce=ce,nc=nc,we=we,nw=nw))
        metrics[name]=dict(**m,character_errors=counts[0],reference_characters=counts[1],word_errors=counts[2],reference_words=counts[3])
        results[name]=records
        if name=='E1':
            old=read(old_root/'test_metrics.json')
            historical_check=dict(metrics_match=old==m,changed_predictions=sum(r['prediction']!=old_predictions[str(Path(row['_resolved_audio']).resolve())]['hypothesis'] for row,r in zip(rows['test'],records)))
            write('historical_e1_check.json',historical_check)
            if not historical_check['metrics_match'] or historical_check['changed_predictions']:
                raise RuntimeError('Reevaluated E1 differs from historical Test result; stop comparison and report')
        print('RESULT',name,json.dumps(metrics[name]),flush=True)
        del trainer,model,prediction,ds;gc.collect();torch.cuda.empty_cache()
    comparison=[];changes={n:dict(changed=0,raw_changed=0,improved=0,worsened=0,equal=0,wer_improved=0,wer_worsened=0,wer_equal=0) for n in ('E4-A','E4-B')}
    for i,row in enumerate(rows['test']):
        base=results['E1'][i]
        item=dict(utterance_id=Path(row['audio_member']).stem,speaker_id='KSM_M_45',batch=row['batch'],recording_id=row['recording_id'],
            audio_member=row['audio_member'],wav_path=row['_resolved_audio'],reference=base['reference'])
        for n in models:
            r=results[n][i];item[n+'_prediction']=r['prediction'];item[n+'_CER']=r['cer'];item[n+'_WER']=r['wer']
            item[n+'_character_errors']=r['ce'];item[n+'_word_errors']=r['we']
            if n!='E1':
                changed=r['prediction']!=base['prediction'];outcome='improved' if r['ce']<base['ce'] else 'worsened' if r['ce']>base['ce'] else 'equal'
                wo='improved' if r['we']<base['we'] else 'worsened' if r['we']>base['we'] else 'equal'
                changes[n]['changed']+=changed;changes[n]['raw_changed']+=r['raw_prediction']!=base['raw_prediction']
                changes[n][outcome]+=1;changes[n]['wer_'+wo]+=1
                item[n+'_prediction_changed']=changed;item[n+'_CER_delta']=r['cer']-base['cer'];item[n+'_CER_outcome']=outcome
                item[n+'_WER_delta']=r['wer']-base['wer'];item[n+'_WER_outcome']=wo
        comparison.append(item)
    table('utterance_comparison.csv',comparison)
    aggregate=[]
    for n,m in metrics.items():
        aggregate.append(dict(experiment=n,**m,cer_delta_pp=100*(m['cer']-metrics['E1']['cer']),wer_delta_pp=100*(m['wer']-metrics['E1']['wer']),
            best_checkpoint=str(models[n]),best_epoch=2,**changes.get(n,dict(changed=0,raw_changed=0,improved=0,worsened=0,equal=67,wer_improved=0,wer_worsened=0,wer_equal=67))))
    table('test_comparison.csv',aggregate)
    summary=dict(metrics=metrics,changes=changes,historical_e1_check=historical_check,training_executed=False)
    write('summary.json',summary)
    changed_inputs=[p for p,h in before.items() if sha(p)!=h];assert not changed_inputs,changed_inputs
    write('preservation.json',dict(all_inputs_unchanged=True,protected_files=len(before),training_executed=False))
    lines=['# KSM_M_45 E4-A/B Personal Test 평가','',
        '저장된 Validation Best 모델만 사용했다. E1도 같은 설정으로 재평가하여 기존 67개 발화별 예측과 지표가 일치함을 확인했다. 추가 학습·최종 모델 재선택·하이퍼파라미터 변경은 수행하지 않았다.','',
        '| 모델 | Best Epoch | Test CER | Test WER | E1 대비 CER %p | E1 대비 WER %p |','|---|---:|---:|---:|---:|---:|']
    for r in aggregate:lines.append(f"| {r['experiment']} | {r['best_epoch']} | {100*r['cer']:.4f}% | {100*r['wer']:.4f}% | {r['cer_delta_pp']:+.4f} | {r['wer_delta_pp']:+.4f} |")
    lines+=['','| E1 대비 | 예측 변경 | CER 개선/악화/동일 | WER 개선/악화/동일 |','|---|---:|---|---|']
    for n,c in changes.items():lines.append(f"| {n} | {c['changed']} | {c['improved']}/{c['worsened']}/{c['equal']} | {c['wer_improved']}/{c['wer_worsened']}/{c['wer_equal']} |")
    lines+=['','동일은 발화별 오류 수 기준이며 예측 문자열 동일과 별도로 집계했다. CER는 정답 문자 수를 분모로 한 전체 micro CER이며, WER도 전체 정답 단어 수로 집계했다. 숫자를 보존하며 E1의 NFC·소문자·문장부호 공백화·공백 축약 정규화를 그대로 사용했다.','',
        '## 데이터·생성 설정','',
        f"분할 CSV: `{cfg['paths']['manifest_csv']}`. WAV root: `{cfg['paths']['audio_root']}`. Train168/Validation65/Test67. Test는 기존 E1 평가의 발화 경로·정답과 정확히 일치했다. Train/Val과 Test의 (batch,audio_member), 실제 WAV 경로, WAV 바이트 SHA256 교집합이 모두 0이었다. 같은 원본 recording_id는 원래 발화 단위 분할에서 공유될 수 있으며 녹음 단위 독립 실험이라고 주장하지 않는다.",
        '`src/personalization/dataset.py`, `trainer.py::build_model_and_processor/make_trainer`, `evaluator.py::compute_asr_metrics/decode_predictions/metric_counts/save_predictions`를 재사용했다. 같은 Processor, mono/16kHz 리샘플링·30초 패딩, Korean/transcribe, max_length160, beam1, samplingFalse, eval_batch2 및 기존 CUDA 혼합정밀도 설정이다. Train/Val 오디오를 모델 입력으로 사용하지 않았다.','',
        '## 발화별 변화','']
    changed_rows=[r for r in comparison if any(r[n+'_prediction_changed'] for n in ('E4-A','E4-B'))]
    if changed_rows:
        lines+=['| ID | 정답 | E1 | E4-A | E4-B |','|---|---|---|---|---|']
        for r in changed_rows:lines.append(f"| {r['utterance_id']} | {r['reference']} | {r['E1_prediction']} | {r['E4-A_prediction']} | {r['E4-B_prediction']} |")
    else:lines.append('세 모델의 정규화된 67개 예측이 모두 같았다. 지표 동일은 개선과 악화가 상쇄된 결과가 아니다.')
    lines+=['','## 경로와 보존','']
    for n,p in models.items():lines.append(f'- {n}: `{p}`')
    lines+=['','평가 전 전체 checkpoint/data 경로를 console.log 및 preflight.json에 출력·기록했다. 원본 코드·WAV·checkpoint·학습 로그·summary의 해시가 종료 후 동일했다. 원래 학습 summary의 test_evaluated 필드도 수정하지 않았다. 평가 결과만 이 새 디렉터리에 저장했다.',
        '기존 참고 경로: E1 `outputs/personalization/E1_v2/KSM_M_45/test_metrics.json`; E3 `outputs/personalization/E3/KSM_M_45/legacy_e1_lambda_0.1/final_test`. E3는 이번에 재평가하지 않았다.']
    (DEST/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
