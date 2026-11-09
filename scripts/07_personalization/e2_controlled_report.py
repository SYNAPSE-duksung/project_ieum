"""Validation-only report for completed E2 controls, including saved-Best reload."""
import csv,gc,json,sys
from pathlib import Path
import torch
from transformers import WhisperProcessor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import e2_controlled_train as run
from src.personalization.ctc_head import WhisperWithAuxiliaryCTC
from src.personalization.e2_controlled import ControlAudit,make_control_trainer,summarize_ctc
from src.personalization.evaluator import compute_asr_metrics,decode_predictions
OUT=run.OUT

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    info={n:read(OUT/n/'summary.json') for n in ('R1','R2')}
    init={n:read(OUT/n/'initialization.json') for n in info}
    assert init['R1']['backbone']==init['R2']['backbone'] and init['R1']['head']==init['R2']['head']
    backward={n:[json.loads(x) for x in (OUT/n/'backward_audit.jsonl').read_text().splitlines()] for n in info}
    order_equal=[r['sample_ids'] for r in backward['R1']]==[r['sample_ids'] for r in backward['R2']]
    assert order_equal and all(len(backward[n])==420 for n in info)
    gradients={n:[json.loads(x) for x in (OUT/n/'gradients_preclip.jsonl').read_text().splitlines()] for n in info}
    assert all(len(gradients[n])==105 and all(r['microbatches']==4 for r in gradients[n]) for n in info)
    assert all(abs(r['backward_loss']-r['total_loss'])<1e-5 and not r['divided_by_accumulation'] for n in info for r in backward[n])
    audits={n:read(OUT/n/'training_audit.json') for n in info}
    assert all(audits[n]['optimizer_steps']==audits[n]['scheduler_last_epoch']==105 for n in info)
    epochs=[]
    for n,audit in audits.items():
        for r in audit['epochs']:
            epochs.append(dict(experiment=n,epoch=r['epoch'],step=r['step'],**r['train'],
                validation_cer=r['validation']['eval_cer'],validation_wer=r['validation']['eval_wer'],
                preclip_total_mean=r['gradient_preclip']['total']['mean'],preclip_encoder_mean=r['gradient_preclip']['encoder']['mean'],
                preclip_head_mean=r['gradient_preclip']['head']['mean'],
                blank_argmax_fraction=r['ctc']['blank_argmax_fraction'] if r['ctc'] else None,
                mean_blank_probability=r['ctc']['mean_blank_probability'] if r['ctc'] else None,
                empty_utterances=r['ctc']['all_blank_utterances'] if r['ctc'] else None,
                collapsed_nonblank_tokens=r['ctc']['nonblank_collapsed_tokens'] if r['ctc'] else None))
    table('epochs.csv',epochs)
    old_e1=read(ROOT/'outputs/personalization/E1_v2/KSM_M_45/checkpoints/checkpoint-105/trainer_state.json')
    e1=min((r for r in old_e1['log_history'] if 'eval_cer' in r),key=lambda r:r['eval_cer'])
    old_e2=read(ROOT/'outputs/personalization/E2/KSM_M_45/lambda_0.03/run_summary.json')
    records=[dict(experiment='E1 historical',best_epoch=e1['epoch'],cer=e1['eval_cer'],wer=e1['eval_wer'],collapse='N/A'),
        dict(experiment='E2 historical',best_epoch=1,cer=old_e2['best_validation_cer'],wer=old_e2['validation_metrics']['eval_wer'],collapse='Best epoch1:0/65 empty; last epoch5:65/65 empty')]
    for n,result in info.items():
        dest=OUT/n/'best_validation_evaluation'
        if dest.exists():raise FileExistsError(dest)
        dest.mkdir()
        model=WhisperWithAuxiliaryCTC.from_pretrained(result['best_checkpoint'],local_files_only=True)
        processor=WhisperProcessor.from_pretrained(result['best_checkpoint'],local_files_only=True)
        ds=run.datasets(run.rows(),processor,model);audit=ControlAudit(dest)
        trainer=make_control_trainer(model,processor,ds,dest/'runtime',run.CFG['training'],audit,n=='R1')
        prediction=trainer.predict(ds['val'],metric_key_prefix='validation')
        metrics=compute_asr_metrics(prediction.predictions,prediction.label_ids,processor.tokenizer)
        refs,hyps=decode_predictions(prediction.predictions,prediction.label_ids,processor.tokenizer)
        assert metrics['cer']==result['best_validation_cer'] and metrics['wer']==result['best_validation_wer']
        run.write(dest/'metrics.json',dict(**metrics,ctc=summarize_ctc(audit.eval_ctc),matches_training_best=True))
        with (dest/'predictions.jsonl').open('w',encoding='utf-8') as f:
            for row,ref,hyp in zip(ds['val'].rows,refs,hyps):f.write(json.dumps(dict(audio_member=row['audio_member'],reference=ref,prediction=hyp),ensure_ascii=False)+'\n')
        # R2 head must be unchanged, despite sharing the model class.
        current_head=run.digest_tensors(model.ctc_head.state_dict().items())
        if n=='R2':assert current_head==init[n]['head']
        run.write(dest/'head_identity.json',dict(initial_head=init[n]['head'],best_head=current_head,unchanged=current_head==init[n]['head']))
        records.append(dict(experiment=n,best_epoch=result['best_epoch'],cer=metrics['cer'],wer=metrics['wer'],
            collapse='epoch2 onward 65/65 empty; best epoch1:0/65' if n=='R1' else 'disabled; head unchanged'))
        del trainer,model,prediction,ds;gc.collect();torch.cuda.empty_cache()
    table('comparison.csv',records)
    sources=read(OUT/'protected_original_hashes.json')
    assert all(run.sha(p)==h for p,h in sources.items())
    r1clipped=sum(r['total']>1 for r in gradients['R1']);r2clipped=sum(r['total']>1 for r in gradients['R2'])
    run.write(OUT/'control_verification.json',dict(initial_backbone_identical=True,initial_ctc_head_identical=True,all_5_epoch_data_order_identical=order_equal,
        microbatches_per_run=420,optimizer_steps_per_run=105,scheduler_steps_per_run=105,
        every_microbatch_legacy_loss_verified=True,r1_head_nonzero_steps=sum(r['head']>0 for r in gradients['R1']),
        r2_head_nonzero_steps=sum(r['head']>0 for r in gradients['R2']),r1_clipped_steps=r1clipped,r2_clipped_steps=r2clipped,
        originals_unchanged=True,protected_original_files=len(sources),test_evaluated=False))
    lines=['# KSM_M_45 E2 Gradient Accumulation 통제 실험','',
        '새 실험 R1/R2를 동일 General에서 독립 초기화해 각각 5epoch 학습했다. 기존 E1/E2 파일과 결과는 보존했고 Test는 모델 입력·선택·조정에 사용하지 않았다.','',
        '## 1. 기존 최적화 차이','',
        '기존 E1은 microbatch 평균 CE를 누적 횟수로 나누지 않는 legacy 경로, 기존 E2는 Total Loss를 4로 나누는 경로다. 둘 다 batch2/accum4/168 Train/5epoch로 105 optimizer step이다. 이번 R1/R2는 둘 다 legacy 경로로 통일했고 lambda를 4배 조정하지 않았다.','',
        '## 2. CTC 구현 검증','',
        '기존 ctc_head.py/ctc_utils.py/e2_trainer.py는 수정하지 않았다. 동일 General 가중치가 Encoder/Decoder에 모두 로딩됨을 확인했다. CTC Linear(768,11183)는 Encoder hidden에 연결되고 Decoder 입력으로 직접 전달되지 않는다. ASR와 CTC는 Encoder 파라미터를 공유하므로 CTC backward가 ASR에 간접 영향을 줄 수 있다.',
        'Blank0, Hangul1~11172, digit0~9=11173~11182의 중복 없는 vocabulary를 확인했다. Targets는 padding 없는 연결 배열이며 blank를 넣지 않는다. 실제 Conv kernel/stride/padding 공식과 Feature Extractor 유효 mask로 길이를 계산하고 반복 token 추가 blank 경로까지 검사했다. Train168/Val65 전체 검사에서 unsupported/empty/unalignable 문제는 없었다. 30초 패딩 전체를 실제 음성 길이로 사용하지 않는다.',
        'CTCLoss는 원래 mean/blank0/zero_infinityTrue, float32 log_softmax, [T,B,C] 입력을 유지했다. 진단 greedy는 유효 프레임만 사용해 연속 중복을 먼저 병합한 다음 blank를 제거한다. 합성 raw path 7,0-digit,blank,0-digit,0-digit이 700이 되는지 확인했다. 명백한 기존 CTC 구현 오류는 발견되지 않아 별도 수정된 CTC 실험을 추가하지 않았다.',
        '실제 General 배치에서 ASR/CTC 각각 conv1.weight에 유한한 비영 Gradient가 존재했고 CTC Head weight에도 전달됐다. CTC→Decoder layer fc1 직접 Gradient는 None이었다. probe 수치는 verification_v2/verification.json에 있으며 conv1 탐침 norm과 전체 Encoder norm을 구분해야 한다.','',
        '## 3. R1/R2 조건 및 실제 누적 검증','',
        'Train168/Val65, full Encoder+Decoder+Head 구조, General SHA709258bb...306e705, epoch5, LR1e-5, AdamW torch, linear scheduler/warmup0/weight_decay0/clip1, batch2/accum4, seed42/data_seedNone, BF16, 원래 생성·정규화를 공유했다. R1은 ASR+.03*CTC, R2는 ASR만 최적화했다.',
        'R2도 기존 모델 forward를 통해 실행해 Accelerate autocast를 보존한다. 원래 forward 내부에서 raw CTC 계산은 수행되지만 최적화 scalar에는 포함하지 않으며, CTC Head에는 gradient가 생기지 않는다. R2 로그의 ctc_loss=0은 비활성 표시이지 실제 raw CTC가 0이라는 뜻이 아니다.',
        '실제 전체 General 모델로 4 microbatch/1 optimizer step의 E1/R2 smoke를 수행했다. backward Loss 4개와 feature 순서가 일치하고 업데이트 후 backbone 최대 차이는 0이었다. 이 검증에는 결정적 CUDA 설정을 사용했으며 전체 학습은 기본 E1 정밀도/비결정적 설정을 유지했다.',
        'backward_audit.jsonl은 각 microbatch의 ASR/CTC/weighted CTC/실제 backward scalar와 sample ID를 기록한다. R1/R2의 초기 backbone·head hash, 5epoch 840개 sample 방문 순서가 동일함을 확인했다. 각 420 backward, 4 microbatch마다 clip/update, 105 optimizer/scheduler step을 검증했다.','',
        '## 4. Validation 결과','',
        '| 실험 | Best Epoch | Val CER | Val WER | CTC 상태 |','|---|---:|---:|---:|---|']
    for r in records:lines.append(f"| {r['experiment']} | {r['best_epoch']:g} | {100*r['cer']:.4f}% | {100*r['wer']:.4f}% | {r['collapse']} |")
    lines += ['', 'R1/R2 Best는 저장 checkpoint를 다시 로딩해 Val65에서 재평가했고 학습 중 Best 지표와 정확히 일치했다. 비교 CSV는 비율 단위다.', '',
        '## 5. Blank collapse와 Gradient','',
        '| R1 Epoch | Train ASR | Train CTC | Val CER | frame argmax blank | empty utterances | nonblank collapsed tokens | 평균 blank 확률 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in audits['R1']['epochs']:
        c=r['ctc'];lines.append(f"| {r['epoch']:g} | {r['train']['asr_loss']:.6f} | {r['train']['ctc_loss']:.4f} | {100*r['validation']['eval_cer']:.4f}% | {100*c['blank_argmax_fraction']:.3f}% | {c['all_blank_utterances']}/65 | {c['nonblank_collapsed_tokens']} | {c['mean_blank_probability']:.6f} |")
    lines += ['', '기존 E2도 직접 재확인했다. checkpoint-21(Best)은 blank argmax 약50.16%, 빈 문장0/65, 비공백 collapsed token5465개였다. checkpoint-105(Epoch5)는 blank argmax100%, 빈 문장65/65, 평균 blank 확률93.68%였다. 기존 Best도 모두 blank였다는 서술은 수정해야 한다.',
        f"R1 Head Gradient는 {sum(r['head']>0 for r in gradients['R1'])}/105 update에서 비영이고, R2는 {sum(r['head']>0 for r in gradients['R2'])}/105이다. R2 Best Head hash는 초기값과 같다. R1의 clipping 발동은 {r1clipped}/105, R2는 {r2clipped}/105이다. 실제 preclip 전체/Encoder/Decoder/Head norm은 gradients_preclip.jsonl과 epochs.csv에 있다.",
        'Epoch별 관측에서 R1은 Epoch2부터 전부-blank가 확인됐다. 정확히 어느 microbatch에서 시작됐는지는 이 간격 측정으로 확정하지 않는다. argmax blank 비율과 blank 확률, 빈 문장 비율은 서로 다른 지표다.','',
        '## 6. 원인 해석','',
        '누적 방식을 맞춘 R1의 Best CER/WER는 기존 E2와 같았다. 따라서 이 설정에서 기존 E2의 악화를 Gradient Accumulation 차이 하나로 설명할 수 없다. 같은 초기값·순서·legacy 방식에서 R2가 E1 성능을 재현하고 R1보다 좋으므로, CTC Head의 존재 자체보다 활성 CTC 목적함수/그 Gradient와 결합된 학습이 성능 저하에 관여한 결과다.',
        '초기 Gradient 전달과 finite CTC Loss가 정상이어도 greedy가 모두 blank가 될 수 있음을 확인했다. CTC Loss 하락만으로 정렬 학습 성공을 판단할 수 없다. Encoder 공유를 통한 간섭·랜덤 대규모 Head의 초기 학습·loss 가중치·데이터와 CTC 목표의 특성이 후보 원인이지만, 이 실험은 이들을 각각 분리하지 않았다. 구조 자체의 결함이나 blank collapse가 ASR 악화의 유일한 원인이라고 단정하지 않는다.',
        'Gradient clipping은 전체 Gradient 스케일 차이를 줄일 수 있으므로, Loss를 4배 누적했다고 parameter update도 정확히 4배인 것은 아니다.','',
        '## 7. Notion 수정 문장','',
        '> 기존 E1과 E2는 Gradient Accumulation 설정은 같았으나 Loss 스케일이 달랐다. 이후 동일 General 초기값·데이터 순서·E1 legacy 누적으로 R1(ASR+0.03CTC)과 R2(ASR-only)를 각각 5epoch 비교했다. R1의 Best CER/WER는 기존 E2와 같은 6.8650%/12.9870%였고 R2는 E1과 같은 2.8604%/8.1169%였다. R1은 CTC Head와 Encoder에 Gradient가 전달되는데도 Epoch2부터 Val65 전체가 greedy blank 출력이 됐다. 다만 기존 E2와 R1의 Best Epoch1은 전부 blank가 아니므로, 성능 악화를 누적 차이나 Head 구조 결함 하나로 단정하지 않는다.', '',
        '## 8. 추가 실험 필요 여부','',
        '누적 차이만이 원인인지와 Head가 사용되지 않을 때 E1을 재현하는지는 이번 통제로 구분했다. 반면 loss 가중치·Head 초기 학습·공유 Encoder 변화·데이터 영향 중 어떤 요인이 주원인인지 확정하려면 별도 통제가 필요하다. 이번에는 추가 lambda·Head 준비 학습·다른 화자 실험·Test 평가를 수행하지 않았다.', '',
        '## 실행상 한계·복구 기록','',
        '검증 초기에 결정적 CUDA 환경변수 설정 시점 오류와 새 R2 실행기의 base-forward 직접 호출에 따른 autocast 우회를 발견했다. 새 실행기에서만 수정하고 실제 E1/R2 update parity를 통과한 뒤 전체 학습했다. 기존 CTC 구현을 수정한 것은 아니다.',
        'R1에서는 새 scheduler 계측이 현재 Trainer가 사용하지 않는 create_optimizer_and_scheduler 후크에 연결돼 raw counter가0으로 남았다. 실제 학습은105 update와 scheduler.last_epoch105까지 끝났지만 종료 후 assert가 실패했다. 원시 audit·실패 status를 보존하고 postrun_verification.json으로 검증해 summary를 완성했다. 이후 R2는 실제 scheduler 상태를 매 update 기록하도록 수정했으며 계산·optimizer 주기는 바꾸지 않았다. R1 source는 verification/code_snapshot_v1에 보존했다.',
        '계측 수정 검증이 R1 후반과 잠시 동시에 GPU를 사용해 중단하고 R1 종료 후 재검증했다. R1 시간·peak는 이 짧은 자원 경합의 영향을 포함할 수 있으므로 순수 속도 비교로 쓰지 않는다. 초기값·데이터 순서·수학적 학습 설정은 동일하지만 BF16 비결정적 변동과 한 seed/한 화자 한계는 남는다.', '',
        '## 파일','',
        'comparison.csv, epochs.csv, control_verification.json; 각 R1/R2의 initialization/configuration/environment/training_arguments/training_audit/backward_audit/gradients_preclip/validation_ctc/summary와 best_validation_evaluation; verification_v2/verification.json. R2의 validation_ctc는 비활성이므로 별도 생성되지 않는다. 기존 파일 해시 보존은 protected_original_hashes.json으로 검증했다.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(comparison=records,clipped=dict(R1=r1clipped,R2=r2clipped),initial_identity=True,data_order_equal=True),indent=2))

if __name__=='__main__':main()
