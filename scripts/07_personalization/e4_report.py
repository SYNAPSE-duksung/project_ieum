"""Summarize completed E4 validation runs without inference or Test access."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/personalization/E4/KSM_M_45'
REPORT=ROOT/'reports/e4_experiment_20261107'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_write(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    verification=read(REPORT/'verification.json')
    assert verification['passed']
    old=read(ROOT/'outputs/personalization/E1_v2/KSM_M_45/checkpoints/checkpoint-105/trainer_state.json')
    evals=[r for r in old['log_history'] if 'eval_cer' in r]
    e1=min(evals,key=lambda r:r['eval_cer'])
    records=[dict(experiment='E1',best_epoch=e1['epoch'],validation_cer=e1['eval_cer'],validation_wer=e1['eval_wer'],
        cer_delta_pp=0.,wer_delta_pp=0.,training_seconds=None,weighted_utterances=0,
        weighted_syllable_types=0,weighted_content_token_ratio=0.,weighted_all_label_token_ratio=0.,
        best_checkpoint=old['best_model_checkpoint'])]
    epochs=[]
    audits={}
    for variant in ('A','B'):
        directory=OUT/f'E4-{variant}'
        if read(directory/'status.json')['status']!='completed':raise ValueError('Cannot report incomplete experiment')
        result=read(directory/'summary.json');audit=read(directory/'training_audit.json');audits[variant]=audit
        assert len(audit['epochs'])==5 and len(audit['nonfinite_events'])==0
        counts=result['weight_info']
        records.append(dict(experiment=f'E4-{variant}',best_epoch=result['best_epoch'],validation_cer=result['best_validation_cer'],
            validation_wer=result['best_validation_wer'],cer_delta_pp=100*(result['best_validation_cer']-e1['eval_cer']),
            wer_delta_pp=100*(result['best_validation_wer']-e1['eval_wer']),training_seconds=result['training_seconds'],
            weighted_utterances=counts['a_weighted_utterances'] if variant=='A' else counts['b_weighted_utterances'],
            weighted_syllable_types=result['weighted_syllable_types'],
            weighted_content_token_ratio=None if variant=='A' else counts['b_weighted_content_tokens']/counts['content_tokens'],
            weighted_all_label_token_ratio=(counts['a_weighted_tokens'] if variant=='A' else counts['b_weighted_content_tokens'])/counts['all_label_tokens'],
            best_checkpoint=result['best_checkpoint']))
        for item in audit['epochs']:
            epochs.append(dict(experiment=f'E4-{variant}',epoch=item['epoch'],asr_loss=item['asr_loss'],weighted_loss=item['weighted_loss'],
                validation_cer=item['validation']['eval_cer'],validation_wer=item['validation']['eval_wer']))
    protected=read(ROOT/'reports/e1_e3_multispeaker_pilot_20261107/selection_fingerprint.json')['source_hashes']
    for name,h in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    # Additional sources read in the E4 feasibility analysis must also remain intact.
    for name,h in read(ROOT/'reports/e4b_feasibility_20261107/input_checksums.json').items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    csv_write(REPORT/'comparison.csv',records);csv_write(REPORT/'epochs.csv',epochs)
    lines=['# KSM_M_45 E1 / E4-A / E4-B Validation 실험 결과','',
        'E4-A와 E4-B를 각각 동일 General 모델에서 독립적으로 5epoch 학습했다. Personal Test 평가는 실행하지 않았다. 기존 E1/E2/E3 코드·체크포인트·결과는 보존했다.', '',
        'E4-A/B each completed five epochs from the same General Whisper Small checkpoint. No Personal Test evaluation was performed. Historical E1/E2/E3 code and results were not modified.', '',
        '| Experiment | Best epoch | Validation CER | Validation WER | CER delta (pp) | WER delta (pp) | Training seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        seconds='not recorded' if r['training_seconds'] is None else f"{r['training_seconds']:.2f}"
        lines.append(f"| {r['experiment']} | {r['best_epoch']:g} | {100*r['validation_cer']:.4f}% | {100*r['validation_wer']:.4f}% | {r['cer_delta_pp']:+.4f} | {r['wer_delta_pp']:+.4f} | {seconds} |")
    lines += ['', '음수 변화량은 개선을 뜻한다. WER는 독립적인 최소값이 아니라 Best Validation CER와 같은 Epoch의 값이다. 한 화자·한 seed 결과이므로 일반화 효과나 통계적 유의성을 입증하지 않는다.']
    lines += ['', 'Negative deltas mean improvement. WER is taken at the minimum-CER epoch, not independently minimized.', '',
        '## Fixed conditions', '',
        'KSM_M_45 Train168/Validation65/Test67 split; Test audio was not loaded. General SHA256: '+verification['weight_info']['model_sha256']+'.',
        'Both runs: AdamW torch, LR=1e-5, linear scheduler, no warmup, weight_decay=0, max_grad_norm=1, batch=2, accumulation=4, epochs=5, seed=42, data_seed=None, BF16, full encoder/decoder, gradient checkpointing, generation max_length=160/beams=1. Historical legacy E1 gradient scaling is retained (model_accepts_loss_kwargs=True; Accelerate accumulation=1).',
        'Only persistence differs from historical E1: model-only best checkpoint, one retained checkpoint, no duplicate best_model directory. No optimizer/scheduler/RNG resume states are saved.', '',
        '## Weighting and normalization', '',
        'Both training losses use sum(valid_token_weight * CE) / sum(valid_token_weight), across all label tokens in each microbatch. Padding (-100) is excluded. This is not an average of utterance means. All weights=1 recovers E1 token-mean CE.',
        'E4-A: the exact 34 E3 eligible Train error utterances have 1.5 on every nonpadding target token, including E1-retained special tokens/EOS; others have 1.0. If all tokens in a batch share the same weight, the weighted-mean scaling cancels. Effective emphasis occurs within mixed batches.',
        'E4-B: n>=5, errors>=2, errors in >=2 utterances; p0=175/2430; r=(e+10*p0)/(n+10); syllable weight=1+min(.5,max(0,r-p0)); otherwise 1. UTF-8 token/character overlap proportion projects weights to positions. Special tokens/EOS/padding have weight1; padding remains excluded from loss.',
        'E4-B: 23 syllable types, 134/168 utterances, 341/2307 content-token positions (14.7811%), or 341/3147 E1-valid label positions (10.8357%), receive weights >1. E4-A: 536/3147 valid label positions receive weight1.5. Static unique-dataset ratios are reported; the same examples are seen each epoch.',
        'All actual E4-A weights lie in [1,1.5], E4-B in [1,1.375]. The smaller actual B maximum follows from byte overlap dilution, including whitespace. weight_validation_details.json contains full-sentence alignment examples and measured bounds.',
        'E4-B byte mapping is a declared approximation to token-level CE, not a unique syllable-level probability decomposition. Weights apply to all occurrences of a selected syllable, including contexts General recognized correctly.', '',
        '## Essential validation', '',
        'verification.json contains deterministic CPU 4-microbatch accumulation/one AdamW update equality plus actual General-model FP32 loss/gradient/update equality. Initial nondeterministic GPU probing failed a strict absolute gradient threshold; deterministic CUBLAS/cuDNN controls in the verification process resolved it without changing the loss or relaxing tolerance. Full General BF16 weighted A/B forward/backward were finite. Training runs use the original E1 precision/scaling rather than enabling deterministic algorithms globally.',
        'The E4 entry point sets CUBLAS_WORKSPACE_CONFIG=:4096:8; torch deterministic algorithms and deterministic cuDNN are enabled only in the verification process. Historical E1 environment variables were not recorded, so bitwise historical-run reproduction is not claimed.',
        'Actual General all-one max gradient difference=0; max updated parameter difference=0; loss difference below 1e-6. All 168 reference/token byte round-trips and weight bounds [1,1.5] passed. Special-prefix/EOS and collator labels are unchanged.', '',
        'Whisper best reload emits the historical missing proj_out.weight warning because the output weight is tied to the decoder embedding; this warning is distinguished from a failed run. Console BPE cleanup deprecation/progress messages can also be rendered as PowerShell NativeCommandError objects despite Python completing. Completed status, summary, all five epoch records and actual Python exit status are used to assess completion.', '',
        'Both actual best checkpoints were separately reloaded on CPU without inference. checkpoint_verification.json confirms the shared proj_out/decoder-embedding storage pointer, one retained checkpoint each, and distinct A/B weight hashes. No additional training or evaluation was performed during this reload check.', '',
        '## Training audit and limitations', '']
    for variant,audit in audits.items():
        lines.append(f"E4-{variant}: nonfinite events={len(audit['nonfinite_events'])}; GPU peak allocated={audit['gpu_peak_allocated_bytes']/1024**3:.3f}GiB; reserved={audit['gpu_peak_reserved_bytes']/1024**3:.3f}GiB.")
    lines += ['', 'Epoch losses in epochs.csv are sample-weighted means of microbatch losses, not a second global weighted-token objective. Original E1 epoch-level training loss/time are not invented when absent from saved logs.',
        'Interpretation is limited to this speaker and seed. Profile errors are highly recording-group dependent; E4-B does not directly target insertion errors. No neutral all-one full five-epoch run was added; equality is supported by the specified batch/update controls. CUDA BF16 repeat variation and historical legacy scaling remain limitations.', '',
        '## Best checkpoints', '']
    lines += [f"- {r['experiment']}: `{r['best_checkpoint']}`" for r in records]
    lines += ['', '## Artifacts', '',
        'comparison.csv: best Validation comparison; epochs.csv: E4 epoch history; verification.json: essential checks, weight coverage and examples; each E4 output directory: provenance, arguments, trainer state, audit, events, status, summary and canonical best checkpoint.',
        'New implementation files: src/personalization/e4_weighted.py, scripts/07_personalization/e4_train.py, scripts/07_personalization/e4_report.py. Original E1 YAML and modules were read and reused; no original code was edited.']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (REPORT/'preservation.json').write_text(json.dumps(dict(legacy_sources_unchanged=True,feasibility_inputs_unchanged=True,
        personal_test_evaluated=False,comparison=records),indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
