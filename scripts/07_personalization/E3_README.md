# E3: Error Hypothesis-guided Correction Learning

Run from `project_ieum`. Existing E1/E2 code/results are not changed. Defaults
are validate only, lambda=.1, margin=.1, KSM_M_45, E1 General initialization.

```powershell
.\.venv\Scripts\python.exe scripts/07_personalization/e3_train.py --mode validate
.\.venv\Scripts\python.exe scripts/07_personalization/test_e3.py
.\.venv\Scripts\python.exe scripts/07_personalization/e3_train.py --mode smoke --smoke-device cuda

# Explicit future commands, NOT automatically run:
.\.venv\Scripts\python.exe scripts/07_personalization/e3_train.py --mode train --lambda-correction 0
.\.venv\Scripts\python.exe scripts/07_personalization/e3_train.py --mode train --lambda-correction 0.1
# After validation review and model selection only:
.\.venv\Scripts\python.exe scripts/07_personalization/e3_train.py --mode test --lambda-correction 0.1
```

## Sources and data

E1 YAML supplies General model, manifest, WAV root, 5 epochs, LR=1e-5, batch=2,
accumulation=4 (effective batch 8), seed=42, checkpointing and decoder generation.
Explicit E1 optimizer defaults are AdamW, betas=.9/.999, epsilon=1e-8, no weight
decay, clipping=1, linear scheduler, no warmup, BF16 if supported. No early
stopping callback exists in E1. Best checkpoint is selected using Validation CER.

Manifest columns: speaker_id, personal_split, batch, audio_member, text,
recording_id. WAV is `audio_root / batch / audio_member`; utterance cache join
key is `(batch, audio_member)` after selecting speaker. E1 read/resample/padding,
decoder labels/collation and CER/WER functions are reused. Splits are 168/65/67.

Profile JSON stores aggregate raw_profile/filtered_profile, not utterance text.
The actual fixed General texts are in `error_profiles/predictions.csv`:
speaker_id,batch,audio_member,reference,prediction. All 168 Train keys/references
match; 134 are equal after E1 normalization and 34 are eligible. None is empty
or over 160 tokens. Missing, duplicate, extra, reference-mismatched or foreign
split keys fail validation. Empty or overlong hypotheses are explicitly reported
and omitted ONLY from correction; ASR retains the sample. References over E1's
limit fail; no text is truncated. Validation/Test carry no correction fields.

Historical cache provenance is supported by profile_generator.py and Profile
experiment_info naming the same General checkpoint. CSV does not contain a
historical model checksum, so exact historical weights cannot be independently
proven from it alone. Current General/cache/manifest checksums are saved at run
start. Candidates are never generated during training or replaced with E1/E2
outputs. Globally wrong cached sentences remain real hypotheses; they are not
necessarily phonetic errors, and the ranking signal may overfit these 34 cases.

## Objective and token policy

`total = asr + lambda * mean(relu(margin + score(hyp) - score(ref)))` over
eligible candidates in each microbatch. All samples contribute to ASR. A batch
with no eligible candidates contributes correction=0. Lambda=0 skips candidate
forward and scoring entirely. Validation/Test loss is ASR-only.

ASR uses stock Whisper CE and E1 labels. The E1 collator checks tokenizer BOS
(50257), while Korean label prefixes actually begin with decoder-start 50258.
Consequently E1 retains startoftranscript in labels. E3 preserves that ASR
convention rather than silently changing the baseline; stock Whisper performs
shift_tokens_right exactly once for both reference and hypothesis labels.

Scores average current-model log probabilities of content tokens PLUS EOS.
Padding (-100), startoftranscript, Korean language, transcribe and notimestamps
control tokens are excluded from score numerator/denominator. EOS is included
consistently to score termination. Scores are float32; no score or encoder state
is detached. Candidate logits reuse the same Encoder states for eligible rows,
adding one Decoder forward, not a second Encoder forward. New parameters, CTC,
alignment weights, Profile Encoder and FiLM are absent.

## Accumulation compatibility

Installed Transformers detects stock E1 forward **kwargs and sets
model_accepts_loss_kwargs=True, although Whisper CE is a microbatch mean. This
historical path does NOT divide by accumulation. E2 explicitly sets False and
does divide. E3 never infers this choice: default `legacy_e1` deliberately matches
the historical E1 gradient scale, optimizer/scheduler and clipping. This is a
compatibility choice, not the recommended standard mean-accumulation formula.
`--accumulation-mode mean` uses explicit False for corrected mean accumulation;
compare it only with a lambda=0 run using the same mean mode. Historical E2 is
not a controlled baseline for legacy-mode E3. Smoke checks lambda=0 loss and
encoder gradient equality against E1's actual training_step, with no optimizer
steps. Prefix/control token scoring does not alter ASR loss.

## Outputs and evaluation

Outputs: `outputs/personalization/E3/KSM_M_45/<mode>_lambda_<value>/`.
Existing directories are refused; every train run starts from General, no resume.
Keep all 5 epoch checkpoints. Per-epoch ASR/correction/total loss and candidate
usage are recorded in training_audit.json; losses are sample-weighted means of
microbatch losses, while ranking within each microbatch is eligible-candidate
mean. training_logs.jsonl stores step logs. Audit records validation CER/WER,
best epoch/checkpoint/CER, time, peak GPU memory and numerical failures.
run_summary.json and best_model are saved after training; test_evaluated=false.

Test requires a separate explicit `--mode test`; it reloads validation-selected
best_model, performs decoder inference only and writes test_metrics.json,
test_predictions.jsonl and e1_comparison.json inside a NEW final_test directory.
Comparison differences are E3 minus historical E1_v2. Always report the chosen
accumulation mode, and prefer the paired lambda=0 run for causal interpretation.
Test results never choose lambda/margin/checkpoints. No Test evaluation is part
of implementation smoke checks.

Smoke runs 2 actual Train batches (eligible and identical candidates), checks
correction Encoder/Decoder gradients, finite total backward, lambda=0 E1
equivalence, and generate/prediction_step. CUDA uses BF16 where supported and
reports allocated/reserved peaks; no optimizer state is allocated, so full
training memory may be higher. Full training starts ONLY with --mode train.
