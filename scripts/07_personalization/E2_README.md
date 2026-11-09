# E2 auxiliary CTC experiment

Run commands from `project_ieum` using `.venv/Scripts/python.exe`.
No command except explicit `--mode train` performs optimizer steps.

```powershell
# Entire pilot preflight (default mode); no weights loaded, no output files.
.\.venv\Scripts\python.exe scripts/07_personalization/e2_train.py --mode validate

# CPU unit/integration checks.
.\.venv\Scripts\python.exe scripts/07_personalization/test_e2.py

# General Whisper Small, one real batch: separate ASR/CTC encoder gradients,
# total backward, Trainer save/reload, decoder generate, Trainer prediction_step.
# Temporary checkpoint only; no optimizer step or experiment result writes.
.\.venv\Scripts\python.exe scripts/07_personalization/e2_train.py --mode smoke

# Explicit training commands; each starts independently from General.
.\.venv\Scripts\python.exe scripts/07_personalization/e2_train.py --mode train --lambda-ctc 0.03
.\.venv\Scripts\python.exe scripts/07_personalization/e2_train.py --mode train --lambda-ctc 0.1
.\.venv\Scripts\python.exe scripts/07_personalization/e2_train.py --mode train --lambda-ctc 0.3

# Read-only optimization diagnostics on four Train batches, CPU + optional BF16 GPU.
# No optimizer steps. Pick a NEW report filename if one already exists.
.\.venv\Scripts\python.exe scripts/07_personalization/e2_preflight.py --cuda-check --report reports/e2_preflight.json
```

Phase 1 weights are 0.03, 0.1 and 0.3. CLI accepts any finite nonnegative
weight, including 0.05 and 0.2 for later runs. It does not automatically sweep
or train. `finalize` is deferred until validation results are reviewed. It now
requires an explicit reviewed list, for example `--mode finalize
--selection-lambdas 0.03 0.05 0.1 0.2 0.3`. Merely completing the current three
runs never selects a final lambda or evaluates Test. Expanding the candidate
grid does not change the data/settings comparison fingerprint.

`configs/e2.yaml` inherits `e1.yaml` without changing it: General checkpoint,
manifest/splits, seed, LR, epochs, batch sizes, accumulation, checkpointing,
generation and metrics match E1. Relative data/output paths resolve against the
repository root. The E2 output tree is `outputs/personalization/E2/KSM_M_45/`:
`lambda_0.03`, `lambda_0.1`, `lambda_0.3`; extra weights get their own directory.
Explicit finalization uses `selected_test`. Existing run or
test directories are refused, not overwritten/resumed. Test loss/metrics are
never produced by the three training commands. Dataset integrity checks inspect
all three splits before training; this is not test performance evaluation.

The subclass `WhisperWithAuxiliaryCTC` has the original Encoder/Decoder and a
trainable `Linear(768, 11183)`. Both backbone and head are saved by Trainer.
Reload using `WhisperWithAuxiliaryCTC.from_pretrained(path)`; the stock AutoModel
class does not reconstruct the auxiliary head. CTC mapping/version/blank/lambda
are embedded in model config. Inference uses `.generate()` and does not execute
the CTC head or use reference labels. Supervised `.forward()` requires all three
CTC tensors and decoder labels. Trainer keeps CTC tensors for loss but removes
them and reference labels from generation inputs.

CTC IDs: blank 0; U+AC00..U+D7A3 -> 1..11172; ASCII `0`..`9` -> 11173..11182.
Raw text is encoded without NFC or numeric verbalization. Only whitespace and
Unicode punctuation are removed. All other unsupported characters are reported
with counts and utterance paths, then stop the run; no row is silently dropped.
E1 decoder labels and CER/WER normalization are reused unchanged. The old
`e2_validate.py` is preserved and still uses its original Hangul-only check;
use the new entry point's validate mode for the finalized numeric vocabulary.

CTC frame lengths use the feature extractor attention mask, not 1500 padded
encoder positions. For the installed Whisper extractor, padded STFT input has
3000 mel frames, valid mel frames are `ceil(resampled_samples / 160)` capped at
3000; the actual conv1/conv2 geometry gives `ceil(valid_mel / 2)`. E1 rejects
audio over 30 seconds; preflight also reports any would-be truncation. The
encoder itself still processes E1's complete padded features (its attention mask
does not remove padded encoder positions). Only CTC loss uses the valid prefix.
Alignment requires `target_length + adjacent_equal_pairs <= valid_encoder_frames`.
`700` needs at least four frames. CTCLoss uses blank 0, mean reduction and
`zero_infinity=True`; impossible targets raise before loss instead of being
silenced by zero_infinity. CTC log-softmax/loss run in float32 under mixed
precision. The large projection is cropped to the batch's maximum valid length.

Loss logs distinguish `asr_loss`, `ctc_loss`, `weighted_ctc_loss`, `total_loss`
and their `eval_` counterparts alongside validation CER/WER. Component logs are
sample-weighted batch means over a logging interval. Trainer's ordinary gradient
accumulation scaling is enabled (`model_accepts_loss_kwargs=False`), since this
loss is a microbatch mean, not a sum divided by `num_items_in_batch`.

All inputs/processor files must already exist locally. No network download,
alignment generation, E3 weighting, or automatic GPU training is performed.
Smoke defaults to CPU; `--smoke-device cuda` explicitly requests a GPU batch test.

Each explicit training run additionally writes:

- `initialization.json`: General path/checksums, seed, lambda, no-resume declaration.
- `training_logs.jsonl`: step logs with separate losses and validation CER/WER.
- `training_audit.json`: sample-weighted epoch Train ASR/CTC/weighted-CTC/total
  means, epoch Validation CER/WER, best epoch/CER/checkpoint, elapsed training
  time (including in-training validation), GPU peak allocated/reserved bytes,
  precision and NaN/Inf events. Losses and gradients are checked every microbatch;
  numerical failure is recorded and stops the run. Epoch losses are the observed
  training losses, not a second evaluation pass over Train.
- `run_summary.json`: completed-run best epoch/checkpoint/CER, resource summary,
  initialization identity and `test_evaluated=false`.

Post-training evaluation of the reloaded best model does not replace the last
epoch's validation measurements. General checksum identity is also checked
across the explicit candidate list before any final Test evaluation.
