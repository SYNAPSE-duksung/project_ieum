# Frozen E1/E3 pilot runner

Run from `project_ieum`. This entry point never evaluates Personal Test.

```powershell
.\.venv\Scripts\python.exe scripts/07_personalization/pilot_runner.py --mode plan
.\.venv\Scripts\python.exe scripts/07_personalization/pilot_runner.py --mode preflight
.\.venv\Scripts\python.exe scripts/07_personalization/pilot_runner.py --mode smoke --speakers YHS_F_26
.\.venv\Scripts\python.exe scripts/07_personalization/test_pilot_runner.py
```

After approval, start the 20 serial jobs:

```powershell
.\.venv\Scripts\python.exe scripts/07_personalization/pilot_runner.py --mode train
```

Completed jobs are skipped on rerun. A failure stops the queue. Inspect its
`status.json` and `console-*.log`; do not change learning settings to mask a failure.
Use `--skip-failed` to leave incomplete jobs untouched, or `--retry-failed` to start
them from General in a new attempt. Optimizer/scheduler/RNG states are deliberately
not saved, so exact mid-job resume is unavailable. All previous attempts remain.
A parent killed forcibly leaves `runner.lock`: inspect its PID and any running
worker before an operator removes a stale lock. Normal termination removes it.

An alternate volume can be selected with
`--output-root D:\ieum_experiments\pilot_e1_e3_v1` (verify actual free space first).
The final directory name must be `pilot_e1_e3_v1`; existing unmarked directories and
protected experiment roots are refused. Use the same option when resuming.

## Reproducibility and storage

Selection and original source SHA256 values are checked against the frozen report.
Original E1 YAML is read directly. Both original trainer factories, data loaders,
collators and metric functions are reused. E3 keeps lambda=.1, margin=.1,
legacy_e1 accumulation; seed=42 and data_seed=None. One fresh subprocess/model per
job isolates model, optimizer, scheduler, seed and GPU memory.

Only persistence differs: save_strategy=best, save_only_model=True and
save_total_limit=1. Validation remains every epoch with minimum CER selection.
Each attempt retains one canonical `checkpoints/checkpoint-N`; `summary.json`
points to it. There is no duplicated `best_model`. The processor, model config,
training arguments, configuration, trainer state, epoch audit, events and console
log are retained. HF rotates only new checkpoints within this attempt. General
and historical KSM/E1/E2/E3 results are not touched.

Space guards run before training and each epoch evaluation. They include actual
new-root usage (including failed attempts), remaining models, two model sizes of
serialization/rotation headroom, 1GiB metadata allowance and 5GiB free reserve.
The overall envelope is 30GiB. A failed capacity check stops execution rather than
deleting results. Capacity is an estimate; unrelated disk use can still exhaust
the volume between checks.

Audit files contain epoch ASR/Correction/Total losses, candidate usage and margin
activation, Validation CER/WER, best epoch/CER, training seconds and peak GPU
memory. Errors are captured in job status and console logs. Completed jobs have
`test_evaluated=false`. Test evaluation is intentionally a later separate stage.

## Verification scope

`preflight` checks all selected speaker/cache/split connections and compares 24
learning/generation arguments. `smoke` compares the same two real YHS Train
utterances using explicit configured CUDA autocast and backward through each
trainer. It includes an E1-repeat numerical baseline. It does not update weights.
CPU tests exercise two synthetic optimizer steps, validation-best selection,
checkpoint replacement, absence of optimizer states and exact weight reload.
This is not a full-duration or all-speaker GPU memory stress test.

Historical legacy_e1 scaling is preserved intentionally, including the original
unnormalized microbatch accumulation. Standard mean accumulation is not silently
substituted. BF16 GPU kernels are not bitwise deterministic; inspect relative
gradient differences against the E1-repeat baseline.
