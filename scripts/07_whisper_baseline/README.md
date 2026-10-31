# Whisper Small Baseline: General Test input

`general_test.py` uses Python standard libraries only. It never imports or runs
the preprocessing pipeline and never extracts or modifies source archives.

Inputs under `--data-root`:

- `manifest_v1/all_utterances.csv`
- `split_v1/speaker_split.csv`
- `processed_candidate2_archives/BATCH_*.zip`

The manifest's historical `audio_zip` absolute paths are relocated using their
exact batch ZIP basename. `audio_member` and `transcript_member` are used exactly
as recorded. There is no fallback to partial local batches or extracted folders.

Run a read-only source audit (the report must be a new file):

```powershell
& .\.venv_local_pilot\Scripts\python.exe -B .\scripts\07_whisper_baseline\general_test.py --report .\results\whisper_baseline\general_test_audit_20261027.json
```

The audit checks every manifest row against the speaker split, duplicate speaker
and utterance keys, per-speaker segment counts, the expected 46,834 test rows and
90 test speakers, and all manifest WAV/TXT links in ZIP central directories.
It also reads one test WAV/TXT pair per test-bearing ZIP, checking ZIP CRC,
WAV duration and transcript agreement. It does not read every audio payload or
establish acoustic/text alignment. A failure raises an exception; no PASS report
is written. Source ZIP provenance/completion is not inferred from filenames alone.

For subsequent inference, import `GeneralTest` from this directory:

```python
dataset = GeneralTest()  # or GeneralTest(data_root)
for row, wav_bytes in dataset.iter_audio():
    # Decode WAV bytes and resample in memory for Whisper input.
    # Preserve row metadata and record predictions in a separate results folder.
    pass
```

The iterator covers all test rows, groups reads by ZIP, and holds only one audio
payload at a time. Reads verify CRC and errors propagate without silently skipping
utterances. Consumers must exhaust the iterator for complete coverage. No model
inference is implemented or started by this input loader.

## Environment

Use Python 3.11.9 and a separate `.venv_whisper_baseline`. Tested GPU: RTX 5060
(8 GB, compute capability 12.0). PyTorch cu128 supports Blackwell; the installed
driver reports CUDA support 13.3, while PyTorch uses its bundled CUDA 12.8 runtime.
No torchvision, torchaudio, datasets, librosa, or training packages are needed.

```powershell
python -m venv .venv_whisper_baseline
& .\.venv_whisper_baseline\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
& .\.venv_whisper_baseline\Scripts\python.exe -m pip install -r .\scripts\07_whisper_baseline\requirements.txt
& .\.venv_whisper_baseline\Scripts\python.exe -m pip check
```

Sources: [PyTorch cu128 packages](https://pytorch.org/get-started/previous-versions/),
[Blackwell support](https://pytorch.org/blog/pytorch-2-7/),
[Transformers 4.53.3 Whisper generation](https://huggingface.co/docs/transformers/v4.53.3/model_doc/whisper).

## Selection and smoke inference

Run from the project root. Paths can be changed with `--data-root`. Source CSVs
are not rewritten. Model safetensors/tokenizer are cached under `models/huggingface`.
The resolved model commit, source hashes, selection hash, code hashes and runtime
versions are saved. Resuming with different settings/code/versions is refused.

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_XET = '1'
& .\.venv_whisper_baseline\Scripts\python.exe -B .\scripts\07_whisper_baseline\prepare_inputs.py --output .\results\whisper_baseline\smoke_selection_20261027.json
.\scripts\07_whisper_baseline\run_smoke.ps1 -StopAfter 6
.\scripts\07_whisper_baseline\run_smoke.ps1
```

Selection: six duration quantiles per disability group, with distinct speaker IDs
(18 utterances). Inference uses FP16 CUDA, batch size 1, Korean transcription,
greedy decoding, and no reference prompts or fine-tuning. WAVs are downmixed and
resampled to 16 kHz in memory. Each selected ZIP transcript must match manifest
text and WAV duration must match metadata before inference. These checks do not
establish independently verified acoustic/text alignment; listening review is
still needed for that claim.

## Outputs, metrics and restart

- `checkpoint.sqlite3`: authoritative transactional per-attempt records. Every
  completed utterance commits with synchronous FULL. Completed attempts survive
  process termination; an in-flight utterance is processed again on restart.
- `predictions.csv`: latest attempt per utterance with reference, hypothesis,
  identity, timing, memory, CRC-read validation and WAV SHA256.
- `failures.csv`: latest unresolved failures. `--retry-failures` retries those;
  earlier attempt history remains in SQLite. Check coverage and failures before
  reporting aggregate metrics. Failures are excluded from CER/WER denominators.
- `scores.json`: corpus micro CER/WER, speaker and disability group aggregates,
  successful/failed/pending counts. CSV/JSON exports refresh every 10 utterances
  and on graceful shutdown. SQLite may be newer after forced termination; rerun
  the same command to regenerate exports.
- `run_config.json`, `session_*.json`: reproducibility settings, completed/skipped
  counts and per-invocation timing. A file lock prevents concurrent writers.

Primary normalized scores use NFC, lowercasing, punctuation-to-space conversion
and whitespace collapsing. CER removes whitespace; WER splits on whitespace.
Numbers remain unchanged. Secondary raw scores preserve punctuation and case
(CER still ignores whitespace). Scores sum edit counts across utterances rather
than averaging utterance rates. Insertions can make CER/WER exceed 100%.
Empty-reference individual rates are null; their insertion counts still enter
corpus totals. GPU peaks are PyTorch allocated/reserved memory, not total VRAM
including desktop/driver. Inference timing synchronizes CUDA and excludes WAV
loading/features; `loading_sec` includes those and host-to-device transfer.

```powershell
& .\.venv_whisper_baseline\Scripts\python.exe -B .\scripts\07_whisper_baseline\test_baseline.py
```

## Full evaluation (manual command only; not run during smoke testing)

Prepare all test rows, then run using a NEW output directory. More than 20 rows
requires the explicit `--allow-full-test` flag. The same command resumes later;
add `--retry-failures` to retry failed rows. Pin the smoke run's model revision.

```powershell
& .\.venv_whisper_baseline\Scripts\python.exe -B .\scripts\07_whisper_baseline\prepare_inputs.py --full-test --output .\results\whisper_baseline\general_test_selection.json
$revision = (Get-Content .\results\whisper_baseline\smoke_20261027_v2\run_config.json -Raw | ConvertFrom-Json).revision
& .\.venv_whisper_baseline\Scripts\python.exe -B .\scripts\07_whisper_baseline\run_baseline.py --selection .\results\whisper_baseline\general_test_selection.json --output .\results\whisper_baseline\general_test_full --revision $revision --allow-full-test
```

Keep Drive mounted and connected; streamed ZIP reads can hydrate large archives
into Drive's local cache. Reserve disk space and avoid system sleep. General Test
contains 90 pseudo-speakers; neither real-person identity nor cross-batch audio
content duplication has been established. Smoke scores are diagnostic only.

The completed 2026-10-27 smoke results are under
`results/whisper_baseline/smoke_20261027_v2/`: `REPORT.md`, `smoke_summary.json`,
`predictions.csv`, `scores.json`, `failures.csv` and `review_audio/review_index.csv`.
The earlier `smoke_20261027` directory preserves a development diagnostic run;
use the v2 directory for final metrics. `summarize_smoke.py` builds timing and
resume summaries; `export_review.py` exports <=20 exact inference WAVs for review.
Time estimates are planning heuristics from 18 samples, not confidence intervals.

## External-drive workflow

`run_external_evaluation.py` prepares `D:\ieum_whisper_baseline`, reuses the seven
ZIPs under `D:\ieum_local_pilot\archives`, and sequentially copies only missing ZIPs
from Drive. It preserves interrupted transfer files and never overwrites a
different completed ZIP. CSVs are copied unchanged. `archive_locations.json` maps
each batch to its actual external-drive ZIP, so inference reads no Drive audio.
Transfer checks cover size and all ZIP central-directory entries; SHA256 is
recorded during copying. Full payload CRC is checked when audio is read for ASR.
This is not a second complete checksum pass over every copied ZIP.

After all data is ready, a local 18-utterance smoke run must succeed before the
46,834-utterance evaluation starts. SQLite commits every utterance; full CSV/JSON
exports refresh every 500 utterances. Transcript entries are loaded once per ZIP
to reduce HDD seeks between audio and text. Existing v2 results remain preserved;
new code fingerprints intentionally require a new run directory.

Monitor under `results/whisper_baseline/`:

- `external_evaluation_status.json`: download progress and workflow stage
- `external_evaluation.log` / `.stderr.log`: process output/errors
- `external_evaluation.pid`: launched process ID
- `smoke_external_20261027/`: local smoke results
- `general_test_external_20261027/`: full test checkpoint, predictions and scores

To resume the workflow, run the same script with `.venv_whisper_baseline` Python.
Completed ZIPs are reused after index checks and completed inference rows skipped.
The cleanup task was stopped at the user's direction; its separate interruption
summary is under `cleanup_20261027_verified/interrupted_summary.json`.

## Evaluate available ZIPs first (2026-10-28)

`run_available_evaluation.py` validates member links in the completed local ZIPs
and runs `run_baseline.py --available-only`. Missing ZIPs are deferred, not failed.
The immutable selection still contains all 46,834 test rows, allowing the SAME
checkpoint to extend as ZIPs arrive. Run the same launcher again to resume;
already successful utterances are skipped. No download or cleanup is started.

Results: `results/whisper_baseline/general_test_external_20261028/`.
Workflow status: `available_evaluation_status.json`; logs:
`available_evaluation_20261028.log` and `.stderr.log`. These are separate from the
failed download workflow's `external_evaluation_status.json` and prior smoke data.
Partial scores describe only available, completed utterances, not the full test.
