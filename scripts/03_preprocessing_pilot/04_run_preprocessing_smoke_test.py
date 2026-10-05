"""Three-recording VAD/transcript feasibility test, executed only by the user.

Reference: yoona-J/Diagnosis-Aware_Multitask_Fine-Tuning_of_Whisper_for_DSR
Commit: 4624249a217e85ff91752c2f689eca4ede15686a
Preprocess/[Distribution]Preprocessing_Stroke.ipynb cells 10, 11, 22, 30, 31.
The VAD body below is copied from cell 10; transcript rules from cells 30/31.
No resampling, denoising, feature extraction, dataset creation or splitting.
"""
import argparse
import csv
import importlib.metadata
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

COMMIT = "4624249a217e85ff91752c2f689eca4ede15686a"
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT / "results/preprocessing_pilot/smoke_test"
VAD1 = dict(frame_ms=500, silence_duration_sec=5, alpha=0.3)
VAD2 = dict(frame_ms=600, silence_duration_sec=3, alpha=0.4)
EXPECTED = {
    "LHJ_F_79": ("뇌신경장애", "ID-01-15-N-LHJ-03-F-79-KK.wav"),
    "JSM_M_35": ("언어·청각 장애", "ID-02-27-N-JSM-01-01-M-35-KK.wav"),
    "KYY_M_58": ("후두장애", "ID-03-31-N-KYY-01-01-M-58-KK2.wav"),
}
SAFETY_FIX = (
    "explicit_manifest_basename_File_id_pairing;"
    "exact_three_recording_allowlist;"
    "per_recording_exception_isolation;"
    "preserve_separate_VAD_attempt_artifacts;"
    "refuse_nonempty_output_directory"
)
RESULT_FIELDS = [
    "speaker_id", "disability_group", "recording_id", "audio_duration_sec",
    "text_segment_count", "vad1_audio_segment_count", "vad1_match",
    "vad2_audio_segment_count", "vad2_match", "final_status", "failure_reason",
    "valid_1_30s_segments", "total_segments", "valid_segment_ratio",
    "processing_time_sec", "implementation_safety_fix",
    "retry_needed", "final_pass", "pairing_method",
    "actual_sampling_rate", "channels", "sample_width_bytes",
    "json_audio_duration_sec", "duration_difference_sec",
    "vad1_processing_time_sec", "vad2_processing_time_sec",
]
LOG = logging.getLogger("smoke_test")


# Exact algorithm from reference cell 10; do not replace with fixed chunking.
def vad_segment_by_energy(audio_segment, frame_ms=500, silence_duration_sec=5, alpha=0.3):
    frame_len = frame_ms
    total_len = len(audio_segment)
    energy_values = []
    for i in range(0, total_len, frame_len):
        frame = audio_segment[i:i+frame_len]
        energy_values.append(frame.rms)
    mean_energy = sum(energy_values) / len(energy_values)
    threshold = mean_energy * alpha
    silence_flags = [rms < threshold for rms in energy_values]
    frame_duration_sec = frame_ms / 1000
    min_silence_frames = int(silence_duration_sec / frame_duration_sec)
    segments = []
    is_silent = False
    start = 0
    for idx, silent in enumerate(silence_flags):
        if not is_silent and silent:
            silence_run = silence_flags[idx:idx+min_silence_frames]
            if len(silence_run) == min_silence_frames and all(silence_run):
                end = idx * frame_len
                if end - start > 0:
                    segments.append((start, end))
                is_silent = True
        elif is_silent and not silent:
            start = idx * frame_len
            is_silent = False
    if not is_silent and start < total_len:
        segments.append((start, total_len))
    if not segments:
        print("VAD Failed", flush=True)
    return segments


def split_transcript(transcript):
    """Reference cell 30. Slash has priority even if punctuation also exists."""
    transcript = str(transcript).strip()
    if "/" in transcript:
        segments = transcript.split("/")
    elif re.search(r"[\.?!]", transcript):
        segments = re.split(r"[\.?!]", transcript)
    else:
        segments = transcript.split()
    return [s.strip() for s in segments if s.strip()]


def clean_transcript(text):
    """Reference cell 31: no additional normalization or whitespace correction."""
    return re.sub(r"[+\*\(\)\?!,\.~\-']", "", text)


class Heartbeat:
    """Logging only; does not change VAD decisions or read audio."""
    def __init__(self, phase, interval=10):
        self.phase = phase
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        started = time.perf_counter()
        while not self.stop.wait(self.interval):
            LOG.info("  working: %s (%.1fs elapsed)", self.phase,
                     time.perf_counter() - started)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=1)


def load_manifest(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        needed = {"speaker_id", "disability_group", "wav_path", "json_path"}
        if not needed.issubset(reader.fieldnames or []):
            raise ValueError("Manifest needs: " + ", ".join(sorted(needed)))
        rows = list(reader)
    if len(rows) != 3 or {r["speaker_id"] for r in rows} != set(EXPECTED):
        raise ValueError("Manifest must contain exactly the three allowed speakers, once each.")
    return rows


def input_path(value, manifest):
    if not value.strip():
        raise ValueError("Empty input path; edit manifest.")
    if "EDIT_LOCAL_PATH" in value or "REPLACE" in value:
        raise ValueError("Placeholder input path; edit wav_path/json_path in manifest.")
    path = Path(value.strip()).expanduser()
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def pair_inputs(row, manifest):
    group, expected_wav = EXPECTED[row["speaker_id"]]
    if row["disability_group"] != group:
        raise ValueError("disability_group differs from the allowed manifest group.")
    wav = input_path(row["wav_path"], manifest)
    label = input_path(row["json_path"], manifest)
    if wav.name != expected_wav or label.name != Path(expected_wav).with_suffix(".json").name:
        raise ValueError("Input basename is not the selected recording or matching JSON.")
    if not wav.is_file() or not label.is_file():
        raise FileNotFoundError(f"Missing WAV/JSON: {wav} | {label}")
    with label.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON must be an object.")
    if data.get("File_id") != wav.name:
        raise ValueError(f"File_id mismatch: {data.get('File_id')!r} != {wav.name!r}")
    if "Transcript" not in data or not isinstance(data["Transcript"], str):
        raise ValueError("JSON Transcript must exist and be a string.")
    return wav, label, data


def make_result(row):
    result = dict.fromkeys(RESULT_FIELDS, "")
    result.update(speaker_id=row["speaker_id"], disability_group=row["disability_group"],
                  recording_id=Path(EXPECTED[row["speaker_id"]][1]).stem,
                  final_status="ERROR", implementation_safety_fix=SAFETY_FIX,
                  retry_needed=False, pairing_method="manifest+basename+JSON.File_id")
    return result


def save_pass(audio, segments, texts, directory, person_code, pass_id, matched):
    """Keep mismatch WAV and unpaired TXT separate: never fabricate alignment."""
    directory.mkdir(parents=True, exist_ok=False)
    transcript_dir = directory / "transcript_segments"
    transcript_dir.mkdir()
    for i, text in enumerate(texts):
        name = f"output_PN_{person_code}_{i}.txt" if matched else f"text_segment_{i}.txt"
        (transcript_dir / name).write_text(clean_transcript(text), encoding="utf-8")
    with (directory / "segments.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pass", "segment_index", "wav_name", "start_ms", "end_ms",
            "duration_sec", "valid_1_30s", "transcript_path", "pairing_status"])
        writer.writeheader()
        for i, (start, end) in enumerate(segments):
            segment = audio[start:end]
            seconds = segment.duration_seconds
            name = f"output_PN_{person_code}_{i}.wav"
            with Heartbeat(f"export {pass_id} segment {i+1}/{len(segments)}"):
                with segment.export(str(directory / name), format="wav") as exported:
                    pass
            writer.writerow(dict(
                **{'pass': pass_id}, segment_index=i, wav_name=name, start_ms=start,
                end_ms=end, duration_sec=seconds, valid_1_30s=1 <= seconds <= 30,
                transcript_path=f"transcript_segments/{Path(name).with_suffix('.txt').name}" if matched else "",
                pairing_status="COUNT_MATCH_ORDER_ONLY" if matched else "UNPAIRED"))
            f.flush()
            LOG.info("  exported %s segment %d/%d: %.3fs", pass_id, i+1,
                     len(segments), seconds)


def process_recording(row, manifest, output, audio_class, index):
    result = make_result(row)
    started = time.perf_counter()
    LOG.info("[%d/3] %s", index, row["speaker_id"])
    LOG.info("recording: %s", EXPECTED[row["speaker_id"]][1])
    LOG.info("implementation_safety_fix: %s", SAFETY_FIX)
    try:
        wav, label, data = pair_inputs(row, manifest)
        texts = split_transcript(data["Transcript"])
        result["text_segment_count"] = len(texts)
        LOG.info("pairing: File_id + explicit filenames OK")
        LOG.info("loading WAV...")
        with Heartbeat("loading WAV"):
            audio = audio_class.from_file(str(wav))
        if len(audio) == 0:
            raise ValueError("EMPTY_AUDIO: no frames available for reference VAD.")
        result.update(audio_duration_sec=audio.duration_seconds,
                      actual_sampling_rate=audio.frame_rate, channels=audio.channels,
                      sample_width_bytes=audio.sample_width)
        json_duration = data.get("playTime", data.get("Meta_info", {}).get("PlayTime"))
        if json_duration is not None:
            result["json_audio_duration_sec"] = float(json_duration)
            result["duration_difference_sec"] = audio.duration_seconds - float(json_duration)
        LOG.info("duration: %.3fs; sampling_rate: %d; channels: %d (unchanged)",
                 audio.duration_seconds, audio.frame_rate, audio.channels)
        chosen = []
        recording_dir = output / result["recording_id"]
        recording_dir.mkdir(exist_ok=False)
        suffix = re.fullmatch(r"ID-\d{2}-\d{2}-N-(.+)\.wav", wav.name)
        if not suffix:
            raise ValueError("Reference recording filename regex failed.")
        person_code = suffix.group(1)
        for pass_number, settings in ((1, VAD1), (2, VAD2)):
            if pass_number == 2:
                result["retry_needed"] = True
            LOG.info("VAD pass 1" if pass_number == 1 else "VAD retry (pass 2)")
            LOG.info("  frame=%dms; silence=%ss; threshold=mean RMS x %.1f",
                     settings["frame_ms"], settings["silence_duration_sec"], settings["alpha"])
            pass_start = time.perf_counter()
            with Heartbeat(f"VAD pass {pass_number}"):
                segments = vad_segment_by_energy(audio, **settings)
            result[f"vad{pass_number}_processing_time_sec"] = time.perf_counter() - pass_start
            matched = len(segments) == len(texts) and len(segments) > 0
            result[f"vad{pass_number}_audio_segment_count"] = len(segments)
            result[f"vad{pass_number}_match"] = matched
            result["final_pass"] = pass_number
            chosen = segments
            result["total_segments"] = len(chosen)
            valid = sum(1 <= audio[s:e].duration_seconds <= 30 for s, e in chosen)
            result["valid_1_30s_segments"] = valid
            result["valid_segment_ratio"] = valid / len(chosen) if chosen else 0.0
            LOG.info("audio segments: %d", len(segments))
            LOG.info("text segments: %d", len(texts))
            LOG.info("MATCH" if matched else "MISMATCH")
            save_pass(audio, segments, texts, recording_dir / f"vad{pass_number}",
                      person_code, f"vad{pass_number}", matched)
            if matched:
                result["final_status"] = "SUCCESS"
                result["failure_reason"] = ""
                break
        else:
            result["final_status"] = "NEEDS_MANUAL_REVIEW"
            result["failure_reason"] = (
                f"VAD_TEXT_COUNT_MISMATCH_OR_EMPTY: text={len(texts)}, "
                f"vad1={result['vad1_audio_segment_count']}, vad2={len(chosen)}; "
                "no automatic alignment performed")
        LOG.info("1~30s: %s/%s; ratio=%.4f (last VAD pass)",
                 result["valid_1_30s_segments"], result["total_segments"],
                 result["valid_segment_ratio"])
        if result["final_status"] == "SUCCESS":
            LOG.info("SUCCESS means nonempty count agreement and completed exports, not verified semantic alignment.")
    except Exception as exc:
        result["final_status"] = "ERROR"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        LOG.exception("recording failed; continuing to the next recording")
        exception_details = traceback.format_exc()
        try:
            (output / f"{result['recording_id']}_exception.txt").write_text(
                exception_details, encoding="utf-8")
        except Exception:
            LOG.exception("Could not save exception details; recording failure remains in the result row")
    finally:
        result["processing_time_sec"] = time.perf_counter() - started
        LOG.info("status: %s", result["final_status"])
        if result["failure_reason"]:
            LOG.info("failure_reason: %s", result["failure_reason"])
        LOG.info("elapsed: %.3fs", result["processing_time_sec"])
    return result


def check_imports():
    """Safe dependency probe: no manifest, output directory or audio access."""
    try:
        from pydub import AudioSegment
        print("pydub import: OK; version=" + importlib.metadata.version("pydub"), flush=True)
        print("converter=" + AudioSegment.converter, flush=True)
        return 0
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"pydub import unavailable: {exc}", flush=True)
        print("Install pydub==0.25.1; Python >=3.13 also needs audioop-lts.", flush=True)
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="CSV with the three explicit WAV/JSON pairs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="New/empty output folder; existing results are never overwritten.")
    parser.add_argument("--check-imports", action="store_true",
                        help="Only check pydub import; never load inputs or create results.")
    args = parser.parse_args()
    if args.check_imports:
        return check_imports()
    if args.manifest is None:
        parser.error("--manifest is required for a real run.")
    manifest = args.manifest.resolve()
    try:
        rows = load_manifest(manifest)
        output = args.output_dir.resolve()
        if output == PROJECT or output == manifest.parent:
            raise ValueError("Use a dedicated output directory.")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Output directory is not empty. Choose a new --output-dir for a rerun.")
        for row in rows:
            for key in ("wav_path", "json_path"):
                value = row[key].strip()
                if value and "EDIT_LOCAL_PATH" not in value and "REPLACE" not in value:
                    if input_path(value, manifest).is_relative_to(output):
                        raise ValueError("Input must be outside the output directory.")
        from pydub import AudioSegment
    except Exception as exc:
        print(f"Preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    formatter = logging.Formatter("%(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOG.addHandler(console)
    LOG.addHandler(file_handler)
    configuration = dict(reference_commit=COMMIT, vad1=VAD1, vad2=VAD2,
                         manifest=str(manifest), python=sys.version,
                         pydub=importlib.metadata.version("pydub"),
                         implementation_safety_fix=SAFETY_FIX,
                         success_definition="Nonempty segment count agreement; not semantic alignment",
                         ratio_definition="1..30 seconds inclusive, unresampled audio duration; last attempted VAD pass",
                         timing_definition="Pairing, JSON, WAV decode, VAD and artifact export; no downloads",
                         retry_definition="Only when first VAD count is mismatched or empty",
                         inputs=rows)
    (output / "run_config.json").write_text(json.dumps(configuration, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    outcomes = []
    with (output / "preprocessing_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        f.flush()
        for index, row in enumerate(rows, 1):
            result = process_recording(row, manifest, output, AudioSegment, index)
            writer.writerow(result)
            f.flush()
            os.fsync(f.fileno())
            outcomes.append(result["final_status"])
    LOG.info("Results: %s", output / "preprocessing_results.csv")
    for handler in list(LOG.handlers):
        handler.flush()
        handler.close()
        LOG.removeHandler(handler)
    return 0 if all(status == "SUCCESS" for status in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())



