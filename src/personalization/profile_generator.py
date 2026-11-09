"""Generate per-speaker syllable error profiles from Whisper predictions.

Run from repository root, for speakers with all profile_train audio available:
    python -m src.personalization.profile_generator \
        --csv data/splits/personalization_split.csv \
        --audio-root ../ieum_datasets --ready-only --device cuda

Only personal_split == 'profile_train' rows are used. Predictions are cached.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import librosa
import soundfile as sf
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
)

from .error_profile import build_raw_error_profile, filter_error_profile

LOG = logging.getLogger(__name__)
FIELDS = ("speaker_id", "batch", "audio_member", "reference", "prediction")


def row_key(row: dict) -> tuple[str, str]:
    """Stable utterance key independent of old source_wav_path."""
    return (
        row["batch"].strip(),
        row["audio_member"].strip().replace("\\", "/"),
    )


def audio_path_for(row: dict, audio_root: Path) -> Path:
    return audio_root / row["batch"].strip() / row["audio_member"].strip().replace("\\", "/")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="personalization_split.csv")
    p.add_argument("--audio-root", type=Path, required=True, help="root of extracted WAV files")
    p.add_argument("--model-dir", type=Path, default=Path("checkpoints/whisper_small_general_v1/model"))
    p.add_argument("--processor-dir", type=Path, default=Path("checkpoints/whisper_small_general_v1/processor"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/personalization/error_profiles"))
    p.add_argument("--speaker-id", help="process one speaker")
    p.add_argument("--ready-only", action="store_true", help="process only speakers with every profile_train WAV available")
    p.add_argument("--limit", type=int, help="maximum number of NEW predictions (pilot only)")
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--min-ratio", type=float, default=0.1)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--allow-missing", action="store_true", help="skip missing or invalid WAV files")
    return p.parse_args()


def read_training_rows(csv_path: Path, speaker_id: str | None) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"speaker_id", "personal_split", "text", "batch", "audio_member"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        rows = []
        seen = set()
        for row in reader:
            if row["personal_split"].strip() != "profile_train":
                continue
            sid = row["speaker_id"].strip()
            if speaker_id and sid != speaker_id:
                continue
            key = row_key(row)
            if key in seen:
                raise ValueError(f"Duplicate batch/audio pair: {key}")
            seen.add(key)
            if not row["text"].strip():
                raise ValueError(f"Empty reference text for {key}")
            rows.append(row)
    if not rows:
        raise ValueError("No profile_train rows matched the selection")
    return rows


def select_ready_rows(rows: list[dict], audio_root: Path) -> list[dict]:
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"].strip()].append(row)

    ready_speakers = set()
    unavailable = []
    for sid, speaker_rows in sorted(by_speaker.items()):
        missing = sum(not audio_path_for(row, audio_root).is_file() for row in speaker_rows)
        if missing == 0:
            ready_speakers.add(sid)
        else:
            unavailable.append((sid, missing, len(speaker_rows)))

    selected = [row for row in rows if row["speaker_id"].strip() in ready_speakers]
    LOG.info("Ready speakers: %d/%d; selected utterances: %d", len(ready_speakers), len(by_speaker), len(selected))
    for sid, missing, total in unavailable:
        LOG.info("Not ready: %s (%d/%d WAV files missing)", sid, missing, total)
    if not selected:
        raise ValueError("No speakers have all profile_train WAV files available")
    return selected


class WhisperPredictor:
    def __init__(self, model_dir: Path, processor_dir: Path, device: str):
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.tokenizer = WhisperTokenizerFast.from_pretrained(processor_dir)
        with (processor_dir / "processor_config.json").open(encoding="utf-8") as f:
            processor_config = json.load(f)
        self.extractor = WhisperFeatureExtractor(**processor_config["feature_extractor"])
        generation_config = self.model.generation_config
        generation_config.language = "korean"
        generation_config.task = "transcribe"
        generation_config.max_length = 160
        generation_config.do_sample = False
        generation_config.num_beams = 1
        LOG.info("Loaded Whisper on %s", self.device)

    def predict(self, audio_path: Path) -> str:
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if not len(audio) or len(audio) > 30 * 16000:
            raise ValueError(f"Audio must be nonempty and <=30s: {audio_path}")
        extracted = self.extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        with torch.inference_mode():
            ids = self.model.generate(
                input_features=extracted.input_features.to(self.device),
                attention_mask=extracted.attention_mask.to(self.device),
                generation_config=self.model.generation_config,
            )
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


def load_predictions(path: Path) -> dict:
    results = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected prediction CSV columns: {reader.fieldnames}")
        for row in reader:
            key = row_key(row)
            if key in results:
                raise ValueError(f"Duplicate prediction: {key}")
            results[key] = row
    return results


def save_profiles(rows, predictions, output_dir: Path, min_count: int, min_ratio: float, experiment_info: dict):
    pairs_by_speaker = defaultdict(list)
    expected_by_speaker = defaultdict(int)
    for row in rows:
        sid = row["speaker_id"].strip()
        expected_by_speaker[sid] += 1
        key = row_key(row)
        if key in predictions:
            pred = predictions[key]
            if pred["speaker_id"].strip() != sid or pred["reference"] != row["text"].strip():
                raise ValueError(f"Cached prediction mismatch: {key}")
            pairs_by_speaker[sid].append((pred["reference"], pred["prediction"]))

    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    complete = 0
    for sid, expected in sorted(expected_by_speaker.items()):
        pairs = pairs_by_speaker[sid]
        if len(pairs) != expected:
            LOG.info("Skipping incomplete speaker %s: %d/%d", sid, len(pairs), expected)
            continue
        raw = build_raw_error_profile(pairs, supported_reference_syllables=None)
        filtered = filter_error_profile(raw, min_count=min_count, min_ratio=min_ratio)
        result = {
            "speaker_id": sid,
            "split": "profile_train",
            "utterance_count": expected,
            "min_count": min_count,
            "min_ratio": min_ratio,
            "experiment_info": experiment_info,
            "raw_profile": raw,
            "filtered_profile": filtered,
        }
        (profiles_dir / f"{sid}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        complete += 1
    LOG.info("Saved %d complete speaker profiles", complete)


def main():
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.min_count < 1 or not 0 <= args.min_ratio <= 1:
        raise ValueError("min-count must be >=1 and min-ratio must be between 0 and 1")
    if args.ready_only and args.allow_missing:
        raise ValueError("--ready-only and --allow-missing should not be combined")

    experiment_info = {
        "model_dir": str(args.model_dir.resolve()),
        "processor_dir": str(args.processor_dir.resolve()),
        "dataset_csv": str(args.csv.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "model_type": "Whisper Small Encoder-Decoder",
        "profile_source": "whisper_predictions",
        "profile_split": "profile_train",
        "min_count": args.min_count,
        "min_ratio": args.min_ratio,
    }

    rows = read_training_rows(args.csv, args.speaker_id)
    if args.ready_only:
        rows = select_ready_rows(rows, args.audio_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.csv"
    predictions = load_predictions(prediction_path)
    for row in rows:
        key = row_key(row)
        if key in predictions:
            cached = predictions[key]
            if cached["speaker_id"].strip() != row["speaker_id"].strip() or cached["reference"] != row["text"].strip():
                raise ValueError(f"Cached prediction mismatch: {key}")

    pending = [row for row in rows if row_key(row) not in predictions]
    if args.limit is not None:
        pending = pending[:args.limit]
    selected_cached = sum(row_key(row) in predictions for row in rows)
    LOG.info(
        "Selected %d training rows; %d selected cached; %d new pending",
        len(rows), selected_cached, len(pending),
    )

    if pending:
        predictor = WhisperPredictor(args.model_dir, args.processor_dir, args.device)
        with prediction_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                writer.writeheader()
                f.flush()
            for i, row in enumerate(pending, 1):
                path = audio_path_for(row, args.audio_root)
                try:
                    if not path.is_file():
                        raise FileNotFoundError(f"Audio not found: {path}")
                    predicted = predictor.predict(path)
                except (FileNotFoundError, ValueError) as e:
                    if not args.allow_missing:
                        raise
                    LOG.warning("Skipped %s: %s", path, e)
                    continue
                result = {
                    "speaker_id": row["speaker_id"].strip(),
                    "batch": row["batch"].strip(),
                    "audio_member": row["audio_member"].strip(),
                    "reference": row["text"].strip(),
                    "prediction": predicted,
                }
                writer.writerow(result)
                f.flush()
                predictions[row_key(row)] = result
                if i % 20 == 0 or i == len(pending):
                    LOG.info("Processed %d/%d new utterances", i, len(pending))

    save_profiles(rows, predictions, args.output_dir, args.min_count, args.min_ratio, experiment_info)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
