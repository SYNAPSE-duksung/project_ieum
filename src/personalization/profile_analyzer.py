"""Analyze cached Whisper predictions and speaker Error Profiles (profile_train only).

Run from repository root:
    python -m src.personalization.profile_analyzer \
      --input-dir outputs/personalization/error_profiles \
      --output-dir outputs/personalization/error_profiles/analysis \
      --dataset-csv data/splits/personalization_split.csv

Rerun after generating more profiles; no Whisper inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from .error_profile import align_syllables, normalize_for_syllable_alignment

LOG = logging.getLogger(__name__)
ERROR_TYPES = ("substitution", "deletion", "insertion")
PREDICTION_COLUMNS = {"speaker_id", "batch", "audio_member", "reference", "prediction"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/personalization/error_profiles"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-csv", type=Path, default=Path("data/splits/personalization_split.csv"))
    parser.add_argument("--snapshot", action="store_true", help="also save this run under snapshots/YYYYMMDD_HHMMSS")
    return parser.parse_args()


def read_csv(path: Path, required: set[str]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def utterance_key(row: dict) -> tuple[str, str]:
    return (row["batch"].strip(), row["audio_member"].strip().replace("\\", "/"))


def word_tokens(text: str) -> list[str]:
    """Whitespace-separated Korean ASR WER; punctuation removed, spacing preserved."""
    text = re.sub(r"[^\uac00-\ud7a3a-zA-Z0-9\s]", "", text.lower())
    return text.split()


def edit_distance(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, pred in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != pred)))
        previous = current
    return previous[-1]


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ratio(numerator: int, denominator: int):
    return round(numerator / denominator, 6) if denominator else None


def main():
    args = arguments()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "analysis"
    profiles_dir = input_dir / "profiles"
    if not profiles_dir.is_dir():
        raise FileNotFoundError(profiles_dir)

    # The dataset manifest is authoritative for profile_train membership and completeness.
    manifest = read_csv(args.dataset_csv, {"speaker_id", "personal_split", "text", "batch", "audio_member"})
    expected = defaultdict(dict)
    for row in manifest:
        if row["personal_split"].strip() != "profile_train":
            continue
        sid = row["speaker_id"].strip()
        key = utterance_key(row)
        if key in expected[sid]:
            raise ValueError(f"Duplicate manifest key for {sid}: {key}")
        expected[sid][key] = row["text"].strip()

    cached = read_csv(input_dir / "predictions.csv", PREDICTION_COLUMNS)
    predictions = defaultdict(dict)
    seen = set()
    for row in cached:
        sid = row["speaker_id"].strip()
        key = utterance_key(row)
        if key in seen:
            raise ValueError(f"Duplicate prediction key: {key}")
        seen.add(key)
        if sid not in expected or key not in expected[sid]:
            LOG.warning("Ignoring prediction not in profile_train manifest: %s %s", sid, key)
            continue
        if row["reference"] != expected[sid][key]:
            raise ValueError(f"Cached reference differs from manifest for {sid} {key}")
        predictions[sid][key] = row

    profiles = {}
    for path in sorted(profiles_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            profile = json.load(handle)
        sid = profile.get("speaker_id")
        if sid != path.stem:
            raise ValueError(f"Speaker ID/file name mismatch: {path}")
        if sid not in expected:
            LOG.warning("Ignoring profile not in manifest: %s", path)
            continue
        if profile.get("split") != "profile_train":
            raise ValueError(f"Unexpected split for {sid}: {profile.get('split')}")
        if profile.get("utterance_count") != len(expected[sid]):
            raise ValueError(f"Profile utterance_count mismatch for {sid}")
        if len(predictions[sid]) != len(expected[sid]):
            LOG.warning("Skipping %s: JSON exists but only %d/%d predictions available", sid, len(predictions[sid]), len(expected[sid]))
            continue
        profiles[sid] = profile

    if not profiles:
        raise ValueError("No complete speaker profiles found; check profiles/ and predictions.csv")

    speaker_rows = []
    type_rows = []
    syllable_rows = []
    total_s = total_d = total_i = total_chars = total_word_edits = total_words = 0
    empty_count = 0
    pair_repeated_speakers = 0
    syllable_repeated_speakers = 0

    for sid, profile in sorted(profiles.items()):
        records = [predictions[sid][key] for key in expected[sid]]
        ref_syllables = Counter()
        by_type = Counter()
        by_reference_type = defaultdict(Counter)
        syllable_predictions = defaultdict(Counter)
        char_edits = 0
        word_edits = 0
        char_total = 0
        word_total = 0
        empty_ref_utterances = 0

        for record in records:
            ref = normalize_for_syllable_alignment(record["reference"])
            hyp = normalize_for_syllable_alignment(record["prediction"])
            ref_syllables.update(ref)
            char_total += len(ref)
            if not ref:
                empty_ref_utterances += 1
            # Use the exact alignment routine used by the Error Profile generator.
            for a, b in align_syllables(ref, hyp):
                if a == b:
                    continue
                char_edits += 1
                if a == "<EMPTY>":
                    kind = "insertion"
                elif b == "<EMPTY>":
                    kind = "deletion"
                else:
                    kind = "substitution"
                by_type[kind] += 1
                if a != "<EMPTY>":
                    by_reference_type[a][kind] += 1
                    syllable_predictions[a][b] += 1
            words_ref = word_tokens(record["reference"])
            words_hyp = word_tokens(record["prediction"])
            word_total += len(words_ref)
            word_edits += edit_distance(words_ref, words_hyp)

        raw = profile.get("raw_profile", [])
        filtered = profile.get("filtered_profile", [])
        raw_events = sum(int(x["count"]) for x in raw)
        if raw_events != char_edits:
            raise ValueError(f"Error event mismatch for {sid}: profile={raw_events}, aligned={char_edits}")
        min_count = int(profile.get("min_count", 2))
        min_ratio = float(profile.get("min_ratio", 0.1))
        pair_repeated = sum(int(x["count"]) >= 2 for x in raw)
        syllable_repeated = sum(sum(types.values()) >= 2 for types in by_reference_type.values())
        # Alternative aggregation: reference syllable-level substitution + deletion.
        syllable_qualified = sum(
            1 for syll, types in by_reference_type.items()
            if sum(types.values()) >= min_count
            and sum(types.values()) / ref_syllables[syll] >= min_ratio
        )
        empty_count += not bool(filtered)
        pair_repeated_speakers += pair_repeated > 0
        syllable_repeated_speakers += syllable_repeated > 0

        speaker_rows.append({
            "speaker_id": sid, "utterances": len(records),
            "reference_syllables": char_total, "reference_words": word_total,
            "cer": ratio(char_edits, char_total), "wer": ratio(word_edits, word_total),
            "syllable_edits": char_edits, "word_edits": word_edits,
            "substitutions": by_type["substitution"], "deletions": by_type["deletion"],
            "insertions": by_type["insertion"], "raw_profile_items": len(raw),
            "raw_error_events": raw_events, "filtered_profile_items": len(filtered),
            "filtered_error_events": sum(int(x["count"]) for x in filtered),
            "filtered_profile_empty": int(not bool(filtered)),
            "repeated_error_pairs": pair_repeated,
            "repeated_reference_syllables": syllable_repeated,
            "qualified_reference_syllables": syllable_qualified,
            "min_count": min_count, "min_ratio": min_ratio,
            "empty_reference_utterances": empty_ref_utterances,
        })
        for kind in ERROR_TYPES:
            type_rows.append({
                "speaker_id": sid, "error_type": kind, "events": by_type[kind],
                "raw_profile_items": sum(x["error_type"] == kind for x in raw),
                "filtered_profile_items": sum(x["error_type"] == kind for x in filtered),
            })
        for syll in sorted(by_reference_type):
            counts = by_reference_type[syll]
            count = sum(counts.values())
            support = ref_syllables[syll]
            syllable_rows.append({
                "speaker_id": sid, "reference_syllable": syll,
                "reference_count": support, "error_count": count,
                "error_ratio": ratio(count, support),
                "substitutions": counts["substitution"], "deletions": counts["deletion"],
                "distinct_predictions": len(syllable_predictions[syll]),
                "predicted_syllables": json.dumps(dict(syllable_predictions[syll].most_common()), ensure_ascii=False),
                "passes_syllable_threshold": int(count >= min_count and count / support >= min_ratio),
            })
        total_s += by_type["substitution"]
        total_d += by_type["deletion"]
        total_i += by_type["insertion"]
        total_chars += char_total
        total_word_edits += word_edits
        total_words += word_total

    speaker_fields = list(speaker_rows[0])
    type_fields = list(type_rows[0])
    syllable_fields = list(syllable_rows[0]) if syllable_rows else [
        "speaker_id", "reference_syllable", "reference_count", "error_count", "error_ratio",
        "substitutions", "deletions", "distinct_predictions", "predicted_syllables", "passes_syllable_threshold"
    ]
    summary = {
        "analysis_split": "profile_train",
        "metric_definition": {
            "cer": "micro-aggregated Hangul-syllable edit distance / Hangul reference syllable count; spaces, punctuation, Latin and digits excluded",
            "wer": "micro-aggregated whitespace-token edit distance / reference word count; punctuation removed, Hangul/Latin/digits retained",
            "warning": "These normalization rules may differ from the general Whisper baseline; compare only after confirming identical evaluation normalization.",
            "filtered_profile_empty": "No error pair passes the original profile thresholds; not necessarily a low-CER speaker",
            "qualified_reference_syllables": "Alternative aggregation of substitution+deletion by reference syllable; exploratory only",
        },
        "expected_speakers": len(expected),
        "complete_profiles_analyzed": len(profiles),
        "remaining_speakers": len(expected) - len(profiles),
        "analyzed_utterances": sum(x["utterances"] for x in speaker_rows),
        "empty_filtered_profiles": empty_count,
        "empty_filtered_profile_fraction": ratio(empty_count, len(profiles)),
        "speakers_with_repeated_error_pairs": pair_repeated_speakers,
        "speakers_with_repeated_reference_syllable_errors": syllable_repeated_speakers,
        "micro_cer": ratio(total_s + total_d + total_i, total_chars),
        "micro_wer": ratio(total_word_edits, total_words),
        "macro_speaker_cer": round(sum(x["cer"] for x in speaker_rows if x["cer"] is not None) / sum(x["cer"] is not None for x in speaker_rows), 6) if any(x["cer"] is not None for x in speaker_rows) else None,
        "macro_speaker_wer": round(sum(x["wer"] for x in speaker_rows if x["wer"] is not None) / sum(x["wer"] is not None for x in speaker_rows), 6) if any(x["wer"] is not None for x in speaker_rows) else None,
        "error_type_events": {"substitution": total_s, "deletion": total_d, "insertion": total_i},
        "speaker_ids": sorted(profiles),
    }

    def save_all(target: Path):
        target.mkdir(parents=True, exist_ok=True)
        write_csv(target / "speaker_summary.csv", speaker_rows, speaker_fields)
        write_csv(target / "error_type_summary.csv", type_rows, type_fields)
        write_csv(target / "syllable_error_summary.csv", syllable_rows, syllable_fields)
        (target / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    save_all(output_dir)
    if args.snapshot:
        snapshot_dir = output_dir / "snapshots" / datetime.now().strftime("%Y%m%d_%H%M%S")
        if snapshot_dir.exists():
            raise FileExistsError(snapshot_dir)
        save_all(snapshot_dir)
        LOG.info("Snapshot saved: %s", snapshot_dir)
    LOG.info("Analyzed %d/%d speakers, %d utterances", len(profiles), len(expected), summary["analyzed_utterances"])
    LOG.info("Empty filtered profiles: %d/%d", empty_count, len(profiles))
    LOG.info("Micro CER: %s, micro WER: %s", summary["micro_cer"], summary["micro_wer"])
    LOG.info("Results: %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()