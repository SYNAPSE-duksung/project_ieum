"""Build Error Profile v2 from cached Whisper predictions.

Usage:
    python -m src.personalization.profile_builder_v2 \
        --predictions PATH --flags PATH --manifest PATH --output-dir PATH

No Whisper inference. Original inputs are never modified. Strong mismatch
flags are screening heuristics, not audio-verified ground truth.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .error_profile import (
    align_syllables,
    build_raw_error_profile,
    normalize_for_syllable_alignment,
)
from .profile_stability import stability


SCENARIOS = ("A_original", "B_exclude_strong")
EMPTY_TOKEN = "<EMPTY>"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def utterance_key(row: dict) -> tuple[str, str, str]:
    return (
        row["speaker_id"].strip(),
        row["batch"].strip(),
        row["audio_member"].strip().replace("\\", "/"),
    )


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def ensure_unique_keys(rows: list[dict], label: str) -> None:
    keys = [utterance_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate {label} keys")


def calculate_statistics(rows: list[dict]) -> tuple[dict, dict, dict]:
    """Aggregate aligned syllable events, including correct occurrences."""
    syllables = defaultdict(
        lambda: {
            "total_count": 0,
            "correct_count": 0,
            "substitution_count": 0,
            "deletion_count": 0,
            "confusions": Counter(),
        }
    )
    inserted_syllables = Counter()
    substitutions = deletions = insertions = reference_total = 0

    for row in rows:
        reference = normalize_for_syllable_alignment(row["reference"])
        prediction = normalize_for_syllable_alignment(row["prediction"])
        reference_total += len(reference)

        for ref_syllable, pred_syllable in align_syllables(reference, prediction):
            if ref_syllable == EMPTY_TOKEN:
                inserted_syllables[pred_syllable] += 1
                insertions += 1
                continue

            item = syllables[ref_syllable]
            item["total_count"] += 1
            if ref_syllable == pred_syllable:
                item["correct_count"] += 1
            elif pred_syllable == EMPTY_TOKEN:
                item["deletion_count"] += 1
                deletions += 1
            else:
                item["substitution_count"] += 1
                item["confusions"][pred_syllable] += 1
                substitutions += 1

    syllable_profiles = {}
    for syllable, item in sorted(syllables.items()):
        count = item["total_count"]
        error_count = item["substitution_count"] + item["deletion_count"]
        assert item["correct_count"] + error_count == count

        syllable_profiles[syllable] = {
            "total_count": count,
            "correct_count": item["correct_count"],
            "error_count": error_count,
            "error_rate": error_count / count,
            "substitution": {
                "count": item["substitution_count"],
                "rate": item["substitution_count"] / count,
                "confusions": dict(sorted(item["confusions"].items())),
            },
            "deletion": {
                "count": item["deletion_count"],
                "rate": item["deletion_count"] / count,
            },
        }

    assert sum(item["total_count"] for item in syllable_profiles.values()) == reference_total
    overall = {
        "reference_syllable_count": reference_total,
        "substitution_count": substitutions,
        "deletion_count": deletions,
        "insertion_count": insertions,
        "cer": (
            (substitutions + deletions + insertions) / reference_total
            if reference_total
            else None
        ),
    }
    insertion_profile = {
        "total_count": insertions,
        "syllables": {
            syllable: {"count": count, "ratio": count / insertions}
            for syllable, count in sorted(inserted_syllables.items())
        },
    }
    return overall, syllable_profiles, insertion_profile


def prepare_speakers(predictions: list[dict], flags: list[dict], manifest: list[dict]):
    """Validate joins and attach recording IDs and mismatch flags."""
    for label, rows in (
        ("prediction", predictions),
        ("flag", flags),
        ("manifest", manifest),
    ):
        ensure_unique_keys(rows, label)

    flag_map = {utterance_key(row): row for row in flags}
    manifest_map = {utterance_key(row): row for row in manifest}
    prediction_keys = {utterance_key(row) for row in predictions}
    if not set(flag_map) <= prediction_keys:
        raise ValueError("Flags not in predictions")
    if not prediction_keys <= set(manifest_map):
        raise ValueError("Prediction keys missing in manifest")

    speakers = defaultdict(list)
    audit = []
    for prediction in predictions:
        key = utterance_key(prediction)
        manifest_row = manifest_map[key]
        if manifest_row["personal_split"].strip() != "profile_train":
            raise ValueError(f"Leakage: {key}")
        if prediction["reference"] != manifest_row["text"].strip():
            raise ValueError(f"Reference mismatch: {key}")

        flags_row = flag_map.get(key, {})
        strong = is_true(flags_row.get("strong_other_match", False))
        severe = is_true(flags_row.get("severe_mismatch", False))
        enriched = {
            **prediction,
            "_recording_id": manifest_row["recording_id"],
            "_strong": strong,
            "_severe": severe,
        }
        speakers[key[0]].append(enriched)

        if strong or severe:
            audit.append({
                **prediction,
                "strong_other_match": int(strong),
                "severe_mismatch": int(severe),
                "excluded_in_B": int(strong),
                "verified_audio": "NOT_VERIFIED",
            })

    return speakers, audit


def build_profiles(speakers: dict, output_dir: Path, seed: int):
    """Generate A/B profiles, summaries, and split-half evaluations."""
    speaker_summaries = []
    syllable_summaries = []
    validations = []

    for scenario in SCENARIOS:
        profiles_dir = output_dir / scenario / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        for speaker_id, original_rows in sorted(speakers.items()):
            rows = [
                row for row in original_rows
                if scenario == "A_original" or not row["_strong"]
            ]
            overall, syllables, insertions = calculate_statistics(rows)
            pairs = [(row["reference"], row["prediction"]) for row in rows]
            raw_profile = build_raw_error_profile(
                pairs, supported_reference_syllables=None
            )
            total_errors = sum(
                overall[field]
                for field in ("substitution_count", "deletion_count", "insertion_count")
            )
            assert sum(item["count"] for item in raw_profile) == total_errors

            excluded_count = len(original_rows) - len(rows)
            profile = {
                "profile_version": "v2",
                "speaker_id": speaker_id,
                "split": "profile_train",
                "scenario": scenario,
                "utterance_count": len(rows),
                "data_quality": {
                    "original_utterances": len(original_rows),
                    "excluded_suspected_strong": excluded_count,
                    "audio_verified": False,
                },
                "overall_statistics": overall,
                "raw_profile": raw_profile,
                "syllable_profiles": syllables,
                "insertion_profile": insertions,
            }
            profile_path = profiles_dir / f"{speaker_id}.json"
            profile_path.write_text(
                json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            speaker_summaries.append({
                "scenario": scenario,
                "speaker_id": speaker_id,
                "utterances": len(rows),
                "excluded": excluded_count,
                **overall,
                "raw_profile_items": len(raw_profile),
            })
            for syllable, item in syllables.items():
                syllable_summaries.append({
                    "scenario": scenario,
                    "speaker_id": speaker_id,
                    "syllable": syllable,
                    "total_count": item["total_count"],
                    "error_count": item["error_count"],
                    "error_rate": item["error_rate"],
                    "substitution_count": item["substitution"]["count"],
                    "deletion_count": item["deletion"]["count"],
                })
            validations.append(
                stability(rows, seed, scenario, speaker_id, calculate_statistics)
            )

    return speaker_summaries, syllable_summaries, validations


def summarize(speaker_summaries: list[dict], validations: list[dict], seed: int) -> dict:
    """Compute micro-CER and per-speaker split-half summary statistics."""
    scenario_summaries = []
    for scenario in SCENARIOS:
        speakers = [row for row in speaker_summaries if row["scenario"] == scenario]
        reference_total = sum(row["reference_syllable_count"] for row in speakers)
        edits_total = sum(
            row["substitution_count"] + row["deletion_count"] + row["insertion_count"]
            for row in speakers
        )
        checks = [row for row in validations if row["scenario"] == scenario]
        valid = [row for row in checks if row["status"] == "ok"]
        differences = [
            row["weighted_abs_error_rate_difference"] for row in valid
            if row["weighted_abs_error_rate_difference"] is not None
        ]
        jaccards = [
            row["repeated_pair_jaccard"] for row in valid
            if row["repeated_pair_jaccard"] is not None
        ]
        scenario_summaries.append({
            "scenario": scenario,
            "speakers": len(speakers),
            "utterances": sum(row["utterances"] for row in speakers),
            "excluded": sum(row["excluded"] for row in speakers),
            "micro_cer": edits_total / reference_total if reference_total else None,
            "split_half_evaluable": len(valid),
            "mean_weighted_abs_error_rate_difference": (
                sum(differences) / len(differences) if differences else None
            ),
            "mean_repeated_pair_jaccard": (
                sum(jaccards) / len(jaccards) if jaccards else None
            ),
            "pair_jaccard_defined_speakers": len(jaccards),
        })

    return {
        "seed": seed,
        "scenarios": scenario_summaries,
        "notes": [
            "Strong mismatch flags are heuristic, not audio-verified ground truth.",
            "Filtering based on predictions can preferentially remove hard samples.",
            "Split-half uses manifest recording_id groups; no group crosses halves.",
            "Pair Jaccard is undefined when both halves have no repeated pairs.",
            "Error-rate difference uses syllables with >=3 occurrences in each half.",
            "Only profile_train rows are used; no personal_val/test leakage.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "flags", "manifest", "output-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = read_csv(args.predictions)
    flags = read_csv(args.flags)
    manifest = read_csv(args.manifest)
    speakers, audit = prepare_speakers(predictions, flags, manifest)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "suspected_mismatch_audit.csv",
        audit,
        [
            "speaker_id", "batch", "audio_member", "reference", "prediction",
            "strong_other_match", "severe_mismatch", "excluded_in_B", "verified_audio",
        ],
    )
    speaker_summaries, syllable_summaries, validations = build_profiles(
        speakers, output_dir, args.seed
    )
    write_csv(
        output_dir / "speaker_statistics.csv",
        speaker_summaries,
        list(speaker_summaries[0]),
    )
    write_csv(
        output_dir / "syllable_statistics.csv",
        syllable_summaries,
        list(syllable_summaries[0]),
    )
    validation_fields = list(dict.fromkeys(
        field for row in validations for field in row
    ))
    write_csv(
        output_dir / "split_half_results.csv",
        validations,
        validation_fields,
    )
    report = summarize(speaker_summaries, validations, args.seed)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
