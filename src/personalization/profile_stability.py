"""Evaluate error-profile repeatability across recording-group split halves."""

import random
from collections import defaultdict
from typing import Callable

from .error_profile import build_raw_error_profile


def half_split(rows: list[dict], seed: int):
    """Assign whole recording groups to two approximately balanced halves."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["_recording_id"]].append(row)

    if len(groups) < 2:
        return None

    recording_ids = sorted(groups)
    random.Random(seed).shuffle(recording_ids)
    halves = [[], []]
    sizes = [0, 0]

    for recording_id in recording_ids:
        half_index = 0 if sizes[0] <= sizes[1] else 1
        halves[half_index].extend(groups[recording_id])
        sizes[half_index] += len(groups[recording_id])

    return halves


def repeated_error_pairs(rows: list[dict], min_count: int = 2) -> set[tuple]:
    """Collect exact confusion pairs observed at least min_count times."""
    pairs = [(row["reference"], row["prediction"]) for row in rows]
    raw_profile = build_raw_error_profile(pairs)
    return {
        (item["error_type"], item["reference_syllable"], item["predicted_syllable"])
        for item in raw_profile
        if item["count"] >= min_count
    }


def stability(
    rows: list[dict],
    seed: int,
    scenario: str,
    speaker_id: str,
    stats: Callable,
) -> dict:
    """Compare supported syllable error rates and recurring confusion pairs."""
    halves = half_split(rows, seed)
    if halves is None:
        return {
            "scenario": scenario,
            "speaker_id": speaker_id,
            "status": "insufficient_recording_groups",
        }

    half_a, half_b = halves
    overall_a, syllables_a, _ = stats(half_a)
    overall_b, syllables_b, _ = stats(half_b)

    common = set(syllables_a) & set(syllables_b)
    comparable = [
        syllable
        for syllable in common
        if syllables_a[syllable]["total_count"] >= 3
        and syllables_b[syllable]["total_count"] >= 3
    ]

    weighted_difference = 0.0
    total_weight = 0
    for syllable in comparable:
        count_a = syllables_a[syllable]["total_count"]
        count_b = syllables_b[syllable]["total_count"]
        weight = min(count_a, count_b)
        rate_a = syllables_a[syllable]["error_rate"]
        rate_b = syllables_b[syllable]["error_rate"]
        weighted_difference += weight * abs(rate_a - rate_b)
        total_weight += weight

    pairs_a = repeated_error_pairs(half_a)
    pairs_b = repeated_error_pairs(half_b)
    pair_union = pairs_a | pairs_b

    return {
        "scenario": scenario,
        "speaker_id": speaker_id,
        "status": "ok",
        "half_a_utterances": len(half_a),
        "half_b_utterances": len(half_b),
        "half_a_recordings": len({r["_recording_id"] for r in half_a}),
        "half_b_recordings": len({r["_recording_id"] for r in half_b}),
        "comparable_syllables": len(comparable),
        "weighted_abs_error_rate_difference": (
            weighted_difference / total_weight if total_weight else None
        ),
        "repeated_pair_jaccard": (
            len(pairs_a & pairs_b) / len(pair_union) if pair_union else None
        ),
        "repeated_pairs_a": len(pairs_a),
        "repeated_pairs_b": len(pairs_b),
        "half_a_cer": overall_a["cer"],
        "half_b_cer": overall_b["cer"],
    }
