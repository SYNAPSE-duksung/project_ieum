from __future__ import annotations

from typing import Sequence

from src.personalization.error_profile import (
    extract_syllable_errors,
)


def make_profile_key(
    error_type: str,
    reference_syllable: str,
    predicted_syllable: str,
) -> tuple[str, str, str]:
    """
    Error Profile의 오류 패턴을 비교하기 위한 key를 생성합니다.

    예:
        ("substitution", "가", "카")
    """
    return (
        str(error_type),
        str(reference_syllable),
        str(predicted_syllable),
    )


def build_profile_ratio_map(
    profile: Sequence[dict],
) -> dict[tuple[str, str, str], float]:
    """
    Error Profile을 빠르게 검색할 수 있도록

        오류 패턴 -> ratio

    형태의 dictionary로 변환합니다.

    예:
        ("substitution", "가", "카") -> 0.8
    """
    ratio_map: dict[
        tuple[str, str, str],
        float,
    ] = {}

    required_columns = {
        "error_type",
        "reference_syllable",
        "predicted_syllable",
        "ratio",
    }

    for row in profile:
        missing = required_columns - set(row.keys())

        if missing:
            raise ValueError(
                "Error Profile에 필요한 컬럼이 없습니다: "
                f"{sorted(missing)}"
            )

        ratio = float(row["ratio"])

        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"ratio는 0~1 사이여야 합니다. 현재 값: {ratio}"
            )

        key = make_profile_key(
            row["error_type"],
            row["reference_syllable"],
            row["predicted_syllable"],
        )

        ratio_map[key] = ratio

    return ratio_map


def get_matched_profile_errors(
    reference_text: str,
    prediction_text: str,
    profile: Sequence[dict],
) -> list[dict]:
    """
    한 문장에서 발생한 음절 오류 중
    Error Profile에 포함된 오류를 찾습니다.

    같은 Profile 오류가 한 문장에서 여러 번 발생하면
    발생 횟수만큼 각각 포함합니다.
    """
    ratio_map = build_profile_ratio_map(profile)

    errors = extract_syllable_errors(
        reference_text,
        prediction_text,
    )

    matched_errors: list[dict] = []

    for error in errors:
        key = make_profile_key(
            error.error_type,
            error.reference_syllable,
            error.predicted_syllable,
        )

        if key not in ratio_map:
            continue

        matched_errors.append(
            {
                "error_type": error.error_type,
                "reference_syllable": error.reference_syllable,
                "predicted_syllable": error.predicted_syllable,
                "ratio": ratio_map[key],
            }
        )

    return matched_errors


def calculate_sample_weight(
    reference_text: str,
    prediction_text: str,
    profile: Sequence[dict],
    *,
    alpha: float = 0.5,
    base_weight: float = 1.0,
    max_weight: float | None = None,
) -> dict:
    """
    한 문장의 Error Profile 기반 sample weight를 계산합니다.

    최종 정의:

        sample_weight
        = base_weight
        + alpha * sum(profile error ratios)

    같은 오류가 한 문장에서 여러 번 발생하면
    해당 오류의 ratio도 발생 횟수만큼 합산됩니다.

    alpha와 max_weight의 최종값은
    Validation 실험을 통해 결정합니다.
    """
    if alpha < 0:
        raise ValueError(
            "alpha는 0 이상이어야 합니다."
        )

    if base_weight <= 0:
        raise ValueError(
            "base_weight는 0보다 커야 합니다."
        )

    if (
        max_weight is not None
        and max_weight < base_weight
    ):
        raise ValueError(
            "max_weight는 base_weight 이상이어야 합니다."
        )

    matched_errors = get_matched_profile_errors(
        reference_text,
        prediction_text,
        profile,
    )

    ratio_sum = sum(
        error["ratio"]
        for error in matched_errors
    )

    sample_weight = (
        base_weight
        + alpha * ratio_sum
    )

    if max_weight is not None:
        sample_weight = min(
            sample_weight,
            max_weight,
        )

    return {
        "num_profile_errors": len(matched_errors),
        "profile_ratio_sum": float(ratio_sum),
        "sample_weight": float(sample_weight),
        "matched_errors": matched_errors,
    }


def calculate_sample_weights(
    reference_prediction_pairs: Sequence[
        tuple[str, str]
    ],
    profile: Sequence[dict],
    *,
    alpha: float = 0.5,
    base_weight: float = 1.0,
    max_weight: float | None = None,
) -> list[dict]:
    """
    여러 학습 문장에 대해 Error Profile 기반
    sample weight를 계산합니다.
    """
    results: list[dict] = []

    for reference_text, prediction_text in reference_prediction_pairs:
        weight_result = calculate_sample_weight(
            reference_text,
            prediction_text,
            profile,
            alpha=alpha,
            base_weight=base_weight,
            max_weight=max_weight,
        )

        results.append(
            {
                "reference_text": reference_text,
                "prediction_text": prediction_text,
                "num_profile_errors": weight_result[
                    "num_profile_errors"
                ],
                "profile_ratio_sum": weight_result[
                    "profile_ratio_sum"
                ],
                "sample_weight": weight_result[
                    "sample_weight"
                ],
                "matched_errors": weight_result[
                    "matched_errors"
                ],
            }
        )

    return results