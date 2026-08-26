from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence, Collection


EMPTY_TOKEN = "<EMPTY>"


@dataclass(frozen=True)
class SyllableError:
    """
    한 개의 음절 단위 오류를 표현합니다.
    """

    error_type: str
    reference_syllable: str
    predicted_syllable: str


def is_korean_syllable(
    char: str,
) -> bool:
    """
    완성형 한글 음절인지 확인합니다.

    예:
        가, 나, 한, 글
    """

    if len(char) != 1:
        return False

    return "\uAC00" <= char <= "\uD7A3"


def normalize_for_syllable_alignment(
    text: str,
) -> list[str]:
    """
    문자열에서 완성형 한글 음절만 추출합니다.

    예:
        "자동차를 탑니다."
        -> ["자", "동", "차", "를", "탑", "니", "다"]

    현재 Error Profile은 한글 음절 오류를 대상으로 하므로
    공백, 문장부호, 영문, 숫자 등은 제외합니다.
    """

    if text is None:
        return []

    return [
        char
        for char in str(text)
        if is_korean_syllable(char)
    ]


def align_syllables(
    reference: Sequence[str],
    prediction: Sequence[str],
) -> list[tuple[str, str]]:
    """
    reference와 prediction을
    Levenshtein 기반으로 정렬합니다.

    반환 예:
        [
            ("자", "잔"),
            ("동", "동"),
            ("차", "차"),
            ("를", "<EMPTY>"),
        ]

    substitution / deletion / insertion을
    모두 처리합니다.

    중요:
        OOV 음절도 alignment 단계에서는 제거하지 않습니다.

        OOV 음절을 먼저 제거하면
        앞뒤 음절의 정렬 결과가 달라질 수 있기 때문입니다.

        OOV 여부는 alignment 이후
        Error Profile 집계 단계에서 처리합니다.
    """

    n = len(reference)
    m = len(prediction)

    dp = [
        [0] * (m + 1)
        for _ in range(n + 1)
    ]

    backtrace = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        dp[i][0] = i
        backtrace[i][0] = "delete"

    for j in range(1, m + 1):
        dp[0][j] = j
        backtrace[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            ref = reference[i - 1]
            pred = prediction[j - 1]

            if ref == pred:
                substitution_cost = (
                    dp[i - 1][j - 1]
                )
            else:
                substitution_cost = (
                    dp[i - 1][j - 1] + 1
                )

            deletion_cost = (
                dp[i - 1][j] + 1
            )

            insertion_cost = (
                dp[i][j - 1] + 1
            )

            best_cost = min(
                substitution_cost,
                deletion_cost,
                insertion_cost,
            )

            dp[i][j] = best_cost

            # 동일 비용일 경우 diagonal 우선
            # → 가능한 한 음절 대 음절 대응 유지
            if best_cost == substitution_cost:

                if ref == pred:
                    backtrace[i][j] = "equal"
                else:
                    backtrace[i][j] = "substitute"

            elif best_cost == deletion_cost:
                backtrace[i][j] = "delete"

            else:
                backtrace[i][j] = "insert"

    aligned: list[
        tuple[str, str]
    ] = []

    i = n
    j = m

    while i > 0 or j > 0:

        operation = backtrace[i][j]

        if operation in {
            "equal",
            "substitute",
        }:

            aligned.append(
                (
                    reference[i - 1],
                    prediction[j - 1],
                )
            )

            i -= 1
            j -= 1

        elif operation == "delete":

            aligned.append(
                (
                    reference[i - 1],
                    EMPTY_TOKEN,
                )
            )

            i -= 1

        elif operation == "insert":

            aligned.append(
                (
                    EMPTY_TOKEN,
                    prediction[j - 1],
                )
            )

            j -= 1

        else:

            raise RuntimeError(
                "알 수 없는 alignment operation: "
                f"{operation}"
            )

    aligned.reverse()

    return aligned


def extract_syllable_errors(
    reference_text: str,
    prediction_text: str,
) -> list[SyllableError]:
    """
    한 문장의 reference / prediction에서
    음절 오류를 추출합니다.

    error_type: substitution, deletion, insertion

    여기서는 OOV 여부와 관계없이
    전체 한글 음절을 대상으로 alignment합니다.

    OOV filtering은
    build_raw_error_profile()에서 수행합니다.
    """

    reference = (
        normalize_for_syllable_alignment(
            reference_text
        )
    )

    prediction = (
        normalize_for_syllable_alignment(
            prediction_text
        )
    )

    aligned = align_syllables(
        reference,
        prediction,
    )

    errors: list[
        SyllableError
    ] = []

    for ref, pred in aligned:

        if ref == pred:
            continue

        if ref == EMPTY_TOKEN:

            error_type = "insertion"

        elif pred == EMPTY_TOKEN:

            error_type = "deletion"

        else:

            error_type = "substitution"

        errors.append(
            SyllableError(
                error_type=error_type,
                reference_syllable=ref,
                predicted_syllable=pred,
            )
        )

    return errors


def build_raw_error_profile(
    reference_prediction_pairs: Iterable[
        tuple[str, str]
    ],
    *,
    supported_reference_syllables: (
        Collection[str] | None
    ) = None,
) -> list[dict]:
    """
    여러 train 문장의 reference / prediction을 입력받아
    화자 단위 Raw Error Profile을 생성합니다.

    Parameters
    ----------
    reference_prediction_pairs:
        (reference_text, prediction_text) 쌍.

    supported_reference_syllables:
        범용 모델의 기존 vocab이 지원하는
        reference 음절 집합.

        None이면 기존 동작과 동일하게
        모든 한글 reference 음절을 사용합니다.

        값을 전달하면 substitution / deletion 중
        reference 음절이 해당 집합에 없는 오류는
        Speaker Error Profile에서 제외합니다.

    중요
    ----
    OOV reference 음절도 alignment 자체에는 포함됩니다.

    alignment를 전체 문장 기준으로 먼저 수행한 뒤,
    Error Profile 집계 단계에서만 제외합니다.

    따라서:

        범용 vocab 지원 reference
            → Speaker Error Profile

        범용 vocab 미지원 reference
            → Speaker Error Profile 제외
            → vocabulary limitation으로 별도 관리

    insertion은 reference가 <EMPTY>이므로
    기존과 동일하게 Profile에 포함합니다.

    ratio 정의
    ----------
    substitution / deletion:

        해당 reference 음절의 오류 count / train 정답 전체에서 해당 reference 음절 등장 횟수

    insertion:

        reference 음절이 없기 때문에 전체 문장 수 기준 ratio 사용
    """

    pairs = list(
        reference_prediction_pairs
    )

    supported_set: (
        set[str] | None
    )

    if supported_reference_syllables is None:

        supported_set = None

    else:

        supported_set = {
            str(syllable)
            for syllable
            in supported_reference_syllables
            if is_korean_syllable(
                str(syllable)
            )
        }

    reference_counter: Counter[
        str
    ] = Counter()

    error_counter: Counter[
        tuple[str, str, str]
    ] = Counter()

    for (
        reference_text,
        prediction_text,
    ) in pairs:

        reference_syllables = (
            normalize_for_syllable_alignment(
                reference_text
            )
        )

        # ----------------------------------------------------
        # ratio denominator
        #
        # supported_set이 전달된 경우
        # 기존 범용 vocab이 지원하는 reference 음절만
        # denominator에 포함한다.
        # ----------------------------------------------------

        if supported_set is None:

            supported_reference = (
                reference_syllables
            )

        else:

            supported_reference = [
                syllable
                for syllable
                in reference_syllables
                if syllable in supported_set
            ]

        reference_counter.update(
            supported_reference
        )

        # ----------------------------------------------------
        # Alignment / Error extraction
        #
        # 전체 reference로 먼저 alignment한다.
        # ----------------------------------------------------

        errors = extract_syllable_errors(
            reference_text,
            prediction_text,
        )

        for error in errors:

            # ------------------------------------------------
            # substitution / deletion의 reference가
            # 기존 범용 vocab에 없는 경우:
            #
            # → Speaker Error Profile에서는 제외
            # → vocabulary limitation으로 별도 관리
            # ------------------------------------------------

            if (
                supported_set is not None
                and error.error_type
                in {
                    "substitution",
                    "deletion",
                }
                and error.reference_syllable
                not in supported_set
            ):
                continue

            key = (
                error.error_type,
                error.reference_syllable,
                error.predicted_syllable,
            )

            error_counter[
                key
            ] += 1

    num_sentences = len(
        pairs
    )

    rows: list[
        dict
    ] = []

    for (
        error_type,
        reference_syllable,
        predicted_syllable,
    ), count in error_counter.items():

        if error_type == "insertion":

            denominator = (
                num_sentences
            )

        else:

            denominator = (
                reference_counter[
                    reference_syllable
                ]
            )

        ratio = (
            count / denominator
            if denominator > 0
            else 0.0
        )

        rows.append(
            {
                "error_type": (
                    error_type
                ),
                "reference_syllable": (
                    reference_syllable
                ),
                "predicted_syllable": (
                    predicted_syllable
                ),
                "count": (
                    count
                ),
                "reference_count": (
                    denominator
                ),
                "ratio": (
                    ratio
                ),
            }
        )

    rows.sort(
        key=lambda x: (
            -x["count"],
            -x["ratio"],
            x["error_type"],
            x["reference_syllable"],
            x["predicted_syllable"],
        )
    )

    return rows


def analyze_vocab_unsupported_errors(
    reference_prediction_pairs: Iterable[
        tuple[str, str]
    ],
    *,
    supported_reference_syllables: Collection[str],
) -> tuple[list[dict], dict]:
    """
    전체 alignment 오류 중에서
    범용 vocab이 지원하지 않는 reference 음절 때문에
    Speaker Error Profile에서 제외되는 오류를 분석합니다.

    반환
    ----
    unsupported_rows:
        vocab 미지원 오류 유형별 상세 통계

    summary:
        전체 alignment 오류 수,
        vocab 미지원 오류 수,
        vocab 미지원 오류 비율 등을 포함한 요약
    """

    pairs = list(
        reference_prediction_pairs
    )

    supported_set = {
        str(syllable)
        for syllable
        in supported_reference_syllables
        if is_korean_syllable(
            str(syllable)
        )
    }

    total_alignment_errors = 0
    vocab_unsupported_errors = 0

    unsupported_counter: Counter[
        tuple[str, str, str]
    ] = Counter()

    for (
        reference_text,
        prediction_text,
    ) in pairs:

        errors = extract_syllable_errors(
            reference_text,
            prediction_text,
        )

        total_alignment_errors += len(
            errors
        )

        for error in errors:

            # insertion은 reference가 없으므로
            # vocab 지원 여부를 판단하지 않는다.
            if error.error_type == "insertion":
                continue

            if (
                error.reference_syllable
                not in supported_set
            ):

                vocab_unsupported_errors += 1

                key = (
                    error.error_type,
                    error.reference_syllable,
                    error.predicted_syllable,
                )

                unsupported_counter[
                    key
                ] += 1

    unsupported_rows: list[
        dict
    ] = []

    for (
        error_type,
        reference_syllable,
        predicted_syllable,
    ), count in unsupported_counter.items():

        unsupported_rows.append(
            {
                "error_type": (
                    error_type
                ),
                "reference_syllable": (
                    reference_syllable
                ),
                "predicted_syllable": (
                    predicted_syllable
                ),
                "count": (
                    count
                ),
            }
        )

    unsupported_rows.sort(
        key=lambda row: (
            -row["count"],
            row["error_type"],
            row["reference_syllable"],
            row["predicted_syllable"],
        )
    )

    analyzable_errors = (
        total_alignment_errors
        - vocab_unsupported_errors
    )

    unsupported_rate = (
        vocab_unsupported_errors
        / total_alignment_errors
        if total_alignment_errors > 0
        else 0.0
    )

    summary = {
        "total_alignment_errors": (
            total_alignment_errors
        ),
        "speaker_analyzable_errors": (
            analyzable_errors
        ),
        "vocab_unsupported_errors": (
            vocab_unsupported_errors
        ),
        "vocab_unsupported_error_rate": (
            unsupported_rate
        ),
    }

    return (
        unsupported_rows,
        summary,
    )


def filter_error_profile(
    raw_profile: Sequence[
        dict
    ],
    *,
    min_count: int = 1,
    min_ratio: float = 0.0,
) -> list[dict]:
    """
    Raw Error Profile에서
    threshold 조건을 만족하는 오류만 선택합니다.

    Threshold 규칙
    ----------------
    substitution / deletion:
        count >= min_count
        AND
        ratio >= min_ratio

    insertion:
        count >= min_count

    insertion은 대응되는 reference 음절이 없어서
    substitution / deletion과 동일한 의미의
    ratio를 정의할 수 없으므로
    min_ratio 조건을 적용하지 않습니다.
    """

    if min_count < 1:

        raise ValueError(
            "min_count는 1 이상이어야 합니다."
        )

    if not (
        0.0
        <= min_ratio
        <= 1.0
    ):

        raise ValueError(
            "min_ratio는 0~1 사이여야 합니다."
        )

    filtered_profile = []

    for row in raw_profile:

        # 모든 오류 유형에 공통 적용
        if row["count"] < min_count:
            continue

        # insertion은 count만 적용
        if row["error_type"] == "insertion":
            filtered_profile.append(row)
            continue

        # substitution / deletion은
        # count + ratio 모두 적용
        if row["ratio"] >= min_ratio:
            filtered_profile.append(row)

    return filtered_profile