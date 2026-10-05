from datasets import load_dataset, Audio
from transformers import WhisperTokenizer
import pandas as pd
import os
import re
import time


# ============================================================
# 설정
# ============================================================

BEFORE_DATASET = "yoona-J/ASR_Disease_General_Dataset"
AFTER_DATASET = "yoona-J/ASR_Preprocess_Disease_General_Dataset"

SPLITS = {
    "train": {
        "before_count": 38911,
        "after_count": 36170
    },
    "valid": {
        "before_count": 2162,
        "after_count": 2010
    },
    "test": {
        "before_count": 2162,
        "after_count": 2025
    }
}

DATA_DIR = "data"
RESULT_DIR = "results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 함수
# ============================================================

def normalize_text(text):
    return " ".join(str(text).strip().split())


def extract_speaker_info(filename):

    pattern = r"output_PN_([^-]+).*?-(F|M)-(\d+)-"

    match = re.search(
        pattern,
        str(filename),
        flags=re.IGNORECASE
    )

    if match is None:
        return None

    speaker_code = match.group(1).upper()
    sex = match.group(2).upper()
    age = int(match.group(3))

    return {
        "speaker_code": speaker_code,
        "sex": sex,
        "age": age,
        "pseudo_speaker_id":
            f"PN_{speaker_code}-{sex}-{age}"
    }


def valid_csv(path, expected_count):
    """
    CSV가 존재하고 행 수도 정확한 경우에만 재사용.
    """
    if not os.path.exists(path):
        return False

    try:
        df = pd.read_csv(path)

        if len(df) == expected_count:
            return True

        print(
            f"  기존 CSV 행 수 불일치: "
            f"{len(df):,} != {expected_count:,}"
        )

        return False

    except Exception:
        return False


# ============================================================
# 시작
# ============================================================

print("========== General 전체 Mapping ==========\n")

print("[준비] Hugging Face 데이터셋 연결 중...")

before = load_dataset(
    BEFORE_DATASET,
    streaming=True
)

before = before.cast_column(
    "audio",
    Audio(decode=False)
)

after = load_dataset(
    AFTER_DATASET,
    streaming=True
)

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Korean",
    task="transcribe",
    clean_up_tokenization_spaces=False
)

print("데이터셋 연결 완료\n")


all_summary = []


# ============================================================
# split별 처리
# ============================================================

for split_name, config in SPLITS.items():

    expected_before = config["before_count"]
    expected_after = config["after_count"]

    print("\n" + "=" * 65)
    print(f"{split_name.upper()} 처리 시작")
    print("=" * 65)


    before_csv = os.path.join(
        DATA_DIR,
        f"general_before_{split_name}.csv"
    )

    after_csv = os.path.join(
        DATA_DIR,
        f"general_after_{split_name}.csv"
    )


    # ========================================================
    # 1. Before metadata
    # ========================================================

    print(
        f"\n[1/5] {split_name} Before metadata 확인..."
    )

    if valid_csv(
        before_csv,
        expected_before
    ):

        print(
            f"  ✓ 기존 CSV 발견 → 재사용"
        )

        before_df = pd.read_csv(
            before_csv
        )

        print(
            f"  {len(before_df):,}개 로드 완료"
        )

    else:

        print(
            "  기존 완성 CSV 없음 → "
            "Hugging Face에서 읽기 시작"
        )

        before_rows = []

        start = time.time()

        for i, row in enumerate(
            before[split_name]
        ):

            before_rows.append({
                "before_row": i,
                "filename":
                    row["audio"]["path"],
                "transcript":
                    normalize_text(
                        row["transcripts"]
                    )
            })

            current = i + 1

            if (
                current % 1000 == 0
                or current == expected_before
            ):

                elapsed = (
                    time.time() - start
                )

                print(
                    f"  Before "
                    f"{current:,} / "
                    f"{expected_before:,} "
                    f"({current / expected_before * 100:.1f}%) "
                    f"| {elapsed:.1f}초"
                )

            if current >= expected_before:
                break


        before_df = pd.DataFrame(
            before_rows
        )

        if len(before_df) != expected_before:

            raise ValueError(
                f"{split_name} Before 개수 불일치: "
                f"{len(before_df):,} != "
                f"{expected_before:,}"
            )

        before_df.to_csv(
            before_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"  ✓ Before CSV 저장 완료: "
            f"{before_csv}"
        )


    # ========================================================
    # 2. After metadata
    # ========================================================

    print(
        f"\n[2/5] {split_name} After metadata 확인..."
    )

    if valid_csv(
        after_csv,
        expected_after
    ):

        print(
            "  ✓ 기존 CSV 발견 → 재사용"
        )

        after_df = pd.read_csv(
            after_csv
        )

        print(
            f"  {len(after_df):,}개 로드 완료"
        )

    else:

        print(
            "  기존 완성 CSV 없음 → "
            "Hugging Face에서 읽기 시작"
        )

        after_rows = []

        start = time.time()

        for i, row in enumerate(
            after[split_name]
        ):

            decoded = tokenizer.decode(
                row["labels"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            after_rows.append({
                "processed_row": i,
                "transcript":
                    normalize_text(decoded)
            })

            current = i + 1

            if (
                current % 1000 == 0
                or current == expected_after
            ):

                elapsed = (
                    time.time() - start
                )

                print(
                    f"  After "
                    f"{current:,} / "
                    f"{expected_after:,} "
                    f"({current / expected_after * 100:.1f}%) "
                    f"| {elapsed:.1f}초"
                )

            if current >= expected_after:
                break


        after_df = pd.DataFrame(
            after_rows
        )

        if len(after_df) != expected_after:

            raise ValueError(
                f"{split_name} After 개수 불일치: "
                f"{len(after_df):,} != "
                f"{expected_after:,}"
            )

        after_df.to_csv(
            after_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"  ✓ After CSV 저장 완료: "
            f"{after_csv}"
        )


    # ========================================================
    # 3. 개수 확인
    # ========================================================

    print(
        f"\n[3/5] {split_name} 개수 검증..."
    )

    print(
        f"  Before : {len(before_df):,}"
    )

    print(
        f"  After  : {len(after_df):,}"
    )

    print(
        f"  Filtered: "
        f"{len(before_df) - len(after_df):,}"
    )


    # ========================================================
    # 4. Forward / Backward Mapping
    # ========================================================

    print(
        f"\n[4/5] {split_name} "
        "Forward / Backward mapping..."
    )

    B = (
        before_df["transcript"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    A = (
        after_df["transcript"]
        .fillna("")
        .astype(str)
        .tolist()
    )


    # Forward
    forward = [-1] * len(A)

    b = 0

    for a in range(len(A)):

        while b < len(B):

            if A[a] == B[b]:

                forward[a] = b
                b += 1
                break

            b += 1


    # Backward
    backward = [-1] * len(A)

    b = len(B) - 1

    for a in range(
        len(A) - 1,
        -1,
        -1
    ):

        while b >= 0:

            if A[a] == B[b]:

                backward[a] = b
                b -= 1
                break

            b -= 1


    # ========================================================
    # Mapping 결과 생성
    # ========================================================

    mapping_rows = []

    reliable = 0
    ambiguous = 0
    failed = 0
    speaker_failed = 0


    for i in range(len(A)):

        f_row = forward[i]
        b_row = backward[i]


        if (
            f_row == -1
            or b_row == -1
        ):

            status = "FAILED"

            failed += 1

            before_row = None
            filename = None
            speaker_code = None
            sex = None
            age = None
            pseudo_id = None


        elif f_row != b_row:

            status = "AMBIGUOUS"

            ambiguous += 1

            before_row = None
            filename = None
            speaker_code = None
            sex = None
            age = None
            pseudo_id = None


        else:

            status = "RELIABLE"

            reliable += 1

            before_row = f_row

            filename = before_df.iloc[
                before_row
            ]["filename"]

            info = extract_speaker_info(
                filename
            )

            if info is None:

                speaker_failed += 1

                speaker_code = None
                sex = None
                age = None
                pseudo_id = None

            else:

                speaker_code = (
                    info["speaker_code"]
                )

                sex = info["sex"]
                age = info["age"]

                pseudo_id = (
                    info[
                        "pseudo_speaker_id"
                    ]
                )


        mapping_rows.append({
            "processed_row": i,
            "before_row": before_row,
            "forward_before_row": f_row,
            "backward_before_row": b_row,
            "filename": filename,
            "transcript": A[i],
            "speaker_code": speaker_code,
            "sex": sex,
            "age": age,
            "pseudo_speaker_id":
                pseudo_id,
            "mapping_status": status
        })


    mapping_df = pd.DataFrame(
        mapping_rows
    )


    # ========================================================
    # 5. 저장
    # ========================================================

    print(
        f"\n[5/5] {split_name} 결과 저장..."
    )

    mapping_file = os.path.join(
        RESULT_DIR,
        f"general_{split_name}_mapping.csv"
    )

    mapping_df.to_csv(
        mapping_file,
        index=False,
        encoding="utf-8-sig"
    )


    reliable_df = mapping_df[
        mapping_df[
            "mapping_status"
        ] == "RELIABLE"
    ]


    unique_speakers = (
        reliable_df[
            "pseudo_speaker_id"
        ]
        .dropna()
        .nunique()
    )


    print(
        f"\n========== "
        f"{split_name.upper()} 결과 "
        f"=========="
    )

    print(
        f"Before rows      : {len(B):,}"
    )

    print(
        f"Processed rows   : {len(A):,}"
    )

    print(
        f"Reliable         : {reliable:,}"
    )

    print(
        f"Ambiguous        : {ambiguous:,}"
    )

    print(
        f"Failed           : {failed:,}"
    )

    print(
        f"Speaker 실패     : {speaker_failed:,}"
    )

    print(
        f"Pseudo-speakers  : "
        f"{unique_speakers:,}"
    )

    print(
        f"Reliable rate    : "
        f"{reliable / len(A) * 100:.2f}%"
    )

    print(
        f"Filename 중복    : "
        f"{reliable_df['filename'].duplicated().sum():,}"
    )

    print(
        f"저장 위치        : {mapping_file}"
    )


    # 문제 row
    problem_df = mapping_df[
        mapping_df[
            "mapping_status"
        ] != "RELIABLE"
    ]

    if len(problem_df) > 0:

        print(
            "\n========== 문제 row 예시 =========="
        )

        print(
            problem_df[
                [
                    "processed_row",
                    "transcript",
                    "forward_before_row",
                    "backward_before_row",
                    "mapping_status"
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


    all_summary.append({
        "split": split_name,
        "before": len(B),
        "after": len(A),
        "filtered":
            len(B) - len(A),
        "reliable": reliable,
        "ambiguous": ambiguous,
        "failed": failed,
        "speaker_failed":
            speaker_failed,
        "pseudo_speakers":
            unique_speakers
    })


# ============================================================
# 전체 Summary
# ============================================================

print("\n" + "=" * 65)
print("GENERAL 최종 SUMMARY")
print("=" * 65)

summary_df = pd.DataFrame(
    all_summary
)

print(
    summary_df.to_string(
        index=False
    )
)

summary_file = os.path.join(
    RESULT_DIR,
    "general_mapping_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nSummary 저장 위치: "
    f"{summary_file}"
)

print("\n전체 처리 완료!")