from datasets import load_dataset, Audio
from transformers import WhisperTokenizer
import pandas as pd
import os
import re
import time


# ============================================================
# 설정
# ============================================================

BEFORE_DATASET = "yoona-J/ASR_Peripheral_Neuropathy_Dataset"
AFTER_DATASET = "yoona-J/ASR_Preprocess_Peripheral_Neuropathy_Dataset"

SPLITS = {
    "train": {
        "before_count": 19508,
        "after_count": 18679
    },
    "valid": {
        "before_count": 1084,
        "after_count": 1040
    },
    "test": {
        "before_count": 1084,
        "after_count": 1036
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
    """
    예:
    output_PN_ABC-01-03-F-52-KK_55.wav

    화자코드 + 성별 + 나이를 추출하여
    기존과 동일한 pseudo-speaker ID 생성
    """

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


# ============================================================
# 데이터 연결
# ============================================================

print("========== Peripheral 전체 Mapping ==========\n")

print("[1/2] Hugging Face 데이터셋 연결 중...")

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


# ============================================================
# Split 처리
# ============================================================

all_summary = []


for split_name, config in SPLITS.items():

    expected_before = config["before_count"]
    expected_after = config["after_count"]

    print("\n" + "=" * 65)
    print(f"{split_name.upper()} 처리 시작")
    print("=" * 65)


    # ========================================================
    # Before metadata 저장
    # ========================================================

    before_csv = os.path.join(
        DATA_DIR,
        f"peripheral_before_{split_name}.csv"
    )

    print(f"\n[1/5] {split_name} Before 읽는 중...")

    before_rows = []
    start = time.time()

    for i, row in enumerate(before[split_name]):

        before_rows.append({
            "before_row": i,
            "filename": row["audio"]["path"],
            "transcript":
                normalize_text(row["transcripts"])
        })

        current = i + 1

        if (
            current % 1000 == 0
            or current == expected_before
        ):
            print(
                f"  Before {current:,} / "
                f"{expected_before:,} "
                f"({current / expected_before * 100:.1f}%)"
            )

        if current >= expected_before:
            break

    before_df = pd.DataFrame(before_rows)

    before_df.to_csv(
        before_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Before 저장 완료: {len(before_df):,}개 "
        f"| {time.time() - start:.1f}초"
    )


    # ========================================================
    # After metadata 저장
    # ========================================================

    after_csv = os.path.join(
        DATA_DIR,
        f"peripheral_after_{split_name}.csv"
    )

    print(f"\n[2/5] {split_name} After 읽는 중...")

    after_rows = []
    start = time.time()

    for i, row in enumerate(after[split_name]):

        decoded = tokenizer.decode(
            row["labels"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        after_rows.append({
            "processed_row": i,
            "transcript": normalize_text(decoded)
        })

        current = i + 1

        if (
            current % 1000 == 0
            or current == expected_after
        ):
            print(
                f"  After {current:,} / "
                f"{expected_after:,} "
                f"({current / expected_after * 100:.1f}%)"
            )

        if current >= expected_after:
            break

    after_df = pd.DataFrame(after_rows)

    after_df.to_csv(
        after_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"After 저장 완료: {len(after_df):,}개 "
        f"| {time.time() - start:.1f}초"
    )


    # ========================================================
    # 개수 검증
    # ========================================================

    print(f"\n[3/5] {split_name} 개수 검증 중...")

    if len(before_df) != expected_before:
        raise ValueError(
            f"{split_name} Before 개수 불일치: "
            f"{len(before_df)} != {expected_before}"
        )

    if len(after_df) != expected_after:
        raise ValueError(
            f"{split_name} After 개수 불일치: "
            f"{len(after_df)} != {expected_after}"
        )

    print("개수 정상")


    # ========================================================
    # Forward / Backward Mapping
    # ========================================================

    print(f"\n[4/5] {split_name} ambiguity 검사 중...")

    B = before_df["transcript"].tolist()
    A = after_df["transcript"].tolist()


    # ---------- Forward ----------
    forward = [-1] * len(A)

    b = 0

    for a in range(len(A)):

        while b < len(B):

            if A[a] == B[b]:
                forward[a] = b
                b += 1
                break

            b += 1


    # ---------- Backward ----------
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
    # 최종 Mapping 생성
    # ========================================================

    mapping_rows = []

    reliable = 0
    ambiguous = 0
    failed = 0
    speaker_failed = 0


    for i in range(len(A)):

        f_row = forward[i]
        b_row = backward[i]


        if f_row == -1 or b_row == -1:

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

                speaker_code = info["speaker_code"]
                sex = info["sex"]
                age = info["age"]

                pseudo_id = info[
                    "pseudo_speaker_id"
                ]


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
            "pseudo_speaker_id": pseudo_id,
            "mapping_status": status
        })


    mapping_df = pd.DataFrame(
        mapping_rows
    )


    # ========================================================
    # 결과 저장
    # ========================================================

    print(f"\n[5/5] {split_name} 결과 저장 중...")

    mapping_file = os.path.join(
        RESULT_DIR,
        f"peripheral_{split_name}_mapping.csv"
    )

    mapping_df.to_csv(
        mapping_file,
        index=False,
        encoding="utf-8-sig"
    )


    reliable_df = mapping_df[
        mapping_df["mapping_status"]
        == "RELIABLE"
    ]


    unique_speakers = (
        reliable_df["pseudo_speaker_id"]
        .dropna()
        .nunique()
    )


    print(
        f"\n========== {split_name.upper()} 결과 =========="
    )

    print(f"Before rows      : {len(B):,}")
    print(f"Processed rows   : {len(A):,}")
    print(f"Reliable         : {reliable:,}")
    print(f"Ambiguous        : {ambiguous:,}")
    print(f"Failed           : {failed:,}")
    print(f"Speaker 실패     : {speaker_failed:,}")
    print(f"Pseudo-speakers  : {unique_speakers:,}")

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


    # 문제 row 출력
    problem_df = mapping_df[
        mapping_df["mapping_status"]
        != "RELIABLE"
    ]

    if len(problem_df) > 0:

        print("\n문제 row 예시:")

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
        "filtered": len(B) - len(A),
        "reliable": reliable,
        "ambiguous": ambiguous,
        "failed": failed,
        "speaker_failed": speaker_failed,
        "pseudo_speakers": unique_speakers
    })


# ============================================================
# 전체 Summary
# ============================================================

print("\n" + "=" * 65)
print("PERIPHERAL 최종 SUMMARY")
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
    "peripheral_mapping_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nSummary 저장 위치: {summary_file}"
)

print("\n전체 처리 완료!")