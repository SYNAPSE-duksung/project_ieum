from datasets import load_dataset, Audio
from transformers import WhisperTokenizer
import pandas as pd
import os
import re
import time


# ============================================================
# 설정
# ============================================================

SPLITS = {
    "valid": {
        "before_count": 843,
        "after_count": 776
    },
    "test": {
        "before_count": 843,
        "after_count": 777
    }
}

DATA_DIR = "data"
RESULT_DIR = "results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


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


print("========== Stroke Valid/Test Mapping ==========\n")


# ============================================================
# 1. 데이터셋 연결
# ============================================================

print("[1/5] Hugging Face 데이터셋 연결 중...")

before = load_dataset(
    "yoona-J/ASR_Stroke_Dataset",
    streaming=True
)

before = before.cast_column(
    "audio",
    Audio(decode=False)
)

after = load_dataset(
    "yoona-J/ASR_Preprocess_Stroke_Dataset",
    streaming=True
)

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Korean",
    task="transcribe",
    clean_up_tokenization_spaces=False
)

print("연결 완료\n")


# ============================================================
# split별 처리
# ============================================================

summary = []


for split_name, config in SPLITS.items():

    before_count = config["before_count"]
    after_count = config["after_count"]

    print("\n")
    print("=" * 60)
    print(f"{split_name.upper()} 처리 시작")
    print("=" * 60)


    # ========================================================
    # 2. Before metadata
    # ========================================================

    print(
        f"\n[2/5] {split_name} Before metadata 읽는 중..."
    )

    before_rows = []

    start = time.time()

    for i, row in enumerate(before[split_name]):

        before_rows.append({
            "before_row": i,
            "filename": row["audio"]["path"],
            "transcript": normalize_text(
                row["transcripts"]
            )
        })

        current = i + 1

        if (
            current % 100 == 0
            or current == before_count
        ):
            print(
                f"  Before {current:,} / "
                f"{before_count:,} "
                f"({current / before_count * 100:.1f}%)"
            )

        if current >= before_count:
            break


    before_df = pd.DataFrame(before_rows)

    before_csv = os.path.join(
        DATA_DIR,
        f"stroke_before_{split_name}.csv"
    )

    before_df.to_csv(
        before_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Before 완료: {len(before_df):,}개 "
        f"| {time.time() - start:.1f}초"
    )


    # ========================================================
    # 3. After metadata
    # ========================================================

    print(
        f"\n[3/5] {split_name} After metadata 읽는 중..."
    )

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
            current % 100 == 0
            or current == after_count
        ):
            print(
                f"  After {current:,} / "
                f"{after_count:,} "
                f"({current / after_count * 100:.1f}%)"
            )

        if current >= after_count:
            break


    after_df = pd.DataFrame(after_rows)

    after_csv = os.path.join(
        DATA_DIR,
        f"stroke_after_{split_name}.csv"
    )

    after_df.to_csv(
        after_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"After 완료: {len(after_df):,}개 "
        f"| {time.time() - start:.1f}초"
    )


    # ========================================================
    # 4. Forward / Backward mapping
    # ========================================================

    print(
        f"\n[4/5] {split_name} ambiguity 검사 중..."
    )

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
    # 최종 row 생성
    # ========================================================

    mapping_rows = []

    reliable = 0
    ambiguous = 0
    failed = 0
    speaker_failed = 0


    for i in range(len(A)):

        f_row = forward[i]
        b_row = backward[i]


        # 매핑 실패
        if f_row == -1 or b_row == -1:

            status = "FAILED"

            failed += 1

            filename = None
            speaker_code = None
            sex = None
            age = None
            pseudo_id = None
            before_row = None


        # ambiguity 존재
        elif f_row != b_row:

            status = "AMBIGUOUS"

            ambiguous += 1

            filename = None
            speaker_code = None
            sex = None
            age = None
            pseudo_id = None
            before_row = None


        # Reliable
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

                speaker_code = info[
                    "speaker_code"
                ]

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
    # 5. 저장 및 검증
    # ========================================================

    print(
        f"\n[5/5] {split_name} 결과 저장 중..."
    )

    output_file = os.path.join(
        RESULT_DIR,
        f"stroke_{split_name}_mapping.csv"
    )

    mapping_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )


    reliable_df = mapping_df[
        mapping_df["mapping_status"]
        == "RELIABLE"
    ]


    unique_speakers = (
        reliable_df[
            "pseudo_speaker_id"
        ]
        .dropna()
        .nunique()
    )


    print(
        f"\n========== {split_name.upper()} 결과 =========="
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
        f"Pseudo-speakers  : {unique_speakers:,}"
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
        f"저장 위치        : {output_file}"
    )


    # Ambiguous 예시
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
            .head(10)
            .to_string(index=False)
        )


    summary.append({
        "split": split_name,
        "before": len(B),
        "after": len(A),
        "reliable": reliable,
        "ambiguous": ambiguous,
        "failed": failed,
        "speaker_failed": speaker_failed,
        "pseudo_speakers": unique_speakers
    })


# ============================================================
# 전체 Summary
# ============================================================

print("\n")
print("=" * 60)
print("STROKE VALID / TEST 최종 SUMMARY")
print("=" * 60)

summary_df = pd.DataFrame(summary)

print(
    summary_df.to_string(
        index=False
    )
)

print("\n전체 처리 완료!")