from datasets import load_dataset, Audio
from transformers import WhisperTokenizer
import pandas as pd
import os
import time


# ============================================================
# 설정
# ============================================================

BEFORE_COUNT = 15166
AFTER_COUNT = 14047
PROGRESS_INTERVAL = 1000

OUTPUT_DIR = "data"

BEFORE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stroke_before_train.csv"
)

AFTER_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "stroke_after_train.csv"
)


def normalize_text(text):
    """앞뒤 공백 및 연속 공백 정리"""
    return " ".join(str(text).strip().split())


# ============================================================
# 출력 폴더 생성
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("========== Stroke Metadata 저장 ==========\n")


# ============================================================
# 1. 데이터셋 연결
# ============================================================

print("[1/5] Hugging Face 데이터셋 연결 중...")

before = load_dataset(
    "yoona-J/ASR_Stroke_Dataset",
    streaming=True
)

after = load_dataset(
    "yoona-J/ASR_Preprocess_Stroke_Dataset",
    streaming=True
)

# 실제 WAV를 decoding하지 않음
before = before.cast_column(
    "audio",
    Audio(decode=False)
)

print("데이터셋 연결 완료\n")


# ============================================================
# 2. Tokenizer 준비
# ============================================================

print("[2/5] Whisper tokenizer 준비 중...")

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Korean",
    task="transcribe",
    clean_up_tokenization_spaces=False
)

print("Tokenizer 준비 완료\n")


# ============================================================
# 3. Before metadata 저장
# ============================================================

print("[3/5] Before Train metadata 읽는 중...")

before_metadata = []

start_time = time.time()


for i, row in enumerate(before["train"], start=1):

    before_metadata.append({
        "before_row": i - 1,
        "filename": row["audio"]["path"],
        "transcript": row["transcripts"],
        "normalized_transcript": normalize_text(
            row["transcripts"]
        )
    })

    if i % PROGRESS_INTERVAL == 0:
        elapsed = time.time() - start_time

        print(
            f"  Before: {i:,} / {BEFORE_COUNT:,} "
            f"({i / BEFORE_COUNT * 100:.1f}%) "
            f"| {elapsed:.1f}초"
        )

    if i >= BEFORE_COUNT:
        break


before_df = pd.DataFrame(before_metadata)

before_df.to_csv(
    BEFORE_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nBefore 저장 완료: "
    f"{len(before_df):,}개"
)

print(f"저장 위치: {BEFORE_OUTPUT}\n")


# ============================================================
# 4. After metadata 저장
# ============================================================

print("[4/5] Processed Train metadata 읽는 중...")

after_metadata = []

start_time = time.time()


for i, row in enumerate(after["train"], start=1):

    decoded = tokenizer.decode(
        row["labels"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    after_metadata.append({
        "processed_row": i - 1,
        "decoded_transcript": decoded,
        "normalized_transcript": normalize_text(decoded)
    })

    if i % PROGRESS_INTERVAL == 0:
        elapsed = time.time() - start_time

        print(
            f"  After: {i:,} / {AFTER_COUNT:,} "
            f"({i / AFTER_COUNT * 100:.1f}%) "
            f"| {elapsed:.1f}초"
        )

    if i >= AFTER_COUNT:
        break


after_df = pd.DataFrame(after_metadata)

after_df.to_csv(
    AFTER_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nAfter 저장 완료: "
    f"{len(after_df):,}개"
)

print(f"저장 위치: {AFTER_OUTPUT}\n")


# ============================================================
# 5. 최종 확인
# ============================================================

print("[5/5] 저장 결과 확인\n")

print("========== 결과 ==========")

print(
    f"Before rows : "
    f"{len(before_df):,}"
)

print(
    f"After rows  : "
    f"{len(after_df):,}"
)

print(
    f"Before 파일 존재 : "
    f"{os.path.exists(BEFORE_OUTPUT)}"
)

print(
    f"After 파일 존재  : "
    f"{os.path.exists(AFTER_OUTPUT)}"
)

print("\n저장 완료!")