from datasets import load_dataset, Audio
from transformers import WhisperTokenizer
from collections import Counter
import time


# ============================================================
# 설정
# ============================================================

BEFORE_COUNT = 15166
AFTER_COUNT = 14047
PROGRESS_INTERVAL = 1000


def normalize_text(text):
    """비교를 위해 앞뒤/연속 공백을 정리"""
    return " ".join(str(text).strip().split())


# ============================================================
# 1. 데이터셋 불러오기
# ============================================================

print("========== Stroke Train 전체 매핑 검증 ==========\n")

print("[1/4] 데이터셋 연결 중...")

before = load_dataset(
    "yoona-J/ASR_Stroke_Dataset",
    streaming=True
)

after = load_dataset(
    "yoona-J/ASR_Preprocess_Stroke_Dataset",
    streaming=True
)

# 실제 WAV decoding 방지
before = before.cast_column(
    "audio",
    Audio(decode=False)
)

print("데이터셋 연결 완료\n")


# ============================================================
# 2. Tokenizer 준비
# ============================================================

print("[2/4] Whisper tokenizer 준비 중...")

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Korean",
    task="transcribe",
    clean_up_tokenization_spaces=False
)

print("Tokenizer 준비 완료\n")


# ============================================================
# 3. Before 데이터 읽기
# ============================================================

print("[3/4] Before Train 데이터 읽는 중...")

before_rows = []

start_time = time.time()

for i, row in enumerate(before["train"], start=1):

    before_rows.append({
        "before_row": i - 1,
        "filename": row["audio"]["path"],
        "transcript": normalize_text(row["transcripts"])
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


print(
    f"Before 읽기 완료: {len(before_rows):,}개\n"
)


# ============================================================
# 4. After를 순서대로 매핑
# ============================================================

print("[4/4] Processed Train 전체 매핑 시작...\n")

before_index = 0

matched = 0
failed = 0
skipped_before = 0

# 어떤 before row가 filter로 건너뛰어졌는지 기록
skipped_rows = []

# 매핑 결과 저장
mapping_results = []

start_time = time.time()


for processed_index, after_row in enumerate(after["train"]):

    # labels -> 실제 정답 문장 복원
    decoded = tokenizer.decode(
        after_row["labels"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    decoded = normalize_text(decoded)

    found = False

    search_start = before_index


    # --------------------------------------------------------
    # 현재 위치부터 앞으로 이동하면서 같은 transcript 탐색
    # --------------------------------------------------------

    while before_index < len(before_rows):

        candidate = before_rows[before_index]

        if candidate["transcript"] == decoded:

            # 이 위치에서 일치
            mapping_results.append({
                "processed_row": processed_index,
                "before_row": candidate["before_row"],
                "filename": candidate["filename"],
                "transcript": decoded,
                "status": "MATCH"
            })

            matched += 1
            before_index += 1
            found = True

            break

        else:
            # processed 데이터에는 없으므로
            # filter 과정에서 제거된 before row로 간주
            skipped_rows.append(before_rows[before_index])
            skipped_before += 1
            before_index += 1


    # --------------------------------------------------------
    # 끝까지 찾아도 일치하지 않은 경우
    # --------------------------------------------------------

    if not found:

        mapping_results.append({
            "processed_row": processed_index,
            "before_row": None,
            "filename": None,
            "transcript": decoded,
            "status": "FAILED"
        })

        failed += 1


    # --------------------------------------------------------
    # 진행 상황 출력
    # --------------------------------------------------------

    current = processed_index + 1

    if (
        current % PROGRESS_INTERVAL == 0
        or current == AFTER_COUNT
    ):

        elapsed = time.time() - start_time

        print(
            f"  After: {current:,} / {AFTER_COUNT:,} "
            f"({current / AFTER_COUNT * 100:.1f}%) "
            f"| MATCH {matched:,} "
            f"| FAILED {failed:,} "
            f"| Before skip {skipped_before:,} "
            f"| {elapsed:.1f}초"
        )


    if current >= AFTER_COUNT:
        break


# ============================================================
# 결과 요약
# ============================================================

print("\n========== Stroke Train Mapping 결과 ==========")

print(f"Before rows     : {len(before_rows):,}")
print(f"Processed rows  : {len(mapping_results):,}")
print(f"Matched         : {matched:,}")
print(f"Failed          : {failed:,}")
print(f"Skipped Before  : {skipped_before:,}")

if mapping_results:

    print(
        f"Match rate      : "
        f"{matched / len(mapping_results) * 100:.2f}%"
    )


# ============================================================
# 추가 일관성 검사
# ============================================================

print("\n========== 일관성 검사 ==========")

# 성공한 filename 중 중복이 있는지 확인
matched_filenames = [
    row["filename"]
    for row in mapping_results
    if row["status"] == "MATCH"
]

filename_counts = Counter(matched_filenames)

duplicate_filenames = [
    filename
    for filename, count in filename_counts.items()
    if count > 1
]

print(f"매핑된 filename 수       : {len(matched_filenames):,}")
print(f"고유 filename 수         : {len(filename_counts):,}")
print(f"중복 매핑 filename 수    : {len(duplicate_filenames):,}")


# 마지막으로 남은 before row 수
remaining_before = len(before_rows) - before_index

print(f"마지막에 남은 Before rows: {remaining_before:,}")


print("\n검증 완료!")