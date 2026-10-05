from datasets import load_dataset, Audio
from transformers import WhisperTokenizer
from itertools import islice


print("데이터셋 불러오는 중...")

before = load_dataset(
    "yoona-J/ASR_Stroke_Dataset",
    streaming=True
)

after = load_dataset(
    "yoona-J/ASR_Preprocess_Stroke_Dataset",
    streaming=True
)

# 실제 음성 decoding 방지
before = before.cast_column(
    "audio",
    Audio(decode=False)
)


print("Tokenizer 불러오는 중...")

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Korean",
    task="transcribe",
    clean_up_tokenization_spaces=False
)


# --------------------------------------------------
# 비교를 위한 간단한 텍스트 정규화
# --------------------------------------------------

def normalize_text(text):
    return " ".join(str(text).strip().split())


# --------------------------------------------------
# 테스트 범위
# --------------------------------------------------

# processed 데이터는 100개만 확인
after_rows = list(islice(after["train"], 100))

# filter로 일부가 제거됐을 수 있으므로
# before 쪽은 조금 더 넉넉하게 읽음
before_rows = list(islice(before["train"], 300))


before_index = 0

matched = 0
failed = 0


print("\n========== Mapping Test ==========\n")


for processed_index, after_row in enumerate(after_rows):

    decoded = tokenizer.decode(
        after_row["labels"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    decoded = normalize_text(decoded)

    found = False

    # 현재 위치부터 앞으로만 탐색
    while before_index < len(before_rows):

        before_row = before_rows[before_index]

        transcript = normalize_text(
            before_row["transcripts"]
        )

        if transcript == decoded:

            filename = before_row["audio"]["path"]

            print(
                f"[Processed {processed_index:03d}] "
                f"→ Before {before_index:03d} | "
                f"{filename} | MATCH"
            )

            matched += 1
            before_index += 1
            found = True

            break

        # 일치하지 않으면
        # filter에서 제거된 row라고 보고 다음 before로 이동
        before_index += 1

    if not found:

        print(
            f"[Processed {processed_index:03d}] "
            f"→ MATCH NOT FOUND | {decoded}"
        )

        failed += 1


print("\n========== 결과 ==========")

print(f"Processed rows : {len(after_rows)}")
print(f"Matched        : {matched}")
print(f"Failed         : {failed}")
print(
    f"Match rate     : "
    f"{matched / len(after_rows) * 100:.2f}%"
)