from datasets import load_dataset, Audio
from collections import Counter
from itertools import islice


print("Stroke train 데이터 불러오는 중...")

dataset = load_dataset(
    "yoona-J/ASR_Stroke_Dataset",
    streaming=True
)

dataset = dataset.cast_column(
    "audio",
    Audio(decode=False)
)


def normalize_text(text):
    return " ".join(str(text).strip().split())


# 이전에 확인한 Stroke train 전체 개수
TRAIN_COUNT = 15166

rows = list(islice(dataset["train"], TRAIN_COUNT))

print(f"읽은 행 수: {len(rows):,}")


# transcript 정규화
texts = [
    normalize_text(row["transcripts"])
    for row in rows
]

counts = Counter(texts)


# 같은 transcript가 2번 이상 등장한 경우
duplicates = {
    text: count
    for text, count in counts.items()
    if count >= 2
}


print("\n========== 중복 Transcript 검사 ==========")

print(f"전체 행 수       : {len(texts):,}")
print(f"고유 transcript  : {len(counts):,}")
print(f"중복 문장 종류   : {len(duplicates):,}")
print(
    f"중복 문장에 속한 행: "
    f"{sum(duplicates.values()):,}"
)


print("\n========== 많이 중복된 문장 TOP 20 ==========")

for text, count in sorted(
    duplicates.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:

    print(f"{count:4d}회 | {text}")