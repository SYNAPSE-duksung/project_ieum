from datasets import load_dataset, Audio


print("========== General Dataset Split 확인 ==========\n")


# ============================================================
# 이미 확인된 Train 개수
# ============================================================

KNOWN_BEFORE_TRAIN = 38911
KNOWN_AFTER_TRAIN = 36170


# ============================================================
# 1. 전처리 이전 데이터
# ============================================================

print("[1/2] 전처리 이전 General 데이터 연결 중...")

before = load_dataset(
    "yoona-J/ASR_Disease_General_Dataset",
    streaming=True
)

# 실제 음성 decoding 방지
before = before.cast_column(
    "audio",
    Audio(decode=False)
)

print("\nBefore split 목록:")
print(list(before.keys()))


before_counts = {
    "train": KNOWN_BEFORE_TRAIN
}


for split_name in before.keys():

    if split_name == "train":
        print(
            f"\ntrain: {KNOWN_BEFORE_TRAIN:,}개 "
            "(기존 확인값 사용)"
        )
        continue

    print(
        f"\nBefore {split_name} 개수 확인 중..."
    )

    count = 0

    for _ in before[split_name]:

        count += 1

        if count % 1000 == 0:
            print(f"  {count:,}개 확인")

    before_counts[split_name] = count

    print(
        f"Before {split_name}: {count:,}개"
    )


# ============================================================
# 2. 전처리 이후 데이터
# ============================================================

print("\n[2/2] 전처리 이후 General 데이터 연결 중...")

after = load_dataset(
    "yoona-J/ASR_Preprocess_Disease_General_Dataset",
    streaming=True
)

print("\nAfter split 목록:")
print(list(after.keys()))


after_counts = {
    "train": KNOWN_AFTER_TRAIN
}


for split_name in after.keys():

    if split_name == "train":
        print(
            f"\ntrain: {KNOWN_AFTER_TRAIN:,}개 "
            "(기존 확인값 사용)"
        )
        continue

    print(
        f"\nAfter {split_name} 개수 확인 중..."
    )

    count = 0

    for _ in after[split_name]:

        count += 1

        if count % 1000 == 0:
            print(f"  {count:,}개 확인")

    after_counts[split_name] = count

    print(
        f"After {split_name}: {count:,}개"
    )


# ============================================================
# 3. 비교
# ============================================================

print("\n========== General Split 비교 ==========")

all_splits = sorted(
    set(before_counts)
    | set(after_counts)
)


for split_name in all_splits:

    before_count = before_counts.get(
        split_name,
        0
    )

    after_count = after_counts.get(
        split_name,
        0
    )

    removed = before_count - after_count

    print(
        f"{split_name:8s} | "
        f"Before {before_count:6,d} | "
        f"After {after_count:6,d} | "
        f"차이 {removed:5,d}"
    )


print("\n확인 완료!")