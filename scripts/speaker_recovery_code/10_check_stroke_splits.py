from datasets import load_dataset, Audio


print("========== Stroke Dataset Split 확인 ==========\n")


# ============================================================
# 이미 확인된 Train 개수
# ============================================================

KNOWN_BEFORE_TRAIN = 15166
KNOWN_AFTER_TRAIN = 14047


# ============================================================
# 1. 전처리 이전 데이터
# ============================================================

print("[1/2] 전처리 이전 데이터 확인 중...")

before = load_dataset(
    "yoona-J/ASR_Stroke_Dataset",
    streaming=True
)

# 중요!
# 실제 음성 데이터를 decode하지 않도록 설정
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

    # Train은 이미 확인했으므로 다시 읽지 않음
    if split_name == "train":
        print(
            f"\ntrain: {KNOWN_BEFORE_TRAIN:,}개 "
            "(기존 확인값 사용)"
        )
        continue

    print(f"\n{split_name} 개수 확인 중...")

    count = 0

    for _ in before[split_name]:

        count += 1

        if count % 1000 == 0:
            print(
                f"  {count:,}개 확인"
            )

    before_counts[split_name] = count

    print(
        f"  {split_name}: "
        f"{count:,}개"
    )


# ============================================================
# 2. 전처리 이후 데이터
# ============================================================

print("\n[2/2] 전처리 이후 데이터 확인 중...")

after = load_dataset(
    "yoona-J/ASR_Preprocess_Stroke_Dataset",
    streaming=True
)

print("\nAfter split 목록:")
print(list(after.keys()))


after_counts = {
    "train": KNOWN_AFTER_TRAIN
}


for split_name in after.keys():

    # Train은 이미 확인했으므로 다시 읽지 않음
    if split_name == "train":
        print(
            f"\ntrain: {KNOWN_AFTER_TRAIN:,}개 "
            "(기존 확인값 사용)"
        )
        continue

    print(f"\n{split_name} 개수 확인 중...")

    count = 0

    for _ in after[split_name]:

        count += 1

        if count % 1000 == 0:
            print(
                f"  {count:,}개 확인"
            )

    after_counts[split_name] = count

    print(
        f"  {split_name}: "
        f"{count:,}개"
    )


# ============================================================
# 3. Before / After 비교
# ============================================================

print("\n========== Split 비교 ==========")


all_splits = sorted(
    set(before_counts.keys())
    | set(after_counts.keys())
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