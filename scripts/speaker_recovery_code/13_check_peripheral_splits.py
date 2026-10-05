from datasets import load_dataset, Audio


print("========== Peripheral Neuropathy Split 확인 ==========\n")


# ============================================================
# 데이터셋 연결
# ============================================================

print("[1/2] 전처리 이전 데이터 연결 중...")

before = load_dataset(
    "yoona-J/ASR_Peripheral_Neuropathy_Dataset",
    streaming=True
)

before = before.cast_column(
    "audio",
    Audio(decode=False)
)

print("Before split 목록:")
print(list(before.keys()))


print("\n[2/2] 전처리 이후 데이터 연결 중...")

after = load_dataset(
    "yoona-J/ASR_Preprocess_Peripheral_Neuropathy_Dataset",
    streaming=True
)

print("After split 목록:")
print(list(after.keys()))


# ============================================================
# Train은 이미 확인했으므로 기존 값 사용
# ============================================================

before_counts = {
    "train": 19508
}

after_counts = {
    "train": 18679
}


# ============================================================
# Before valid / test만 확인
# ============================================================

for split_name in before.keys():

    if split_name == "train":
        continue

    print(
        f"\nBefore {split_name} 개수 확인 중..."
    )

    count = 0

    for _ in before[split_name]:

        count += 1

        if count % 1000 == 0:
            print(f"  {count:,}개")

    before_counts[split_name] = count

    print(
        f"Before {split_name}: {count:,}"
    )


# ============================================================
# After valid / test만 확인
# ============================================================

for split_name in after.keys():

    if split_name == "train":
        continue

    print(
        f"\nAfter {split_name} 개수 확인 중..."
    )

    count = 0

    for _ in after[split_name]:

        count += 1

        if count % 1000 == 0:
            print(f"  {count:,}개")

    after_counts[split_name] = count

    print(
        f"After {split_name}: {count:,}"
    )


# ============================================================
# 결과
# ============================================================

print("\n========== Peripheral Split 비교 ==========")

all_splits = sorted(
    set(before_counts)
    | set(after_counts)
)

for split_name in all_splits:

    b = before_counts.get(split_name, 0)
    a = after_counts.get(split_name, 0)

    print(
        f"{split_name:8s} | "
        f"Before {b:6,d} | "
        f"After {a:6,d} | "
        f"차이 {b-a:5,d}"
    )


print("\n확인 완료!")