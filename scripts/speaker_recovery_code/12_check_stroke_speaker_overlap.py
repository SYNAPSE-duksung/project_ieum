import pandas as pd
import os


# ============================================================
# 파일 경로
# ============================================================

TRAIN_FILE = "results/stroke_train_mapping.csv"
VALID_FILE = "results/stroke_valid_mapping.csv"
TEST_FILE = "results/stroke_test_mapping.csv"

RESULT_FILE = "results/stroke_speaker_overlap.csv"


print("========== Stroke Speaker Overlap 검사 ==========\n")


# ============================================================
# 1. Mapping 파일 읽기
# ============================================================

print("[1/4] Mapping 파일 읽는 중...")

train = pd.read_csv(TRAIN_FILE)
valid = pd.read_csv(VALID_FILE)
test = pd.read_csv(TEST_FILE)


# valid/test는 혹시 모르니 RELIABLE만 사용
if "mapping_status" in valid.columns:
    valid = valid[
        valid["mapping_status"] == "RELIABLE"
    ].copy()

if "mapping_status" in test.columns:
    test = test[
        test["mapping_status"] == "RELIABLE"
    ].copy()


print(f"Train rows : {len(train):,}")
print(f"Valid rows : {len(valid):,}")
print(f"Test rows  : {len(test):,}")


# ============================================================
# 2. 화자 집합 만들기
# ============================================================

print("\n[2/4] 화자 집합 생성 중...")


train_speakers = set(
    train["pseudo_speaker_id"]
    .dropna()
    .unique()
)

valid_speakers = set(
    valid["pseudo_speaker_id"]
    .dropna()
    .unique()
)

test_speakers = set(
    test["pseudo_speaker_id"]
    .dropna()
    .unique()
)


print(f"Train speakers : {len(train_speakers):,}")
print(f"Valid speakers : {len(valid_speakers):,}")
print(f"Test speakers  : {len(test_speakers):,}")


# ============================================================
# 3. Split 간 화자 중복 검사
# ============================================================

print("\n[3/4] Split 간 중복 검사 중...")


train_valid = train_speakers & valid_speakers
train_test = train_speakers & test_speakers
valid_test = valid_speakers & test_speakers

all_three = (
    train_speakers
    & valid_speakers
    & test_speakers
)


print("\n========== Speaker Overlap 결과 ==========")

print(
    f"Train ∩ Valid      : "
    f"{len(train_valid):,}"
)

print(
    f"Train ∩ Test       : "
    f"{len(train_test):,}"
)

print(
    f"Valid ∩ Test       : "
    f"{len(valid_test):,}"
)

print(
    f"Train ∩ Valid ∩ Test : "
    f"{len(all_three):,}"
)


# ============================================================
# 각 split에만 존재하는 화자
# ============================================================

train_only = (
    train_speakers
    - valid_speakers
    - test_speakers
)

valid_only = (
    valid_speakers
    - train_speakers
    - test_speakers
)

test_only = (
    test_speakers
    - train_speakers
    - valid_speakers
)


print("\n========== Split 전용 Speaker ==========")

print(
    f"Train only : {len(train_only):,}"
)

print(
    f"Valid only : {len(valid_only):,}"
)

print(
    f"Test only  : {len(test_only):,}"
)


# ============================================================
# 4. 화자별 split 발화 수 표 생성
# ============================================================

print("\n[4/4] 화자별 발화 수 계산 중...")


train_counts = (
    train["pseudo_speaker_id"]
    .value_counts()
)

valid_counts = (
    valid["pseudo_speaker_id"]
    .value_counts()
)

test_counts = (
    test["pseudo_speaker_id"]
    .value_counts()
)


all_speakers = sorted(
    train_speakers
    | valid_speakers
    | test_speakers
)


rows = []


for speaker in all_speakers:

    train_count = int(
        train_counts.get(speaker, 0)
    )

    valid_count = int(
        valid_counts.get(speaker, 0)
    )

    test_count = int(
        test_counts.get(speaker, 0)
    )

    split_count = sum([
        train_count > 0,
        valid_count > 0,
        test_count > 0
    ])

    rows.append({
        "pseudo_speaker_id": speaker,
        "train_count": train_count,
        "valid_count": valid_count,
        "test_count": test_count,
        "total_count":
            train_count
            + valid_count
            + test_count,
        "num_splits": split_count
    })


summary = pd.DataFrame(rows)

summary = summary.sort_values(
    "total_count",
    ascending=False
)


# ============================================================
# 결과 출력
# ============================================================

print("\n========== 전체 화자별 분포 ==========")

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 저장
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

summary.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"\n결과 저장 위치: {RESULT_FILE}"
)

print("\n검사 완료!")