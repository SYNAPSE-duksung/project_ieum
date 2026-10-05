import pandas as pd
import os


# ============================================================
# 설정
# ============================================================

RESULT_DIR = "results"

TRAIN_FILE = os.path.join(
    RESULT_DIR,
    "cerebral_train_mapping_resolved.csv"
)

VALID_FILE = os.path.join(
    RESULT_DIR,
    "cerebral_valid_mapping.csv"
)

TEST_FILE = os.path.join(
    RESULT_DIR,
    "cerebral_test_mapping.csv"
)


OUTPUT_TRAIN = os.path.join(
    RESULT_DIR,
    "cerebral_arg_train_mapping.csv"
)

OUTPUT_VALID = os.path.join(
    RESULT_DIR,
    "cerebral_arg_valid_mapping.csv"
)

OUTPUT_TEST = os.path.join(
    RESULT_DIR,
    "cerebral_arg_test_mapping.csv"
)

OUTPUT_SUMMARY = os.path.join(
    RESULT_DIR,
    "cerebral_arg_mapping_summary.csv"
)


# ============================================================
# 기대 row 개수
# ============================================================

BASE_TRAIN = 3226
BASE_VALID = 180
BASE_TEST = 165

ARG_TRAIN = 6452
ARG_VALID = 180
ARG_TEST = 165


# ============================================================
# 시작
# ============================================================

print(
    "========== Cerebral Palsy _Arg Mapping 생성 ==========\n"
)


# ============================================================
# 1. Base Mapping 불러오기
# ============================================================

print("[1/5] Cerebral Base Mapping 불러오는 중...")


train_df = pd.read_csv(
    TRAIN_FILE
)

valid_df = pd.read_csv(
    VALID_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)


print(
    f"  Train : {len(train_df):,}"
)

print(
    f"  Valid : {len(valid_df):,}"
)

print(
    f"  Test  : {len(test_df):,}"
)


# ============================================================
# Base 개수 검증
# ============================================================

if len(train_df) != BASE_TRAIN:

    raise ValueError(
        f"Train 개수 오류: "
        f"{len(train_df):,} != {BASE_TRAIN:,}"
    )


if len(valid_df) != BASE_VALID:

    raise ValueError(
        f"Valid 개수 오류: "
        f"{len(valid_df):,} != {BASE_VALID:,}"
    )


if len(test_df) != BASE_TEST:

    raise ValueError(
        f"Test 개수 오류: "
        f"{len(test_df):,} != {BASE_TEST:,}"
    )


# ============================================================
# 누락 여부 확인
# ============================================================

for split_name, df in [
    ("train", train_df),
    ("valid", valid_df),
    ("test", test_df)
]:

    filename_missing = (
        df["filename"]
        .isna()
        .sum()
    )

    speaker_missing = (
        df["pseudo_speaker_id"]
        .isna()
        .sum()
    )

    if filename_missing != 0:

        raise ValueError(
            f"{split_name}: "
            f"filename 누락 "
            f"{filename_missing:,}개"
        )

    if speaker_missing != 0:

        raise ValueError(
            f"{split_name}: "
            f"speaker 누락 "
            f"{speaker_missing:,}개"
        )


print("  ✓ Base Mapping 검증 완료")


# ============================================================
# 공통 함수
# ============================================================

def make_arg_row(
    arg_row,
    source_row,
    source,
    is_augmented,
    split_name
):

    return {
        "split":
            split_name,

        "processed_row":
            arg_row,

        # 이 Arg row가 어느 Base processed row에서 왔는지
        "original_processed_row":
            source_row,

        "filename":
            source["filename"],

        "transcript":
            source["transcript"],

        "speaker_code":
            source["speaker_code"],

        "sex":
            source["sex"],

        "age":
            source["age"],

        "pseudo_speaker_id":
            source["pseudo_speaker_id"],

        # 증강 여부
        "is_augmented":
            is_augmented,

        # 증강 데이터인 경우 원본 source row
        # original인 경우 자기 자신의 source row
        "augmentation_source_row":
            source_row,

        "mapping_status":
            "INHERITED_FROM_BASE"
    }


# ============================================================
# 2. Train _Arg Mapping
# ============================================================

print(
    "\n[2/5] Train _Arg Mapping 생성 중..."
)

train_rows = []


# ------------------------------------------------------------
# Train 앞쪽 3,226개
# = original
# ------------------------------------------------------------

for i in range(BASE_TRAIN):

    source = train_df.iloc[i]

    train_rows.append(
        make_arg_row(
            arg_row=i,
            source_row=i,
            source=source,
            is_augmented=False,
            split_name="train"
        )
    )


# ------------------------------------------------------------
# Train 뒤쪽 3,226개
# = 앞쪽 original의 augmentation
#
# Arg row 3226 → Base row 0
# Arg row 3227 → Base row 1
# ...
# Arg row 6451 → Base row 3225
# ------------------------------------------------------------

for i in range(BASE_TRAIN):

    arg_row = (
        BASE_TRAIN + i
    )

    source = train_df.iloc[i]

    train_rows.append(
        make_arg_row(
            arg_row=arg_row,
            source_row=i,
            source=source,
            is_augmented=True,
            split_name="train"
        )
    )


arg_train_df = pd.DataFrame(
    train_rows
)


print(
    f"  생성 rows : {len(arg_train_df):,}"
)

print(
    f"  Original  : "
    f"{(~arg_train_df['is_augmented']).sum():,}"
)

print(
    f"  Augmented : "
    f"{arg_train_df['is_augmented'].sum():,}"
)


# ============================================================
# 3. Valid _Arg Mapping
# ============================================================

print(
    "\n[3/5] Valid _Arg Mapping 생성 중..."
)

valid_rows = []


for i in range(BASE_VALID):

    source = valid_df.iloc[i]

    valid_rows.append(
        make_arg_row(
            arg_row=i,
            source_row=i,
            source=source,
            is_augmented=False,
            split_name="valid"
        )
    )


arg_valid_df = pd.DataFrame(
    valid_rows
)


print(
    f"  생성 rows : {len(arg_valid_df):,}"
)


# ============================================================
# 4. Test _Arg Mapping
# ============================================================

print(
    "\n[4/5] Test _Arg Mapping 생성 중..."
)

test_rows = []


for i in range(BASE_TEST):

    source = test_df.iloc[i]

    test_rows.append(
        make_arg_row(
            arg_row=i,
            source_row=i,
            source=source,
            is_augmented=False,
            split_name="test"
        )
    )


arg_test_df = pd.DataFrame(
    test_rows
)


print(
    f"  생성 rows : {len(arg_test_df):,}"
)


# ============================================================
# 5. 최종 검증
# ============================================================

print(
    "\n[5/5] 최종 검증 중..."
)


# ------------------------------------------------------------
# 전체 row 수
# ------------------------------------------------------------

if len(arg_train_df) != ARG_TRAIN:

    raise ValueError(
        "Arg Train row 수가 "
        "예상값과 다릅니다."
    )


if len(arg_valid_df) != ARG_VALID:

    raise ValueError(
        "Arg Valid row 수가 "
        "예상값과 다릅니다."
    )


if len(arg_test_df) != ARG_TEST:

    raise ValueError(
        "Arg Test row 수가 "
        "예상값과 다릅니다."
    )


# ------------------------------------------------------------
# Train original / augmentation 쌍 검증
# ------------------------------------------------------------

original_df = arg_train_df[
    arg_train_df["is_augmented"] == False
].reset_index(drop=True)


augmented_df = arg_train_df[
    arg_train_df["is_augmented"] == True
].reset_index(drop=True)


if len(original_df) != BASE_TRAIN:

    raise ValueError(
        "Original Train 개수 오류"
    )


if len(augmented_df) != BASE_TRAIN:

    raise ValueError(
        "Augmented Train 개수 오류"
    )


# ------------------------------------------------------------
# Original ↔ Augmentation source가 동일한지 검증
# ------------------------------------------------------------

filename_match = (
    original_df["filename"]
    .reset_index(drop=True)
    ==
    augmented_df["filename"]
    .reset_index(drop=True)
).all()


speaker_match = (
    original_df["pseudo_speaker_id"]
    .reset_index(drop=True)
    ==
    augmented_df["pseudo_speaker_id"]
    .reset_index(drop=True)
).all()


transcript_match = (
    original_df["transcript"]
    .reset_index(drop=True)
    ==
    augmented_df["transcript"]
    .reset_index(drop=True)
).all()


if not filename_match:

    raise ValueError(
        "Original ↔ Augmentation "
        "filename 불일치"
    )


if not speaker_match:

    raise ValueError(
        "Original ↔ Augmentation "
        "speaker 불일치"
    )


if not transcript_match:

    raise ValueError(
        "Original ↔ Augmentation "
        "transcript 불일치"
    )


# ------------------------------------------------------------
# 누락 확인
# ------------------------------------------------------------

all_arg_df = pd.concat(
    [
        arg_train_df,
        arg_valid_df,
        arg_test_df
    ],
    ignore_index=True
)


filename_missing = (
    all_arg_df["filename"]
    .isna()
    .sum()
)

speaker_missing = (
    all_arg_df["pseudo_speaker_id"]
    .isna()
    .sum()
)


if filename_missing != 0:

    raise ValueError(
        f"Filename 누락: "
        f"{filename_missing:,}"
    )


if speaker_missing != 0:

    raise ValueError(
        f"Speaker 누락: "
        f"{speaker_missing:,}"
    )


# ============================================================
# 저장
# ============================================================

arg_train_df.to_csv(
    OUTPUT_TRAIN,
    index=False,
    encoding="utf-8-sig"
)

arg_valid_df.to_csv(
    OUTPUT_VALID,
    index=False,
    encoding="utf-8-sig"
)

arg_test_df.to_csv(
    OUTPUT_TEST,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Summary 생성
# ============================================================

summary_df = pd.DataFrame([
    {
        "split": "train",
        "arg_rows":
            len(arg_train_df),

        "original_rows":
            (~arg_train_df[
                "is_augmented"
            ]).sum(),

        "augmented_rows":
            arg_train_df[
                "is_augmented"
            ].sum(),

        "unique_original_utterances":
            arg_train_df[
                "original_processed_row"
            ].nunique(),

        "pseudo_speakers":
            arg_train_df[
                "pseudo_speaker_id"
            ].nunique()
    },

    {
        "split": "valid",
        "arg_rows":
            len(arg_valid_df),

        "original_rows":
            len(arg_valid_df),

        "augmented_rows":
            0,

        "unique_original_utterances":
            len(arg_valid_df),

        "pseudo_speakers":
            arg_valid_df[
                "pseudo_speaker_id"
            ].nunique()
    },

    {
        "split": "test",
        "arg_rows":
            len(arg_test_df),

        "original_rows":
            len(arg_test_df),

        "augmented_rows":
            0,

        "unique_original_utterances":
            len(arg_test_df),

        "pseudo_speakers":
            arg_test_df[
                "pseudo_speaker_id"
            ].nunique()
    }
])


summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 결과 출력
# ============================================================

print("\n========== _Arg 최종 검증 ==========")

print(
    f"Arg Train rows       : "
    f"{len(arg_train_df):,}"
)

print(
    f"  ├─ Original        : "
    f"{len(original_df):,}"
)

print(
    f"  └─ Augmented       : "
    f"{len(augmented_df):,}"
)

print(
    f"Arg Valid rows       : "
    f"{len(arg_valid_df):,}"
)

print(
    f"Arg Test rows        : "
    f"{len(arg_test_df):,}"
)

print(
    f"\n전체 Arg rows        : "
    f"{len(all_arg_df):,}"
)

print(
    "실제 원본 발화 수    : "
    f"{BASE_TRAIN + BASE_VALID + BASE_TEST:,}"
)

print(
    f"Filename 누락        : "
    f"{filename_missing:,}"
)

print(
    f"Speaker ID 누락      : "
    f"{speaker_missing:,}"
)

print(
    f"Original-Aug filename 일치 : "
    f"{filename_match}"
)

print(
    f"Original-Aug speaker 일치  : "
    f"{speaker_match}"
)

print(
    f"Original-Aug transcript 일치: "
    f"{transcript_match}"
)

print(
    "\n========== Summary =========="
)

print(
    summary_df.to_string(
        index=False
    )
)

print(
    "\n저장 완료:"
)

print(
    f"  {OUTPUT_TRAIN}"
)

print(
    f"  {OUTPUT_VALID}"
)

print(
    f"  {OUTPUT_TEST}"
)

print(
    f"  {OUTPUT_SUMMARY}"
)

print(
    "\nCerebral Palsy _Arg Mapping 완료!"
)