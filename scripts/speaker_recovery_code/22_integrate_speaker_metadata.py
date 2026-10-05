import pandas as pd
import os


# ============================================================
# 설정
# ============================================================

RESULT_DIR = "results"

FILES = {
    "stroke": {
        "train": "stroke_train_mapping.csv",
        "valid": "stroke_valid_mapping.csv",
        "test": "stroke_test_mapping.csv",
    },

    "peripheral": {
        "train": "peripheral_train_mapping.csv",
        "valid": "peripheral_valid_mapping.csv",
        "test": "peripheral_test_mapping.csv",
    },

    "general": {
        "train": "general_train_mapping.csv",
        "valid": "general_valid_mapping.csv",
        "test": "general_test_mapping.csv",
    },

    "cerebral": {
        # Train은 ambiguous 2개를 해결한 최종 파일 사용
        "train": "cerebral_train_mapping_resolved.csv",
        "valid": "cerebral_valid_mapping.csv",
        "test": "cerebral_test_mapping.csv",
    },
}


OUTPUT_ALL = os.path.join(
    RESULT_DIR,
    "all_original_utterances.csv"
)

OUTPUT_SPEAKER_SUMMARY = os.path.join(
    RESULT_DIR,
    "all_pseudo_speaker_summary.csv"
)

OUTPUT_OVERLAP = os.path.join(
    RESULT_DIR,
    "cross_dataset_speaker_overlap.csv"
)

OUTPUT_CODE_CONFLICT = os.path.join(
    RESULT_DIR,
    "speaker_code_conflicts.csv"
)


# ============================================================
# 시작
# ============================================================

print("=" * 70)
print("전체 데이터셋 화자 통합 및 중복 검사")
print("=" * 70)


# ============================================================
# 1. 모든 Mapping 불러오기
# ============================================================

print("\n[1/6] Mapping 파일 불러오는 중...\n")

all_dfs = []


for dataset_name, split_files in FILES.items():

    for split_name, filename in split_files.items():

        path = os.path.join(
            RESULT_DIR,
            filename
        )

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다: {path}"
            )

        df = pd.read_csv(path)

        # 데이터셋 / 기존 split 정보 추가
        df["source_dataset"] = dataset_name
        df["original_split"] = split_name

        # ----------------------------------------------------
        # 아직 해결되지 않은 AMBIGUOUS / FAILED 제외
        # ----------------------------------------------------

        if "mapping_status" in df.columns:

            before_count = len(df)

            df = df[
                ~df["mapping_status"].isin(
                    ["AMBIGUOUS", "FAILED"]
                )
            ].copy()

            excluded = before_count - len(df)

        else:
            excluded = 0

        print(
            f"{dataset_name:10s} "
            f"{split_name:5s} | "
            f"사용 {len(df):6,d} | "
            f"제외 {excluded:,}"
        )

        all_dfs.append(df)


# ============================================================
# 2. 전체 통합
# ============================================================

print("\n[2/6] 전체 데이터 통합 중...")

all_df = pd.concat(
    all_dfs,
    ignore_index=True
)


print(
    f"통합 row 수        : {len(all_df):,}"
)

print(
    f"고유 filename 수   : "
    f"{all_df['filename'].nunique():,}"
)

print(
    f"고유 pseudo-speaker: "
    f"{all_df['pseudo_speaker_id'].nunique():,}"
)


# ============================================================
# 3. 동일 filename 중복 검사
# ============================================================

print("\n[3/6] 동일 원본 filename 중복 검사 중...")


filename_summary = (
    all_df
    .groupby("filename")
    .agg(
        row_count=(
            "filename",
            "size"
        ),

        dataset_count=(
            "source_dataset",
            "nunique"
        ),

        datasets=(
            "source_dataset",
            lambda x:
            "|".join(
                sorted(set(x))
            )
        ),

        pseudo_speaker_count=(
            "pseudo_speaker_id",
            "nunique"
        )
    )
    .reset_index()
)


duplicate_filename_df = filename_summary[
    filename_summary["row_count"] > 1
].copy()


print(
    f"중복 filename 종류 : "
    f"{len(duplicate_filename_df):,}"
)

print(
    "여러 dataset에 등장한 filename: "
    f"{(duplicate_filename_df['dataset_count'] > 1).sum():,}"
)


# ============================================================
# 4. Pseudo-speaker별 전체 요약
# ============================================================

print("\n[4/6] Pseudo-speaker별 통합 현황 계산 중...")


speaker_summary = (
    all_df
    .groupby("pseudo_speaker_id")
    .agg(
        total_rows=(
            "pseudo_speaker_id",
            "size"
        ),

        unique_filenames=(
            "filename",
            "nunique"
        ),

        dataset_count=(
            "source_dataset",
            "nunique"
        ),

        datasets=(
            "source_dataset",
            lambda x:
            "|".join(
                sorted(set(x))
            )
        ),

        split_count=(
            "original_split",
            "nunique"
        ),

        speaker_code=(
            "speaker_code",
            "first"
        ),

        sex=(
            "sex",
            "first"
        ),

        age=(
            "age",
            "first"
        )
    )
    .reset_index()
)


speaker_summary = speaker_summary.sort_values(
    [
        "dataset_count",
        "unique_filenames"
    ],
    ascending=[
        False,
        False
    ]
)


cross_dataset_speakers = speaker_summary[
    speaker_summary["dataset_count"] > 1
].copy()


print(
    f"전체 pseudo-speaker 수 : "
    f"{len(speaker_summary):,}"
)

print(
    f"2개 이상 dataset 등장 : "
    f"{len(cross_dataset_speakers):,}"
)


# ============================================================
# 5. speaker_code 충돌 검사
# ============================================================

print("\n[5/6] Speaker code 충돌 검사 중...")


# 같은 speaker_code인데
# sex 또는 age가 다르게 나타나는 경우 확인
code_summary = (
    speaker_summary
    .groupby("speaker_code")
    .agg(
        pseudo_speaker_count=(
            "pseudo_speaker_id",
            "nunique"
        ),

        pseudo_speakers=(
            "pseudo_speaker_id",
            lambda x:
            "|".join(
                sorted(set(x))
            )
        ),

        sex_count=(
            "sex",
            "nunique"
        ),

        sexes=(
            "sex",
            lambda x:
            "|".join(
                sorted(
                    set(
                        str(v)
                        for v in x
                    )
                )
            )
        ),

        age_count=(
            "age",
            "nunique"
        ),

        ages=(
            "age",
            lambda x:
            "|".join(
                sorted(
                    set(
                        str(v)
                        for v in x
                    )
                )
            )
        )
    )
    .reset_index()
)


code_conflicts = code_summary[
    code_summary["pseudo_speaker_count"] > 1
].copy()


print(
    f"고유 speaker_code 수      : "
    f"{len(code_summary):,}"
)

print(
    f"여러 pseudo ID를 가진 code: "
    f"{len(code_conflicts):,}"
)


# ============================================================
# 6. 저장
# ============================================================

print("\n[6/6] 결과 저장 중...")


all_df.to_csv(
    OUTPUT_ALL,
    index=False,
    encoding="utf-8-sig"
)


speaker_summary.to_csv(
    OUTPUT_SPEAKER_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


cross_dataset_speakers.to_csv(
    OUTPUT_OVERLAP,
    index=False,
    encoding="utf-8-sig"
)


code_conflicts.to_csv(
    OUTPUT_CODE_CONFLICT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 결과 출력
# ============================================================

print("\n" + "=" * 70)
print("전체 통합 결과")
print("=" * 70)

print(
    f"전체 rows              : "
    f"{len(all_df):,}"
)

print(
    f"고유 filename           : "
    f"{all_df['filename'].nunique():,}"
)

print(
    f"고유 pseudo-speaker     : "
    f"{all_df['pseudo_speaker_id'].nunique():,}"
)

print(
    f"여러 dataset 공유 화자  : "
    f"{len(cross_dataset_speakers):,}"
)

print(
    f"speaker_code 충돌       : "
    f"{len(code_conflicts):,}"
)


# ============================================================
# 데이터셋별 현황
# ============================================================

print("\n========== 데이터셋별 현황 ==========")

dataset_summary = (
    all_df
    .groupby("source_dataset")
    .agg(
        rows=(
            "filename",
            "size"
        ),

        unique_filenames=(
            "filename",
            "nunique"
        ),

        pseudo_speakers=(
            "pseudo_speaker_id",
            "nunique"
        )
    )
)

print(
    dataset_summary.to_string()
)


# ============================================================
# Cross-dataset speaker 출력
# ============================================================

print(
    "\n========== 여러 데이터셋에 등장한 화자 =========="
)

if len(cross_dataset_speakers) == 0:

    print("없음")

else:

    print(
        cross_dataset_speakers[
            [
                "pseudo_speaker_id",
                "unique_filenames",
                "dataset_count",
                "datasets"
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# Speaker code conflict 출력
# ============================================================

print(
    "\n========== Speaker Code 충돌 =========="
)

if len(code_conflicts) == 0:

    print("없음")

else:

    print(
        code_conflicts[
            [
                "speaker_code",
                "pseudo_speaker_count",
                "pseudo_speakers",
                "sex_count",
                "sexes",
                "age_count",
                "ages"
            ]
        ]
        .to_string(
            index=False
        )
    )


print("\n========== 저장 파일 ==========")

print(
    f"전체 발화        : {OUTPUT_ALL}"
)

print(
    f"화자 요약        : {OUTPUT_SPEAKER_SUMMARY}"
)

print(
    f"dataset 중복 화자: {OUTPUT_OVERLAP}"
)

print(
    f"화자 코드 충돌   : {OUTPUT_CODE_CONFLICT}"
)

print(
    "\n22단계 완료!"
)