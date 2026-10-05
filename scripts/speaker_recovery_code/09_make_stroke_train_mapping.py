import pandas as pd
import os


# ============================================================
# 설정
# ============================================================

AMBIGUITY_FILE = "results/stroke_train_ambiguity_check.csv"
SPEAKER_FILE = "results/stroke_speaker_extraction_check.csv"

RESULT_DIR = "results"

MAPPING_OUTPUT = os.path.join(
    RESULT_DIR,
    "stroke_train_mapping.csv"
)

SPEAKER_OUTPUT = os.path.join(
    RESULT_DIR,
    "stroke_train_speaker_summary.csv"
)


print("========== Stroke Train 최종 Mapping 생성 ==========\n")


# ============================================================
# 1. 파일 읽기
# ============================================================

print("[1/4] 기존 결과 파일 읽는 중...")

ambiguity_df = pd.read_csv(AMBIGUITY_FILE)
speaker_df = pd.read_csv(SPEAKER_FILE)

print(f"전체 Processed rows : {len(ambiguity_df):,}")
print(f"Speaker 추출 rows   : {len(speaker_df):,}")
print("파일 읽기 완료\n")


# ============================================================
# 2. Reliable 데이터 결합
# ============================================================

print("[2/4] 최종 mapping 생성 중...")

reliable_df = ambiguity_df[
    ambiguity_df["status"] == "RELIABLE"
].copy()

# speaker 정보 중 필요한 컬럼만 사용
speaker_info = speaker_df[
    [
        "processed_row",
        "speaker_code",
        "sex",
        "age",
        "pseudo_speaker_id"
    ]
].copy()


final_df = reliable_df.merge(
    speaker_info,
    on="processed_row",
    how="left",
    validate="one_to_one"
)


# 컬럼 정리
final_df = final_df[
    [
        "processed_row",
        "forward_before_row",
        "filename",
        "transcript",
        "speaker_code",
        "sex",
        "age",
        "pseudo_speaker_id",
        "status"
    ]
].copy()


final_df = final_df.rename(
    columns={
        "forward_before_row": "before_row",
        "status": "mapping_status"
    }
)


# ============================================================
# 3. 최종 검증
# ============================================================

print("[3/4] 최종 데이터 검증 중...\n")

print("========== 최종 Mapping 검증 ==========")

print(f"최종 rows             : {len(final_df):,}")

print(
    f"고유 filename         : "
    f"{final_df['filename'].nunique():,}"
)

print(
    f"고유 pseudo-speaker   : "
    f"{final_df['pseudo_speaker_id'].nunique():,}"
)

print(
    f"Speaker ID 누락       : "
    f"{final_df['pseudo_speaker_id'].isna().sum():,}"
)

print(
    f"Filename 누락         : "
    f"{final_df['filename'].isna().sum():,}"
)

print(
    f"Processed row 중복    : "
    f"{final_df['processed_row'].duplicated().sum():,}"
)

print(
    f"Filename 중복         : "
    f"{final_df['filename'].duplicated().sum():,}"
)


# ============================================================
# 4. 화자별 Summary 생성
# ============================================================

speaker_summary = (
    final_df
    .groupby(
        [
            "pseudo_speaker_id",
            "speaker_code",
            "sex",
            "age"
        ],
        as_index=False
    )
    .agg(
        utterance_count=("processed_row", "count")
    )
    .sort_values(
        "utterance_count",
        ascending=False
    )
)


print("\n========== 화자별 데이터 분포 ==========")

print(
    f"화자 수        : "
    f"{len(speaker_summary):,}"
)

print(
    f"최소 발화 수   : "
    f"{speaker_summary['utterance_count'].min():,}"
)

print(
    f"최대 발화 수   : "
    f"{speaker_summary['utterance_count'].max():,}"
)

print(
    f"평균 발화 수   : "
    f"{speaker_summary['utterance_count'].mean():.1f}"
)

print(
    f"중앙값 발화 수 : "
    f"{speaker_summary['utterance_count'].median():.1f}"
)


print("\n========== 발화 수 TOP 20 ==========")

print(
    speaker_summary
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 저장
# ============================================================

print("\n[4/4] CSV 저장 중...")

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

final_df.to_csv(
    MAPPING_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

speaker_summary.to_csv(
    SPEAKER_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


print(f"\n최종 Mapping : {MAPPING_OUTPUT}")
print(f"화자별 Summary: {SPEAKER_OUTPUT}")

print("\n완료!")