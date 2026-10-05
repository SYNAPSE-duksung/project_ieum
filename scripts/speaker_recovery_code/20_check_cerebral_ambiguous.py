import pandas as pd
import os


# ============================================================
# 설정
# ============================================================

BEFORE_FILE = "data/cerebral_before_train.csv"
MAPPING_FILE = "results/cerebral_train_mapping.csv"

RESULT_DIR = "results"

OUTPUT_FILE = os.path.join(
    RESULT_DIR,
    "cerebral_ambiguous_candidates.csv"
)


# ============================================================
# 파일 존재 여부 확인
# ============================================================

print("========== Cerebral Ambiguous 후보 확인 ==========\n")

if not os.path.exists(BEFORE_FILE):
    raise FileNotFoundError(
        f"파일을 찾을 수 없습니다: {BEFORE_FILE}"
    )

if not os.path.exists(MAPPING_FILE):
    raise FileNotFoundError(
        f"파일을 찾을 수 없습니다: {MAPPING_FILE}"
    )


# ============================================================
# CSV 불러오기
# ============================================================

print("[1/3] CSV 불러오는 중...")

before_df = pd.read_csv(BEFORE_FILE)
mapping_df = pd.read_csv(MAPPING_FILE)

print(
    f"  Before rows  : {len(before_df):,}"
)

print(
    f"  Mapping rows : {len(mapping_df):,}"
)


# ============================================================
# AMBIGUOUS 행 찾기
# ============================================================

print("\n[2/3] AMBIGUOUS 행 확인 중...")

ambiguous_df = mapping_df[
    mapping_df["mapping_status"] == "AMBIGUOUS"
].copy()

print(
    f"  AMBIGUOUS 개수: {len(ambiguous_df):,}"
)

if len(ambiguous_df) == 0:

    print("\nAMBIGUOUS 행이 없습니다.")
    raise SystemExit


# ============================================================
# 각 AMBIGUOUS의 후보 확인
# ============================================================

print("\n[3/3] 후보 원본 row 확인 중...\n")

candidate_rows = []


for _, row in ambiguous_df.iterrows():

    processed_row = int(
        row["processed_row"]
    )

    transcript = row["transcript"]

    forward_row = int(
        row["forward_before_row"]
    )

    backward_row = int(
        row["backward_before_row"]
    )


    print("=" * 70)

    print(
        f"Processed row : {processed_row}"
    )

    print(
        f"Transcript    : {transcript}"
    )

    print(
        f"Forward row   : {forward_row}"
    )

    print(
        f"Backward row  : {backward_row}"
    )


    # ========================================================
    # Forward 후보
    # ========================================================

    forward_data = before_df.iloc[
        forward_row
    ]

    print("\n[Forward 후보]")

    print(
        f"before_row : {forward_row}"
    )

    print(
        f"filename   : "
        f"{forward_data['filename']}"
    )

    print(
        f"transcript : "
        f"{forward_data['transcript']}"
    )


    candidate_rows.append({
        "processed_row":
            processed_row,

        "processed_transcript":
            transcript,

        "candidate_type":
            "forward",

        "before_row":
            forward_row,

        "filename":
            forward_data["filename"],

        "before_transcript":
            forward_data["transcript"]
    })


    # ========================================================
    # Backward 후보
    # ========================================================

    backward_data = before_df.iloc[
        backward_row
    ]

    print("\n[Backward 후보]")

    print(
        f"before_row : {backward_row}"
    )

    print(
        f"filename   : "
        f"{backward_data['filename']}"
    )

    print(
        f"transcript : "
        f"{backward_data['transcript']}"
    )


    candidate_rows.append({
        "processed_row":
            processed_row,

        "processed_transcript":
            transcript,

        "candidate_type":
            "backward",

        "before_row":
            backward_row,

        "filename":
            backward_data["filename"],

        "before_transcript":
            backward_data["transcript"]
    })


    # ========================================================
    # Forward~Backward 사이의 전체 원본 행도 출력
    # ========================================================

    print(
        "\n[Forward ~ Backward 사이 원본 데이터]"
    )

    start_row = min(
        forward_row,
        backward_row
    )

    end_row = max(
        forward_row,
        backward_row
    )

    nearby_df = before_df.iloc[
        start_row:end_row + 1
    ][
        [
            "before_row",
            "filename",
            "transcript"
        ]
    ]

    print(
        nearby_df.to_string(
            index=False
        )
    )

    print()


# ============================================================
# 후보 CSV 저장
# ============================================================

candidate_df = pd.DataFrame(
    candidate_rows
)

candidate_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 최종 출력
# ============================================================

print("=" * 70)

print(
    f"\n후보 결과 저장 완료: {OUTPUT_FILE}"
)

print("\n========== 후보 요약 ==========")

print(
    candidate_df[
        [
            "processed_row",
            "processed_transcript",
            "candidate_type",
            "before_row",
            "filename"
        ]
    ].to_string(index=False)
)

print("\n확인 완료!")