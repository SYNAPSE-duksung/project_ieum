import pandas as pd
import re
import os
import time


# ============================================================
# 설정
# ============================================================

AMBIGUITY_FILE = "results/stroke_train_ambiguity_check.csv"
BEFORE_FILE = "data/stroke_before_train.csv"

RESULT_DIR = "results"
RESULT_FILE = os.path.join(
    RESULT_DIR,
    "stroke_speaker_extraction_check.csv"
)

PROGRESS_INTERVAL = 1000


print("========== Stroke Speaker 정보 추출 검사 ==========\n")


# ============================================================
# 1. CSV 읽기
# ============================================================

print("[1/4] 로컬 CSV 읽는 중...")

ambiguity_df = pd.read_csv(AMBIGUITY_FILE)
before_df = pd.read_csv(BEFORE_FILE)

print(f"Processed rows : {len(ambiguity_df):,}")
print(f"Before rows    : {len(before_df):,}")
print("CSV 읽기 완료\n")


# ============================================================
# 2. filename 파싱 함수
# ============================================================

def extract_speaker_info(filename):

    if pd.isna(filename):
        return None

    filename = str(filename)

    # 예:
    # output_PN_BHR-01-03-F-52-KK_55.wav
    # output_PN_AMG-04-M-56-KK_12.wav
    #
    # 핵심:
    # PN_다음의 화자코드
    # -F-나이 또는 -M-나이
    #
    # 중간에 몇 개의 세션/과제 번호가 있는지는 신경 쓰지 않음.

    pattern = r"output_PN_([^-]+).*?-(F|M)-(\d+)-"

    match = re.search(
        pattern,
        filename,
        flags=re.IGNORECASE
    )

    if match is None:
        return None

    speaker_code = match.group(1).upper()
    sex = match.group(2).upper()
    age = int(match.group(3))

    pseudo_speaker_id = (
        f"PN_{speaker_code}-{sex}-{age}"
    )

    return {
        "speaker_code": speaker_code,
        "sex": sex,
        "age": age,
        "pseudo_speaker_id": pseudo_speaker_id
    }


# ============================================================
# 3. RELIABLE row 전체 검사
# ============================================================

print("[2/4] RELIABLE filename 파싱 중...\n")

reliable_df = ambiguity_df[
    ambiguity_df["status"] == "RELIABLE"
].copy()

results = []

success = 0
failed = 0

start_time = time.time()


for count, (_, row) in enumerate(
    reliable_df.iterrows(),
    start=1
):

    filename = row["filename"]

    info = extract_speaker_info(filename)

    if info is None:

        status = "FAILED"

        speaker_code = None
        sex = None
        age = None
        pseudo_speaker_id = None

        failed += 1

    else:

        status = "SUCCESS"

        speaker_code = info["speaker_code"]
        sex = info["sex"]
        age = info["age"]
        pseudo_speaker_id = info["pseudo_speaker_id"]

        success += 1


    results.append({
        "processed_row": row["processed_row"],
        "filename": filename,
        "transcript": row["transcript"],
        "speaker_code": speaker_code,
        "sex": sex,
        "age": age,
        "pseudo_speaker_id": pseudo_speaker_id,
        "extraction_status": status
    })


    if (
        count % PROGRESS_INTERVAL == 0
        or count == len(reliable_df)
    ):

        elapsed = time.time() - start_time

        print(
            f"  {count:,} / {len(reliable_df):,} "
            f"({count / len(reliable_df) * 100:.1f}%) "
            f"| SUCCESS {success:,} "
            f"| FAILED {failed:,} "
            f"| {elapsed:.2f}초"
        )


# ============================================================
# 4. 결과 분석
# ============================================================

print("\n[3/4] 추출 결과 분석 중...")

result_df = pd.DataFrame(results)

successful_df = result_df[
    result_df["extraction_status"] == "SUCCESS"
]

failed_df = result_df[
    result_df["extraction_status"] == "FAILED"
]


# pseudo-speaker별 utterance 수
speaker_counts = (
    successful_df["pseudo_speaker_id"]
    .value_counts()
)


print("\n========== Speaker 추출 결과 ==========")

print(f"RELIABLE rows        : {len(reliable_df):,}")
print(f"추출 성공            : {success:,}")
print(f"추출 실패            : {failed:,}")

if len(reliable_df) > 0:

    print(
        f"추출 성공률          : "
        f"{success / len(reliable_df) * 100:.2f}%"
    )

print(
    f"고유 pseudo-speaker : "
    f"{successful_df['pseudo_speaker_id'].nunique():,}"
)


# ============================================================
# 기본 sanity check
# ============================================================

print("\n========== 기본 검증 ==========")

print(
    "성별 값:",
    sorted(
        successful_df["sex"]
        .dropna()
        .unique()
        .tolist()
    )
)

print(
    f"최소 나이: "
    f"{successful_df['age'].min()}"
)

print(
    f"최대 나이: "
    f"{successful_df['age'].max()}"
)


# ============================================================
# pseudo-speaker별 데이터 수
# ============================================================

print("\n========== 발화 수가 많은 pseudo-speaker TOP 20 ==========")

for speaker_id, count in speaker_counts.head(20).items():

    print(
        f"{speaker_id:20s} | "
        f"{count:5d}개"
    )


# ============================================================
# 실패 파일 확인
# ============================================================

if failed > 0:

    print("\n========== 추출 실패 filename ==========")

    print(
        failed_df[
            ["processed_row", "filename"]
        ]
        .head(30)
        .to_string(index=False)
    )

else:

    print("\n추출 실패 filename 없음")


# ============================================================
# 결과 CSV 저장
# ============================================================

print("\n[4/4] 결과 저장 중...")

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

result_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"저장 위치: {RESULT_FILE}")

print("\n검사 완료!")