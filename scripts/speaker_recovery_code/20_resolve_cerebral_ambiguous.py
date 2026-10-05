from datasets import load_dataset, Audio
import pandas as pd
import io
import wave
import os
import re


# ============================================================
# 설정
# ============================================================

BEFORE_DATASET = "yoona-J/ASR_Degenerative_Brain_Dataset"

CANDIDATE_FILE = "results/cerebral_ambiguous_candidates.csv"
MAPPING_FILE = "results/cerebral_train_mapping.csv"

OUTPUT_FILE = "results/cerebral_train_mapping_resolved.csv"

MIN_DURATION = 1.0
MAX_DURATION = 30.0


# ============================================================
# pseudo-speaker 추출 함수
# ============================================================

def extract_speaker_info(filename):

    pattern = r"output_PN_([^-]+).*?-(F|M)-(\d+)-"

    match = re.search(
        pattern,
        str(filename),
        flags=re.IGNORECASE
    )

    if match is None:
        return None

    speaker_code = match.group(1).upper()
    sex = match.group(2).upper()
    age = int(match.group(3))

    return {
        "speaker_code": speaker_code,
        "sex": sex,
        "age": age,
        "pseudo_speaker_id":
            f"PN_{speaker_code}-{sex}-{age}"
    }


# ============================================================
# WAV bytes → duration 계산
# ============================================================

def get_wav_duration(audio_bytes):

    with wave.open(
        io.BytesIO(audio_bytes),
        "rb"
    ) as wav_file:

        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()

        duration = frames / float(sample_rate)

    return duration


# ============================================================
# 시작
# ============================================================

print(
    "========== Cerebral Ambiguous Duration 해결 ==========\n"
)


# ============================================================
# 1. 후보 CSV 불러오기
# ============================================================

print("[1/5] 후보 CSV 불러오는 중...")

candidate_df = pd.read_csv(
    CANDIDATE_FILE
)

mapping_df = pd.read_csv(
    MAPPING_FILE
)

print(
    f"  후보 개수     : {len(candidate_df):,}"
)

print(
    f"  Mapping rows : {len(mapping_df):,}"
)


# ============================================================
# 2. 원본 HF Train 연결
# ============================================================

print(
    "\n[2/5] 원본 Cerebral Train 데이터 연결 중..."
)

dataset = load_dataset(
    BEFORE_DATASET,
    split="train",
    streaming=True
)

# audio decoding은 하지 않고 bytes/path만 가져온다.
dataset = dataset.cast_column(
    "audio",
    Audio(decode=False)
)

print("  데이터 연결 완료")


# ============================================================
# 필요한 before_row만 추출
# ============================================================

target_rows = set(
    candidate_df["before_row"]
    .astype(int)
    .tolist()
)

print(
    f"  확인할 원본 row: "
    f"{sorted(target_rows)}"
)


# ============================================================
# 3. 후보 음성 duration 계산
# ============================================================

print(
    "\n[3/5] 후보 음성 길이 계산 중..."
)

duration_info = {}


for i, row in enumerate(dataset):

    if i in target_rows:

        audio = row["audio"]

        audio_bytes = audio["bytes"]

        if audio_bytes is None:

            raise ValueError(
                f"before_row {i}의 "
                "audio.bytes가 없습니다."
            )

        duration = get_wav_duration(
            audio_bytes
        )

        duration_info[i] = {
            "filename": audio["path"],
            "duration": duration
        }

        print(
            f"  row {i:4d} | "
            f"{duration:.3f}초 | "
            f"{audio['path']}"
        )

    # 가장 큰 후보 row까지 읽었으면 종료
    if (
        len(duration_info) == len(target_rows)
        and i >= max(target_rows)
    ):
        break


# 모든 후보를 찾았는지 확인
if len(duration_info) != len(target_rows):

    missing = (
        target_rows
        - set(duration_info.keys())
    )

    raise ValueError(
        f"찾지 못한 before_row: "
        f"{sorted(missing)}"
    )


# ============================================================
# 4. Duration filter로 후보 결정
# ============================================================

print(
    "\n[4/5] Duration filter 적용 중..."
)

resolved_rows = []


for processed_row, group in candidate_df.groupby(
    "processed_row"
):

    print("\n" + "=" * 70)

    transcript = (
        group.iloc[0][
            "processed_transcript"
        ]
    )

    print(
        f"Processed row : {int(processed_row)}"
    )

    print(
        f"Transcript    : {transcript}"
    )


    passed_candidates = []


    for _, candidate in group.iterrows():

        before_row = int(
            candidate["before_row"]
        )

        duration = duration_info[
            before_row
        ]["duration"]

        passed = (
            MIN_DURATION
            <= duration
            <= MAX_DURATION
        )

        print(
            f"  before_row {before_row}"
            f" | {duration:.3f}초"
            f" | {'PASS' if passed else 'REMOVE'}"
        )

        if passed:

            passed_candidates.append(
                before_row
            )


    # 정확히 한 후보만 통과해야 자동 확정
    if len(passed_candidates) != 1:

        raise ValueError(
            f"processed_row {processed_row}: "
            f"duration filter 통과 후보가 "
            f"{len(passed_candidates)}개입니다. "
            "자동 확정하지 않습니다."
        )


    selected_before_row = (
        passed_candidates[0]
    )


    selected = group[
        group["before_row"].astype(int)
        == selected_before_row
    ].iloc[0]


    filename = selected["filename"]


    speaker_info = extract_speaker_info(
        filename
    )


    if speaker_info is None:

        raise ValueError(
            f"Speaker 추출 실패: "
            f"{filename}"
        )


    print(
        f"\n  ✓ 최종 선택: "
        f"before_row {selected_before_row}"
    )

    print(
        f"    filename: {filename}"
    )

    print(
        f"    speaker : "
        f"{speaker_info['pseudo_speaker_id']}"
    )


    resolved_rows.append({
        "processed_row":
            int(processed_row),

        "before_row":
            selected_before_row,

        "filename":
            filename,

        "speaker_code":
            speaker_info["speaker_code"],

        "sex":
            speaker_info["sex"],

        "age":
            speaker_info["age"],

        "pseudo_speaker_id":
            speaker_info[
                "pseudo_speaker_id"
            ]
    })


# ============================================================
# Mapping CSV 수정
# ============================================================

print(
    "\n[5/5] 최종 Mapping 생성 중..."
)


for resolved in resolved_rows:

    processed_row = (
        resolved["processed_row"]
    )

    mask = (
        mapping_df["processed_row"]
        == processed_row
    )


    mapping_df.loc[
        mask,
        "before_row"
    ] = resolved["before_row"]


    mapping_df.loc[
        mask,
        "filename"
    ] = resolved["filename"]


    mapping_df.loc[
        mask,
        "speaker_code"
    ] = resolved["speaker_code"]


    mapping_df.loc[
        mask,
        "sex"
    ] = resolved["sex"]


    mapping_df.loc[
        mask,
        "age"
    ] = resolved["age"]


    mapping_df.loc[
        mask,
        "pseudo_speaker_id"
    ] = resolved[
        "pseudo_speaker_id"
    ]


    mapping_df.loc[
        mask,
        "mapping_status"
    ] = "RESOLVED_BY_DURATION_FILTER"


# ============================================================
# 최종 검증
# ============================================================

unresolved = mapping_df[
    mapping_df["mapping_status"]
    == "AMBIGUOUS"
]

failed = mapping_df[
    mapping_df["mapping_status"]
    == "FAILED"
]

missing_filename = (
    mapping_df["filename"]
    .isna()
    .sum()
)

missing_speaker = (
    mapping_df["pseudo_speaker_id"]
    .isna()
    .sum()
)


print("\n========== 최종 검증 ==========")

print(
    f"전체 rows          : "
    f"{len(mapping_df):,}"
)

print(
    f"RELIABLE           : "
    f"{(mapping_df['mapping_status'] == 'RELIABLE').sum():,}"
)

print(
    f"Duration 해결      : "
    f"{(mapping_df['mapping_status'] == 'RESOLVED_BY_DURATION_FILTER').sum():,}"
)

print(
    f"남은 AMBIGUOUS     : "
    f"{len(unresolved):,}"
)

print(
    f"FAILED             : "
    f"{len(failed):,}"
)

print(
    f"Filename 누락      : "
    f"{missing_filename:,}"
)

print(
    f"Speaker ID 누락    : "
    f"{missing_speaker:,}"
)

print(
    f"고유 pseudo-speaker: "
    f"{mapping_df['pseudo_speaker_id'].nunique():,}"
)


# ============================================================
# 안전성 검사
# ============================================================

if len(unresolved) != 0:
    raise ValueError(
        "아직 AMBIGUOUS row가 남아 있습니다."
    )

if len(failed) != 0:
    raise ValueError(
        "FAILED row가 남아 있습니다."
    )

if missing_filename != 0:
    raise ValueError(
        "Filename 누락이 존재합니다."
    )

if missing_speaker != 0:
    raise ValueError(
        "Speaker ID 누락이 존재합니다."
    )


# ============================================================
# 저장
# ============================================================

mapping_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"\n최종 Mapping 저장 완료:"
    f"\n{OUTPUT_FILE}"
)

print(
    "\nCerebral Train ambiguous 해결 완료!"
)