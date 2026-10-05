from datasets import load_dataset, Audio
from transformers import WhisperTokenizer
import pandas as pd
import os
import re
import time


# ============================================================
# 설정
# ============================================================

# 전처리 이전 데이터
BEFORE_DATASET = "yoona-J/ASR_Degenerative_Brain_Dataset"

# Whisper 전처리 이후 비증강 데이터
AFTER_DATASET = "yoona-J/ASR_Preprocess_Degenerative_Brain_Dataset"

# 앞에서 확인한 실제 split별 행 개수
SPLITS = {
    "train": {
        "before_count": 4236,
        "after_count": 3226
    },
    "valid": {
        "before_count": 235,
        "after_count": 180
    },
    "test": {
        "before_count": 236,
        "after_count": 165
    }
}

DATA_DIR = "data"
RESULT_DIR = "results"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# 함수
# ============================================================

def normalize_text(text):
    """
    transcript 앞뒤 공백과 중복 공백을 정리한다.
    """
    return " ".join(str(text).strip().split())


def extract_speaker_info(filename):
    """
    원본 filename에서
    화자 코드 / 성별 / 나이를 추출하여
    pseudo-speaker ID를 만든다.

    예:
    output_PN_CUJ-02-03-F-36-kk_1.wav
        ↓
    speaker_code = CUJ
    sex = F
    age = 36
        ↓
    PN_CUJ-F-36
    """

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


def valid_csv(path, expected_count):
    """
    기존 CSV가 존재하고,
    행 수도 예상값과 정확히 같으면 재사용한다.

    이렇게 하면 프로그램이 중간에 종료되어도
    이미 완료된 Before/After 데이터를 다시
    Hugging Face에서 읽지 않아도 된다.
    """

    if not os.path.exists(path):
        return False

    try:
        df = pd.read_csv(path)

        if len(df) == expected_count:
            return True

        print(
            f"  기존 CSV 행 수 불일치: "
            f"{len(df):,} != {expected_count:,}"
        )

        return False

    except Exception:
        return False


# ============================================================
# 시작
# ============================================================

print("========== Cerebral Base 전체 Mapping ==========\n")

print("[준비] Hugging Face 데이터셋 연결 중...")


# ============================================================
# 전처리 이전 데이터 연결
# ============================================================

before = load_dataset(
    BEFORE_DATASET,
    streaming=True
)

# 실제 audio decoding을 하지 않는다.
# 여기서는 audio.path만 필요하기 때문이다.
before = before.cast_column(
    "audio",
    Audio(decode=False)
)


# ============================================================
# 전처리 이후 데이터 연결
# ============================================================

after = load_dataset(
    AFTER_DATASET,
    streaming=True
)


# ============================================================
# Whisper tokenizer
# ============================================================

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Korean",
    task="transcribe",
    clean_up_tokenization_spaces=False
)

print("데이터셋 연결 완료\n")


all_summary = []


# ============================================================
# split별 처리
# ============================================================

for split_name, config in SPLITS.items():

    expected_before = config["before_count"]
    expected_after = config["after_count"]

    print("\n" + "=" * 65)
    print(f"{split_name.upper()} 처리 시작")
    print("=" * 65)


    # ========================================================
    # Cerebral 전용 CSV 파일명
    # ========================================================

    before_csv = os.path.join(
        DATA_DIR,
        f"cerebral_before_{split_name}.csv"
    )

    after_csv = os.path.join(
        DATA_DIR,
        f"cerebral_after_{split_name}.csv"
    )


    # ========================================================
    # 1. Before metadata
    # ========================================================

    print(
        f"\n[1/5] {split_name} Before metadata 확인..."
    )


    # --------------------------------------------------------
    # 기존 CSV가 있으면 재사용
    # --------------------------------------------------------

    if valid_csv(
        before_csv,
        expected_before
    ):

        print(
            "  ✓ 기존 CSV 발견 → 재사용"
        )

        before_df = pd.read_csv(
            before_csv
        )

        print(
            f"  {len(before_df):,}개 로드 완료"
        )


    # --------------------------------------------------------
    # 없으면 Hugging Face에서 읽기
    # --------------------------------------------------------

    else:

        print(
            "  기존 완성 CSV 없음 → "
            "Hugging Face에서 읽기 시작"
        )

        before_rows = []

        start = time.time()

        for i, row in enumerate(
            before[split_name]
        ):

            before_rows.append({
                "before_row": i,
                "filename":
                    row["audio"]["path"],
                "transcript":
                    normalize_text(
                        row["transcripts"]
                    )
            })

            current = i + 1

            # 1,000개마다 진행 상황 출력
            if (
                current % 1000 == 0
                or current == expected_before
            ):

                elapsed = (
                    time.time() - start
                )

                print(
                    f"  Before "
                    f"{current:,} / "
                    f"{expected_before:,} "
                    f"({current / expected_before * 100:.1f}%) "
                    f"| {elapsed:.1f}초"
                )

            if current >= expected_before:
                break


        before_df = pd.DataFrame(
            before_rows
        )


        # 예상 개수와 실제 개수가 다른 경우 중단
        if len(before_df) != expected_before:

            raise ValueError(
                f"{split_name} Before 개수 불일치: "
                f"{len(before_df):,} != "
                f"{expected_before:,}"
            )


        # 로컬 CSV 저장
        before_df.to_csv(
            before_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"  ✓ Before CSV 저장 완료: "
            f"{before_csv}"
        )


    # ========================================================
    # 2. After metadata
    # ========================================================

    print(
        f"\n[2/5] {split_name} After metadata 확인..."
    )


    # --------------------------------------------------------
    # 기존 CSV가 있으면 재사용
    # --------------------------------------------------------

    if valid_csv(
        after_csv,
        expected_after
    ):

        print(
            "  ✓ 기존 CSV 발견 → 재사용"
        )

        after_df = pd.read_csv(
            after_csv
        )

        print(
            f"  {len(after_df):,}개 로드 완료"
        )


    # --------------------------------------------------------
    # 없으면 Hugging Face에서 읽기
    # --------------------------------------------------------

    else:

        print(
            "  기존 완성 CSV 없음 → "
            "Hugging Face에서 읽기 시작"
        )

        after_rows = []

        start = time.time()

        for i, row in enumerate(
            after[split_name]
        ):

            # Whisper token labels를 다시 text로 복원
            decoded = tokenizer.decode(
                row["labels"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            after_rows.append({
                "processed_row": i,
                "transcript":
                    normalize_text(decoded)
            })

            current = i + 1


            # 1,000개마다 진행 상황 출력
            if (
                current % 1000 == 0
                or current == expected_after
            ):

                elapsed = (
                    time.time() - start
                )

                print(
                    f"  After "
                    f"{current:,} / "
                    f"{expected_after:,} "
                    f"({current / expected_after * 100:.1f}%) "
                    f"| {elapsed:.1f}초"
                )

            if current >= expected_after:
                break


        after_df = pd.DataFrame(
            after_rows
        )


        # 예상 개수 검증
        if len(after_df) != expected_after:

            raise ValueError(
                f"{split_name} After 개수 불일치: "
                f"{len(after_df):,} != "
                f"{expected_after:,}"
            )


        # 로컬 CSV 저장
        after_df.to_csv(
            after_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"  ✓ After CSV 저장 완료: "
            f"{after_csv}"
        )


    # ========================================================
    # 3. 개수 확인
    # ========================================================

    print(
        f"\n[3/5] {split_name} 개수 검증..."
    )

    print(
        f"  Before  : {len(before_df):,}"
    )

    print(
        f"  After   : {len(after_df):,}"
    )

    print(
        f"  Filtered: "
        f"{len(before_df) - len(after_df):,}"
    )


    # ========================================================
    # 4. Forward / Backward Mapping
    # ========================================================

    print(
        f"\n[4/5] {split_name} "
        "Forward / Backward mapping..."
    )


    # Before transcript sequence
    B = (
        before_df["transcript"]
        .fillna("")
        .astype(str)
        .tolist()
    )


    # After transcript sequence
    A = (
        after_df["transcript"]
        .fillna("")
        .astype(str)
        .tolist()
    )


    # ========================================================
    # Forward mapping
    # ========================================================

    forward = [-1] * len(A)

    b = 0

    for a in range(len(A)):

        while b < len(B):

            if A[a] == B[b]:

                forward[a] = b

                b += 1

                break

            b += 1


    # ========================================================
    # Backward mapping
    # ========================================================

    backward = [-1] * len(A)

    b = len(B) - 1

    for a in range(
        len(A) - 1,
        -1,
        -1
    ):

        while b >= 0:

            if A[a] == B[b]:

                backward[a] = b

                b -= 1

                break

            b -= 1


    # ========================================================
    # Mapping 결과 생성
    # ========================================================

    mapping_rows = []

    reliable = 0
    ambiguous = 0
    failed = 0
    speaker_failed = 0


    for i in range(len(A)):

        f_row = forward[i]
        b_row = backward[i]


        # ----------------------------------------------------
        # 어디에도 연결되지 않은 경우
        # ----------------------------------------------------

        if (
            f_row == -1
            or b_row == -1
        ):

            status = "FAILED"

            failed += 1

            before_row = None
            filename = None
            speaker_code = None
            sex = None
            age = None
            pseudo_id = None


        # ----------------------------------------------------
        # Forward와 Backward가 서로 다른 원본을 가리키는 경우
        # ----------------------------------------------------

        elif f_row != b_row:

            status = "AMBIGUOUS"

            ambiguous += 1

            before_row = None
            filename = None
            speaker_code = None
            sex = None
            age = None
            pseudo_id = None


        # ----------------------------------------------------
        # 유일하게 연결되는 경우
        # ----------------------------------------------------

        else:

            status = "RELIABLE"

            reliable += 1

            before_row = f_row

            filename = before_df.iloc[
                before_row
            ]["filename"]


            # filename에서 pseudo-speaker 추출
            info = extract_speaker_info(
                filename
            )


            if info is None:

                speaker_failed += 1

                speaker_code = None
                sex = None
                age = None
                pseudo_id = None


            else:

                speaker_code = (
                    info["speaker_code"]
                )

                sex = info["sex"]

                age = info["age"]

                pseudo_id = (
                    info[
                        "pseudo_speaker_id"
                    ]
                )


        mapping_rows.append({
            "processed_row": i,
            "before_row": before_row,
            "forward_before_row": f_row,
            "backward_before_row": b_row,
            "filename": filename,
            "transcript": A[i],
            "speaker_code": speaker_code,
            "sex": sex,
            "age": age,
            "pseudo_speaker_id":
                pseudo_id,
            "mapping_status": status
        })


    mapping_df = pd.DataFrame(
        mapping_rows
    )


    # ========================================================
    # 5. 결과 저장
    # ========================================================

    print(
        f"\n[5/5] {split_name} 결과 저장..."
    )


    mapping_file = os.path.join(
        RESULT_DIR,
        f"cerebral_{split_name}_mapping.csv"
    )


    mapping_df.to_csv(
        mapping_file,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # Reliable 데이터만 추출
    # ========================================================

    reliable_df = mapping_df[
        mapping_df[
            "mapping_status"
        ] == "RELIABLE"
    ]


    # 고유 pseudo-speaker 수
    unique_speakers = (
        reliable_df[
            "pseudo_speaker_id"
        ]
        .dropna()
        .nunique()
    )


    # ========================================================
    # Split 결과 출력
    # ========================================================

    print(
        f"\n========== "
        f"{split_name.upper()} 결과 "
        f"=========="
    )

    print(
        f"Before rows      : {len(B):,}"
    )

    print(
        f"Processed rows   : {len(A):,}"
    )

    print(
        f"Reliable         : {reliable:,}"
    )

    print(
        f"Ambiguous        : {ambiguous:,}"
    )

    print(
        f"Failed           : {failed:,}"
    )

    print(
        f"Speaker 실패     : {speaker_failed:,}"
    )

    print(
        f"Pseudo-speakers  : "
        f"{unique_speakers:,}"
    )

    print(
        f"Reliable rate    : "
        f"{reliable / len(A) * 100:.2f}%"
    )

    print(
        f"Filename 중복    : "
        f"{reliable_df['filename'].duplicated().sum():,}"
    )

    print(
        f"저장 위치        : {mapping_file}"
    )


    # ========================================================
    # 문제 row 출력
    # ========================================================

    problem_df = mapping_df[
        mapping_df[
            "mapping_status"
        ] != "RELIABLE"
    ]


    if len(problem_df) > 0:

        print(
            "\n========== 문제 row =========="
        )

        print(
            problem_df[
                [
                    "processed_row",
                    "transcript",
                    "forward_before_row",
                    "backward_before_row",
                    "mapping_status"
                ]
            ]
            .to_string(index=False)
        )


    # ========================================================
    # Summary 기록
    # ========================================================

    all_summary.append({
        "split": split_name,
        "before": len(B),
        "after": len(A),
        "filtered":
            len(B) - len(A),
        "reliable": reliable,
        "ambiguous": ambiguous,
        "failed": failed,
        "speaker_failed":
            speaker_failed,
        "pseudo_speakers":
            unique_speakers
    })


# ============================================================
# 전체 Summary
# ============================================================

print("\n" + "=" * 65)
print("CEREBRAL BASE 최종 SUMMARY")
print("=" * 65)


summary_df = pd.DataFrame(
    all_summary
)


print(
    summary_df.to_string(
        index=False
    )
)


summary_file = os.path.join(
    RESULT_DIR,
    "cerebral_mapping_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


print(
    f"\nSummary 저장 위치: "
    f"{summary_file}"
)

print("\n전체 처리 완료!")