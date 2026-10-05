import pandas as pd
import os
import time


# ============================================================
# 설정
# ============================================================

BEFORE_FILE = "data/stroke_before_train.csv"
AFTER_FILE = "data/stroke_after_train.csv"

RESULT_DIR = "results"
RESULT_FILE = os.path.join(
    RESULT_DIR,
    "stroke_train_ambiguity_check.csv"
)

PROGRESS_INTERVAL = 1000


print("========== Stroke Train Ambiguity 검사 ==========\n")


# ============================================================
# 1. 로컬 CSV 읽기
# ============================================================

print("[1/4] 로컬 CSV 읽는 중...")

before_df = pd.read_csv(BEFORE_FILE)
after_df = pd.read_csv(AFTER_FILE)

before_texts = (
    before_df["normalized_transcript"]
    .fillna("")
    .astype(str)
    .tolist()
)

after_texts = (
    after_df["normalized_transcript"]
    .fillna("")
    .astype(str)
    .tolist()
)

print(f"Before rows : {len(before_texts):,}")
print(f"After rows  : {len(after_texts):,}")
print("CSV 읽기 완료\n")


# ============================================================
# 2. 앞 → 뒤 Greedy Mapping
# ============================================================

print("[2/4] Forward mapping 검사 중...")

forward_mapping = [-1] * len(after_texts)

before_index = 0
start_time = time.time()


for after_index, target in enumerate(after_texts):

    while before_index < len(before_texts):

        if before_texts[before_index] == target:

            forward_mapping[after_index] = before_index
            before_index += 1
            break

        before_index += 1

    current = after_index + 1

    if (
        current % PROGRESS_INTERVAL == 0
        or current == len(after_texts)
    ):
        elapsed = time.time() - start_time

        print(
            f"  Forward: {current:,} / {len(after_texts):,} "
            f"({current / len(after_texts) * 100:.1f}%) "
            f"| {elapsed:.2f}초"
        )


forward_failed = sum(
    row == -1
    for row in forward_mapping
)

print(
    f"Forward 완료 | Failed: {forward_failed:,}\n"
)


# ============================================================
# 3. 뒤 → 앞 Greedy Mapping
# ============================================================

print("[3/4] Backward mapping 검사 중...")

backward_mapping = [-1] * len(after_texts)

before_index = len(before_texts) - 1
start_time = time.time()


for after_index in range(
    len(after_texts) - 1,
    -1,
    -1
):

    target = after_texts[after_index]

    while before_index >= 0:

        if before_texts[before_index] == target:

            backward_mapping[after_index] = before_index
            before_index -= 1
            break

        before_index -= 1

    processed = len(after_texts) - after_index

    if (
        processed % PROGRESS_INTERVAL == 0
        or processed == len(after_texts)
    ):
        elapsed = time.time() - start_time

        print(
            f"  Backward: {processed:,} / {len(after_texts):,} "
            f"({processed / len(after_texts) * 100:.1f}%) "
            f"| {elapsed:.2f}초"
        )


backward_failed = sum(
    row == -1
    for row in backward_mapping
)

print(
    f"Backward 완료 | Failed: {backward_failed:,}\n"
)


# ============================================================
# 4. Forward / Backward 비교
# ============================================================

print("[4/4] 매핑 결과 비교 중...")

results = []

reliable = 0
ambiguous = 0
failed = 0


for i in range(len(after_texts)):

    forward_row = forward_mapping[i]
    backward_row = backward_mapping[i]

    if forward_row == -1 or backward_row == -1:

        status = "FAILED"
        failed += 1

    elif forward_row == backward_row:

        status = "RELIABLE"
        reliable += 1

    else:

        status = "AMBIGUOUS"
        ambiguous += 1


    # RELIABLE인 경우에만 filename 확정
    if status == "RELIABLE":

        filename = before_df.iloc[forward_row]["filename"]

    else:

        filename = None


    results.append({
        "processed_row": i,
        "transcript": after_texts[i],
        "forward_before_row": forward_row,
        "backward_before_row": backward_row,
        "status": status,
        "filename": filename
    })


# ============================================================
# 결과 저장
# ============================================================

os.makedirs(RESULT_DIR, exist_ok=True)

result_df = pd.DataFrame(results)

result_df.to_csv(
    RESULT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 최종 결과
# ============================================================

total = len(result_df)

print("\n========== Ambiguity 검사 결과 ==========")

print(f"Processed rows : {total:,}")
print(f"Reliable       : {reliable:,}")
print(f"Ambiguous      : {ambiguous:,}")
print(f"Failed         : {failed:,}")

print(
    f"Reliable rate  : "
    f"{reliable / total * 100:.2f}%"
)

print(
    f"Ambiguous rate : "
    f"{ambiguous / total * 100:.2f}%"
)

print(f"\n결과 저장 위치: {RESULT_FILE}")


# ============================================================
# Ambiguous 예시 출력
# ============================================================

ambiguous_df = result_df[
    result_df["status"] == "AMBIGUOUS"
]

if len(ambiguous_df) > 0:

    print("\n========== AMBIGUOUS 예시 10개 ==========")

    print(
        ambiguous_df[
            [
                "processed_row",
                "transcript",
                "forward_before_row",
                "backward_before_row"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

else:

    print("\nAMBIGUOUS row 없음")


print("\n검사 완료!")