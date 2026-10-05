from datasets import load_dataset

print("=== 1. 전처리 전 Stroke Dataset ===")

before = load_dataset(
    "yoona-J/ASR_Stroke_Dataset",
    streaming=True
)

print(before)

print("\n=== 2. 전처리 후 Stroke Dataset ===")

after = load_dataset(
    "yoona-J/ASR_Preprocess_Stroke_Dataset",
    streaming=True
)

print(after)

print("\n데이터셋 접근 성공!")