from datasets import load_dataset, Audio
from transformers import WhisperTokenizer
from itertools import islice


# --------------------------------------------------
# 1. Stroke 데이터셋 불러오기
# --------------------------------------------------

print("데이터셋 불러오는 중...")

before = load_dataset(
    "yoona-J/ASR_Stroke_Dataset",
    streaming=True
)

after = load_dataset(
    "yoona-J/ASR_Preprocess_Stroke_Dataset",
    streaming=True
)


# --------------------------------------------------
# 2. audio 실제 decoding 끄기
# --------------------------------------------------
# 우리는 음성 자체가 아니라 audio.path(파일명)만 필요함
before = before.cast_column(
    "audio",
    Audio(decode=False)
)


# --------------------------------------------------
# 3. Whisper-small tokenizer 불러오기
# --------------------------------------------------

print("Whisper tokenizer 불러오는 중...")

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small",
    language="Korean",
    task="transcribe"
)


# --------------------------------------------------
# 4. Train 데이터 첫 10개 가져오기
# --------------------------------------------------

before_rows = list(islice(before["train"], 10))
after_rows = list(islice(after["train"], 10))


print("\n========== 첫 10개 비교 ==========\n")


# --------------------------------------------------
# 5. 전처리 전/후 비교
# --------------------------------------------------

for i, (b, a) in enumerate(zip(before_rows, after_rows)):

    # 실제 음성을 decode하지 않고 파일 경로만 가져옴
    filename = b["audio"]["path"]

    # 전처리 전 정답 문장
    transcript = b["transcripts"]

    # 전처리 후 labels → 문자열
    decoded = tokenizer.decode(
        a["labels"],
        skip_special_tokens=True
    ).strip()

    # 앞뒤 공백 제거 후 비교
    is_match = transcript.strip() == decoded

    print(f"[{i}]")
    print(f"파일명          : {filename}")
    print(f"원본 transcript : {transcript}")
    print(f"labels decode    : {decoded}")
    print(f"일치 여부        : {'MATCH' if is_match else 'DIFFERENT'}")
    print("-" * 60)