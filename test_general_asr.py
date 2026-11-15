
from pathlib import Path
import time

import torch
import soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 범용 ASR 모델이 저장된 위치
MODEL_ROOT = Path.home() / "IEUM_models" / "whisper_small_general_v1"
MODEL_DIR = MODEL_ROOT / "model"
PROCESSOR_DIR = MODEL_ROOT / "processor"

# 테스트할 음성 파일
AUDIO_PATH = Path("sample_audio/sample1.wav")

# 우선 호환성을 확인하기 위해 CPU 사용
DEVICE = "cpu"

print("범용 ASR 모델 로딩 중...")
start = time.time()

processor = WhisperProcessor.from_pretrained(
    str(PROCESSOR_DIR),
    local_files_only=True
)

model = WhisperForConditionalGeneration.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)
model.to(DEVICE)
model.eval()

print(f"모델 로딩 완료! ({time.time() - start:.2f}초)")

# WAV 파일 읽기
audio, sample_rate = sf.read(str(AUDIO_PATH))

if audio.ndim > 1:
    audio = audio.mean(axis=1)

if sample_rate != 16000:
    raise ValueError(
        f"현재 샘플레이트는 {sample_rate}Hz입니다. "
        "16,000Hz WAV 파일이 필요합니다."
    )

# 음성 데이터를 모델 입력 형식으로 변환
inputs = processor(
    audio,
    sampling_rate=sample_rate,
    return_tensors="pt"
)

input_features = inputs.input_features.to(DEVICE)

print("음성 인식 중...")
start = time.time()

with torch.inference_mode():
    predicted_ids = model.generate(
        input_features,
        language="korean",
        task="transcribe",
        max_new_tokens=128
    )

result = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]

print("=" * 50)
print("범용 ASR 인식 결과:", result.strip())
print(f"음성 인식 소요 시간: {time.time() - start:.2f}초")
print("=" * 50)
