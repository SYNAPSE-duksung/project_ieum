
from pathlib import Path
from math import gcd

import torch
import soundfile as sf
from scipy.signal import resample_poly
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# 학습된 범용 ASR 모델 경로
MODEL_ROOT = Path.home() / "IEUM_models" / "whisper_small_general_v1"
MODEL_DIR = MODEL_ROOT / "model"
PROCESSOR_DIR = MODEL_ROOT / "processor"

# 우선 CPU로 실행
DEVICE = "cpu"

# ASR 모델 입력 샘플레이트
TARGET_SAMPLE_RATE = 16000

print("=" * 50)
print("IEUM 범용 ASR 모델 로딩 중...")

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

print("IEUM 범용 ASR 모델 로딩 완료!")
print("=" * 50)


def transcribe(audio_path: str) -> str:
    """
    음성 파일을 받아 범용 ASR로 인식한 문자열을 반환한다.

    전처리:
    1. 음성 파일 읽기
    2. 스테레오 → 모노 변환
    3. 샘플레이트 → 16,000Hz 자동 리샘플링
    """

    # 1. 음성 파일 읽기
    audio, sample_rate = sf.read(audio_path)

    print(f"[Audio] 입력 파일: {audio_path}")
    print(f"[Audio] 원본 샘플레이트: {sample_rate}Hz")

    # 2. 스테레오 → 모노
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        print("[Audio] 스테레오 → 모노 변환 완료")

    # 3. 자동 리샘플링
    if sample_rate != TARGET_SAMPLE_RATE:
        print(
            f"[Audio] 리샘플링: "
            f"{sample_rate}Hz → {TARGET_SAMPLE_RATE}Hz"
        )

        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)

        audio = resample_poly(
            audio,
            up=TARGET_SAMPLE_RATE // divisor,
            down=sample_rate // divisor
        )

        sample_rate = TARGET_SAMPLE_RATE

        print("[Audio] 리샘플링 완료")
    else:
        print("[Audio] 16,000Hz 입력 - 리샘플링 생략")

    # 4. Whisper 입력 데이터 생성
    inputs = processor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    input_features = inputs.input_features.to(DEVICE)

    # 5. 범용 ASR 음성 인식
    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            language="korean",
            task="transcribe",
            max_new_tokens=128
        )

    # 6. 인식 결과 문자열 변환
    result = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0]

    return result.strip()
