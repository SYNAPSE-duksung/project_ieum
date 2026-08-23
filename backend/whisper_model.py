import whisper
import torch

# GPU가 있으면 GPU, 없으면 CPU 사용
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("Whisper 모델 로딩 중...")
print(f"Device : {DEVICE}")

# 프로그램 시작 시 한 번만 모델 로딩
model = whisper.load_model(
    "small",
    device=DEVICE
)

print("Whisper 모델 로드 완료")
print("=" * 50)


def transcribe(audio_path: str) -> str:
    """
    음성 파일 하나를 받아
    Whisper 결과 문자열만 반환한다.
    """

    result = model.transcribe(
        audio_path,
        language="ko",
        task="transcribe",
        fp16=torch.cuda.is_available(),
        verbose=False
    )

    text = result.get("text", "")

    return text.strip()