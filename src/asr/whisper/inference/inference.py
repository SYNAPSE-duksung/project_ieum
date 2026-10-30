import json
from pathlib import Path

import librosa
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizerFast,
)


BASE_DIR = Path(__file__).resolve().parent.parent / "whisper_small_general_v1"

MODEL_DIR = BASE_DIR / "model"
PROCESSOR_DIR = BASE_DIR / "processor"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_tokenizer = None
_feature_extractor = None


def load_model():
    global _model, _tokenizer, _feature_extractor

    if _model is not None:
        return

    _model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_DIR
    ).to(DEVICE)

    _model.eval()

    _tokenizer = WhisperTokenizerFast.from_pretrained(
        PROCESSOR_DIR
    )

    config_path = PROCESSOR_DIR / "processor_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        processor_config = json.load(f)

    _feature_extractor = WhisperFeatureExtractor(
        **processor_config["feature_extractor"]
    )


def transcribe(audio_path: str) -> str:
    load_model()

    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

    audio, sampling_rate = librosa.load(
        str(audio_path),
        sr=16000,
        mono=True,
    )

    if len(audio) == 0:
        raise ValueError("음성 파일이 비어 있습니다.")

    if len(audio) > 30 * sampling_rate:
        raise ValueError("현재 추론 코드는 30초 이하 음성만 지원합니다.")

    inputs = _feature_extractor(
        audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )

    input_features = inputs.input_features.to(DEVICE)

    with torch.inference_mode():
        predicted_ids = _model.generate(
            input_features,
            language="korean",
            task="transcribe",
        )

    text = _tokenizer.batch_decode(
        predicted_ids,
        skip_special_tokens=True,
    )[0]

    return text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("사용법: python inference.py sample.wav")
        sys.exit(1)

    result = transcribe(sys.argv[1])
    print("인식 결과:", result)