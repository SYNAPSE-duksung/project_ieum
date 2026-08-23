from __future__ import annotations

import traceback

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from src.api.model_loader import ModelLoader
from src.asr.inference import ASRInference


# ============================================================
# Path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "best_model.pt"
)

VOCAB_PATH = (
    PROJECT_ROOT
    / "models"
    / "final"
    / "vocab.json"
)


# ============================================================
# Global objects
# ============================================================

model_loader = ModelLoader(
    checkpoint_path=CHECKPOINT_PATH,
    vocab_path=VOCAB_PATH,
)

asr_inference: ASRInference | None = None


# ============================================================
# Utility
# ============================================================

ALLOWED_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
)


def validate_audio_filename(
    filename: str | None,
) -> str:
    """
    업로드된 음성 파일명을 검증한다.
    """

    if filename is None:
        raise HTTPException(
            status_code=400,
            detail="Audio filename is missing.",
        )

    if not filename.lower().endswith(
        ALLOWED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format.",
        )

    return filename


async def read_audio(
    audio: UploadFile,
) -> bytes:
    """
    UploadFile을 읽고 빈 파일인지 검사한다.
    """

    validate_audio_filename(
        audio.filename
    )

    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file.",
        )

    return audio_bytes


def get_inference() -> ASRInference:
    """
    ASRInference 준비 여부를 확인한다.
    """

    if asr_inference is None:
        raise HTTPException(
            status_code=503,
            detail="ASR model is not ready.",
        )

    return asr_inference


# ============================================================
# FastAPI lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 서버 시작 시
    범용 ASR 모델을 한 번만 로드한다.

    개인화 모델은 요청이 들어올 때
    필요한 화자 모델만 로드한다.
    """

    global asr_inference

    print()
    print("=" * 70)
    print("IEUM ASR API 시작")
    print("=" * 70)

    # --------------------------------------------------------
    # General model
    # --------------------------------------------------------

    model_loader.load()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    asr_inference = ASRInference(
        model_loader=model_loader
    )

    print()
    print("ASR inference 준비 완료")

    print(
        "지원 개인화 화자:",
        model_loader
        .get_supported_personalized_speakers(),
    )

    print("=" * 70)

    yield

    print()
    print("=" * 70)
    print("IEUM ASR API 종료")
    print("=" * 70)


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="IEUM ASR API",
    description=(
        "구음장애 범용 및 개인화 "
        "음성인식 모델 추론을 위한 FastAPI 서버"
    ),
    version="0.3.0",
    lifespan=lifespan,
)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "service": "IEUM ASR API",
        "status": "running",
        "general_model": (
            "Whisper Small Last4 + BiGRU CTC"
        ),
        "personalization": True,
        "supported_speakers": (
            model_loader
            .get_supported_personalized_speakers()
        ),
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "general_model_loaded": (
            model_loader.is_loaded
        ),
        "personalized_speaker_loaded": (
            model_loader
            .loaded_personalized_speaker
        ),
        "supported_speakers": (
            model_loader
            .get_supported_personalized_speakers()
        ),
    }


# ============================================================
# General ASR
# ============================================================

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
):
    """
    범용 구음장애 ASR 추론.

    기존 Raspberry Pi 클라이언트와의
    호환성을 위해 유지한다.
    """

    audio_bytes = await read_audio(
        audio
    )

    inference = get_inference()

    try:

        text = inference.transcribe(
            audio_bytes=audio_bytes,
            filename=audio.filename,
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("범용 ASR 추론 중 오류 발생")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(
            status_code=500,
            detail=(
                "General ASR inference failed: "
                f"{type(error).__name__}: {error}"
            ),
        )

    return {
        "filename": audio.filename,
        "text": text,
    }


# ============================================================
# General vs Personalized
# ============================================================

@app.post("/transcribe/compare")
async def transcribe_compare(
    speaker_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    동일 음성을 범용 모델과
    선택된 화자의 개인화 모델에 입력하여
    두 결과를 동시에 반환한다.

    지원 화자
    ---------
    HYH_M_22
    SKY_M_24
    """

    # --------------------------------------------------------
    # Speaker validation
    # --------------------------------------------------------

    supported_speakers = (
        model_loader
        .get_supported_personalized_speakers()
    )

    if speaker_id not in supported_speakers:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Unsupported personalized speaker."
                ),
                "requested_speaker": (
                    speaker_id
                ),
                "supported_speakers": (
                    supported_speakers
                ),
            },
        )

    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    audio_bytes = await read_audio(
        audio
    )

    inference = get_inference()

    # --------------------------------------------------------
    # General + Personalized inference
    # --------------------------------------------------------

    try:

        result = (
            inference.transcribe_compare(
                audio_bytes=audio_bytes,
                speaker_id=speaker_id,
                filename=audio.filename,
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        print()
        print("=" * 70)
        print(
            "범용 / 개인화 비교 추론 중 오류 발생"
        )
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(
            status_code=500,
            detail=(
                "ASR comparison inference failed: "
                f"{type(error).__name__}: {error}"
            ),
        )

    return result


# ============================================================
# Personalized only
# ============================================================

@app.post("/transcribe/personalized")
async def transcribe_personalized(
    speaker_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    선택된 화자의 개인화 모델만 사용하여 추론한다.
    """

    supported_speakers = (
        model_loader
        .get_supported_personalized_speakers()
    )

    if speaker_id not in supported_speakers:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Unsupported personalized speaker."
                ),
                "requested_speaker": (
                    speaker_id
                ),
                "supported_speakers": (
                    supported_speakers
                ),
            },
        )

    audio_bytes = await read_audio(
        audio
    )

    inference = get_inference()

    try:

        text = (
            inference.transcribe_personalized(
                audio_bytes=audio_bytes,
                speaker_id=speaker_id,
                filename=audio.filename,
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("개인화 ASR 추론 중 오류 발생")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(
            status_code=500,
            detail=(
                "Personalized ASR inference failed: "
                f"{type(error).__name__}: {error}"
            ),
        )

    return {
        "speaker_id": speaker_id,
        "filename": audio.filename,
        "text": text,
    }