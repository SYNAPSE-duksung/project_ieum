from __future__ import annotations

import traceback

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.model_loader import ModelLoader
from src.asr.inference import ASRInference
from src.tts.piper_tts import synthesize_text_to_speech


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
# TTS Request Model
# ============================================================

class TTSRequest(BaseModel):
    text: str


# ============================================================
# FastAPI lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 서버 시작 시 최종 ASR 모델을 한 번만 로드한다.
    """

    global asr_inference

    print()
    print("=" * 70)
    print("IEUM ASR API 시작")
    print("=" * 70)

    # 최종 모델 로딩
    model_loader.load()

    # 추론 객체 생성
    asr_inference = ASRInference(
        model_loader=model_loader
    )

    print("ASR inference 준비 완료")

    yield

    print()
    print("=" * 70)
    print("IEUM ASR API 종료")
    print("=" * 70)


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="IEUM ASR + TTS API",
    description="구음장애 음성인식 및 Piper TTS를 위한 FastAPI 서버",
    version="0.3.0",
    lifespan=lifespan,
)


# ============================================================
# API - Root
# ============================================================

@app.get("/")
def root():
    return {
        "service": "IEUM ASR + TTS API",
        "status": "running",
        "model": "Whisper Small Last4 + BiGRU CTC",
        "tts": "Piper ko_KR-kss-medium",
    }


# ============================================================
# API - Health Check
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "model_loaded": model_loader.is_loaded,
    }


# ============================================================
# API - Speech Recognition
# ============================================================

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
):
    """
    Raspberry Pi에서 전달된 음성 파일을 받아
    ASR 결과를 반환한다.
    """

    if audio.filename is None:
        raise HTTPException(
            status_code=400,
            detail="Audio filename is missing.",
        )

    allowed_extensions = (
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
    )

    if not audio.filename.lower().endswith(
        allowed_extensions
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format.",
        )

    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file.",
        )

    if asr_inference is None:
        raise HTTPException(
            status_code=503,
            detail="ASR model is not ready.",
        )

    try:
        text = asr_inference.transcribe(
            audio_bytes=audio_bytes,
            filename=audio.filename,
        )

    except Exception as error:
        print()
        print("=" * 70)
        print("ASR 추론 중 오류 발생")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(
            status_code=500,
            detail=f"ASR inference failed: {type(error).__name__}: {error}",
        )

    return {
        "filename": audio.filename,
        "text": text,
    }


# ============================================================
# API - Text To Speech
# ============================================================

@app.post("/tts")
async def synthesize_tts(request: TTSRequest):
    """
    입력된 텍스트를 Piper TTS로 변환하여
    WAV 파일을 반환한다.
    """

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="TTS text is empty.",
        )

    try:
        output_path = synthesize_text_to_speech(
            request.text
        )

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=output_path.name,
        )

    except Exception as error:
        print()
        print("=" * 70)
        print("TTS 생성 중 오류 발생")
        print("=" * 70)

        traceback.print_exc()

        print("=" * 70)

        raise HTTPException(
            status_code=500,
            detail=f"TTS synthesis failed: {type(error).__name__}: {error}",
        )