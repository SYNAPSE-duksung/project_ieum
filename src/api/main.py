from fastapi import FastAPI, File, HTTPException, UploadFile
from src.asr.inference import ASRInference

app = FastAPI(
    title="IEUM ASR API",
    description="구음장애 음성인식 모델 추론을 위한 FastAPI 서버",
    version="0.1.0",
)

asr_inference = ASRInference()


@app.get("/")
def root():
    """
    API 기본 동작 확인
    """
    return {
        "service": "IEUM ASR API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    """
    Raspberry Pi 또는 외부 클라이언트에서
    서버가 정상적으로 실행 중인지 확인
    """
    return {
        "status": "ok",
    }


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...)
):
    """
    Raspberry Pi에서 전달된 음성 파일을 받아
    음성인식 결과를 반환한다.
    """

    if audio.filename is None:
        raise HTTPException(
            status_code=400,
            detail="Audio filename is missing.",
        )

    allowed_extensions = (".wav", ".mp3", ".m4a", ".flac")

    if not audio.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format.",
        )

    # 실제 음성 데이터 읽기
    audio_bytes = await audio.read()

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file.",
        )

    text = asr_inference.transcribe(
    audio_bytes=audio_bytes,
    filename=audio.filename,
    )

    return {
        "filename": audio.filename,
        "text": text,
    }