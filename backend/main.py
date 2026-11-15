
from fastapi import FastAPI, UploadFile, File
from whisper_model import transcribe

import shutil
import os
import time
from pathlib import Path

app = FastAPI()

# 현재 main.py가 있는 backend 폴더
BASE_DIR = Path(__file__).resolve().parent

# 업로드한 음성 파일 저장 폴더
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "Hello IEUM - General ASR"
    }


@app.post("/speech")
async def upload_audio(audio: UploadFile = File(...)):

    start_time = time.time()

    # 업로드 파일명에서 경로 부분 제거
    filename = os.path.basename(audio.filename or "audio.wav")
    save_path = UPLOAD_FOLDER / filename

    # 1. 음성 파일 저장
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

    except Exception as e:
        return {
            "success": False,
            "stage": "upload",
            "error": str(e)
        }

    # 2. IEUM 범용 ASR 음성 인식
    try:
        asr_text = transcribe(str(save_path))

    except Exception as e:
        return {
            "success": False,
            "stage": "general_asr",
            "error": str(e)
        }

    # 3. Gemini 없이 인식 결과 그대로 사용
    corrected_text = asr_text

    processing_time = round(
        time.time() - start_time,
        2
    )

    return {
        "success": True,
        "filename": filename,
        "asr": asr_text,
        "corrected": corrected_text,
        "processing_time_sec": processing_time
    }
