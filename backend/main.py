from fastapi import FastAPI, UploadFile, File
from whisper_model import transcribe
from gemini_model import correct

import shutil
import os
import time

app = FastAPI()

# 현재 파일(main.py) 기준 backend 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 업로드 폴더
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "Hello IEUM"
    }


@app.post("/speech")
async def upload_audio(audio: UploadFile = File(...)):

    start_time = time.time()

    save_path = os.path.join(
        UPLOAD_FOLDER,
        audio.filename
    )

    # -----------------------------
    # 파일 저장
    # -----------------------------
    try:

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

    except Exception as e:

        return {
            "success": False,
            "stage": "upload",
            "error": str(e)
        }

    # -----------------------------
    # Whisper
    # -----------------------------
    try:

        asr_text = transcribe(save_path)

    except Exception as e:

        return {
            "success": False,
            "stage": "whisper",
            "error": str(e)
        }

    # -----------------------------
    # Gemini
    # -----------------------------
    try:

        corrected_text = correct(asr_text)

    except Exception as e:

        print(f"[Gemini Error] {e}")

        # Gemini 실패 시 Whisper 결과 그대로 사용
        corrected_text = asr_text

    processing_time = round(
        time.time() - start_time,
        2
    )

    return {

        "success": True,

        "filename": audio.filename,

        "asr": asr_text,

        "corrected": corrected_text,

        "processing_time_sec": processing_time

    }