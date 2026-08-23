import os

import requests
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

FASTAPI_SERVER_URL = os.getenv(
    "FASTAPI_SERVER_URL",
    "http://127.0.0.1:8000",
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files.get("audio")

    if audio_file is None:
        return jsonify({
            "success": False,
            "error": "녹음 파일이 전달되지 않았습니다."
        }), 400

    if not audio_file.filename:
        return jsonify({
            "success": False,
            "error": "녹음 파일 이름이 없습니다."
        }), 400

    try:
        files = {
            "audio": (
                audio_file.filename,
                audio_file.stream,
                audio_file.mimetype or "audio/wav",
            )
        }

        response = requests.post(
            f"{FASTAPI_SERVER_URL}/transcribe",
            files=files,
            timeout=60,
        )

        if response.status_code != 200:
            try:
                detail = response.json().get(
                    "detail",
                    "FastAPI 음성 인식 오류",
                )
            except ValueError:
                detail = "FastAPI 음성 인식 오류"

            return jsonify({
                "success": False,
                "error": detail,
            }), response.status_code

        result = response.json()
        text = str(result.get("text", "")).strip()

        if not text:
            return jsonify({
                "success": False,
                "error": "인식된 문장이 없습니다."
            }), 422

        return jsonify({
            "success": True,
            "text": text,
        })

    except requests.Timeout:
        return jsonify({
            "success": False,
            "error": "음성 인식 시간이 초과되었습니다."
        }), 504

    except requests.ConnectionError:
        return jsonify({
            "success": False,
            "error": "FastAPI 서버에 연결할 수 없습니다."
        }), 503

    except requests.RequestException as error:
        print(f"[ERROR] FastAPI request failed: {error}")

        return jsonify({
            "success": False,
            "error": "음성 인식 서버 요청 중 오류가 발생했습니다."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )