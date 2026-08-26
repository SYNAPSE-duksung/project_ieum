import os
import subprocess

import requests
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

device_state = {
    "status": "idle",
    "general_text": "",
    "personalized_text": "",
}

FASTAPI_SERVER_URL = os.getenv(
    "FASTAPI_SERVER_URL",
    "http://192.168.137.1:8000",
)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/transcribe/compare", methods=["POST"])
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
            f"{FASTAPI_SERVER_URL}/transcribe/compare",
            files=files,
            data={"speaker_id": "HYH_M_22"},
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

        general_text = str(
            result.get("general_text", "")
        ).strip()

        personalized_text = str(
            result.get("personalized_text", "")
        ).strip()

        if not general_text and not personalized_text:
            return jsonify({
                "success": False,
                "error": "인식된 문장이 없습니다."
            }), 422

        return jsonify({
            "success": True,
            "general_text": general_text,
            "personalized_text": personalized_text,
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

@app.route("/api/device/state", methods=["GET"])
def get_device_state():
    return jsonify({
        "success": True,
        **device_state,
    })


@app.route("/api/device/state", methods=["POST"])
def update_device_state():
    data = request.get_json(silent=True) or {}

    device_state["status"] = str(
        data.get("status", device_state["status"])
    )

    if "general_text" in data:
        device_state["general_text"] = str(
            data.get("general_text") or ""
        )

    if "personalized_text" in data:
        device_state["personalized_text"] = str(
            data.get("personalized_text") or ""
        )

    return jsonify({
        "success": True
    })

@app.route("/api/device/tts", methods=["POST"])
def device_tts():
    text = (
        device_state.get("personalized_text")
        or device_state.get("general_text")
        or ""
    ).strip()

    if not text:
        return jsonify({
            "success": False,
            "error": "읽어줄 문장이 없습니다."
        }), 400

    output_path = "outputs/ui_tts.wav"

    try:
        response = requests.post(
            f"{FASTAPI_SERVER_URL}/tts",
            json={"text": text},
            timeout=60,
        )

        response.raise_for_status()

        os.makedirs(
            "outputs",
            exist_ok=True,
        )

        with open(output_path, "wb") as file:
            file.write(response.content)

        subprocess.run(
            ["pw-play", output_path],
            check=True,
        )

        return jsonify({
            "success": True,
            "text": text,
        })

    except requests.RequestException as error:
        print(f"[ERROR] TTS API 요청 실패: {error}")

        return jsonify({
            "success": False,
            "error": "TTS 생성에 실패했습니다."
        }), 500

    except (OSError, subprocess.CalledProcessError) as error:
        print(f"[ERROR] TTS 재생 실패: {error}")

        return jsonify({
            "success": False,
            "error": "스피커 재생에 실패했습니다."
        }), 500

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )