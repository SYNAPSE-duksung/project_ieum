from pathlib import Path
import subprocess
import uuid


# project_ieum/
# └── src/
#     └── tts/
#         └── piper_tts.py
#
# parents[2] → project_ieum
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "piper" / "ko_KR-kss-medium.onnx"
CONFIG_PATH = PROJECT_ROOT / "piper" / "ko_KR-kss-medium.onnx.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def synthesize_text_to_speech(text: str) -> Path:
    """
    입력된 한국어 텍스트를 Piper TTS로 변환하여
    WAV 파일을 생성한다.
    """

    if not text or not text.strip():
        raise ValueError("TTS text is empty.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Piper model not found: {MODEL_PATH}"
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Piper config not found: {CONFIG_PATH}"
        )

    output_path = OUTPUT_DIR / f"tts_{uuid.uuid4().hex}.wav"

    command = [
        "piper",
        "-m",
        str(MODEL_PATH),
        "-c",
        str(CONFIG_PATH),
        "-f",
        str(output_path),
        "--length-scale",
        "5",
    ]

    try:
        subprocess.run(
            command,
            input=text,
            text=True,
            check=True,
            capture_output=True,
        )

    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Piper TTS failed: {error.stderr}"
        ) from error

    if not output_path.exists():
        raise RuntimeError(
            "Piper finished but WAV file was not created."
        )

    return output_path