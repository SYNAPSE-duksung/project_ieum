from pathlib import Path

import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

DEVICE = 1
RECORD_SAMPLE_RATE = 44100
OUTPUT_SAMPLE_RATE = 16000
CHANNELS = 1

MAX_DURATION = 30.0


def record_audio(
    output_path: str = "recorded_audio.wav",
    duration: float = 5.0,
) -> str:

    if duration <= 0:
        raise ValueError("Duration must be greater than 0.")

    if duration > MAX_DURATION:
        raise ValueError(
            f"Maximum recording duration is {MAX_DURATION} seconds."
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording started ({duration} sec)...")

    # USB 마이크가 지원하는 44.1kHz로 녹음
    audio = sd.rec(
        int(duration * RECORD_SAMPLE_RATE),
        samplerate=RECORD_SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=DEVICE,
    )

    sd.wait()

    # 모델 입력 규격인 16kHz로 변환
    audio = resample_poly(
        audio,
        OUTPUT_SAMPLE_RATE,
        RECORD_SAMPLE_RATE,
        axis=0,
    )

    # 16kHz, Mono, PCM 16-bit WAV로 저장
    sf.write(
        path,
        audio,
        OUTPUT_SAMPLE_RATE,
        subtype="PCM_16",
    )

    print(f"Recording completed: {path}")

    return str(path)


if __name__ == "__main__":
    record_audio(
        output_path="recorded_audio.wav",
        duration=30.0,
    )