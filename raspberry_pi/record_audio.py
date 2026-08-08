from pathlib import Path

import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def record_audio(
    output_path: str = "recorded_audio.wav",
    duration: float = 5.0,
) -> str:
    """
    마이크 입력을 녹음하여 WAV 파일로 저장한다.

    Parameters
    ----------
    output_path : str
        저장할 WAV 파일 경로

    duration : float
        녹음 시간(초)

    Returns
    -------
    str
        저장된 WAV 파일 경로
    """

    if duration <= 0:
        raise ValueError("Duration must be greater than 0.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording started ({duration} sec)...")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    )

    sd.wait()

    sf.write(
        path,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    print(f"Recording completed: {path}")

    return str(path)


if __name__ == "__main__":
    record_audio(
        output_path="recorded_audio.wav",
        duration=5.0,
    )