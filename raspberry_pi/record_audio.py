from pathlib import Path

import sounddevice as sd
import soundfile as sf


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

MAX_DURATION = 30.0


def record_audio(
    output_path: str = "recorded_audio.wav",
    duration: float = 5.0,
) -> str:

    if duration <= 0:
        raise ValueError(
            "Duration must be greater than 0."
        )

    if duration > MAX_DURATION:
        raise ValueError(
            f"Maximum recording duration is {MAX_DURATION} seconds."
        )

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Recording started ({duration} sec)..."
    )

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

    print(
        f"Recording completed: {path}"
    )

    return str(path)


if __name__ == "__main__":

    record_audio(
        output_path="recorded_audio.wav",
        duration=30.0,
    )