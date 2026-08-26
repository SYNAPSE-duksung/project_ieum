from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio
import torch


# ============================================================
# 설정
# ============================================================

INPUT_WAV = Path(
    r"C:\ieum_asr\original.wav"
)

OUTPUT_WAV = Path(
    r"C:\ieum_asr\test_30sec.wav"
)

TARGET_SAMPLE_RATE = 16000

# FastAPI 제한이 30초이므로
# 경계 문제 방지를 위해 29.5초 권장
TARGET_SECONDS = 29.5

# 음성량을 비교할 window 단위
ENERGY_FRAME_SECONDS = 0.5


# ============================================================
# Audio load
# ============================================================

waveform_np, sample_rate = sf.read(
    INPUT_WAV,
    dtype="float32",
    always_2d=False,
)

print("=" * 80)
print("원본 음성")
print("=" * 80)

print("경로:", INPUT_WAV)
print("Sample rate:", sample_rate)
print(
    "원본 길이:",
    len(waveform_np) / sample_rate,
    "sec"
)


# ============================================================
# Stereo → Mono
# ============================================================

if waveform_np.ndim == 2:
    waveform_np = waveform_np.mean(
        axis=1
    )

waveform = torch.from_numpy(
    np.asarray(
        waveform_np,
        dtype=np.float32,
    )
)


# ============================================================
# Resample → 16 kHz
# ============================================================

if sample_rate != TARGET_SAMPLE_RATE:

    waveform = torchaudio.functional.resample(
        waveform,
        orig_freq=sample_rate,
        new_freq=TARGET_SAMPLE_RATE,
    )

    sample_rate = TARGET_SAMPLE_RATE


# ============================================================
# 이미 29.5초 이하인 경우
# ============================================================

target_samples = int(
    TARGET_SECONDS * sample_rate
)

if waveform.numel() <= target_samples:

    selected = waveform

    start_sample = 0
    end_sample = waveform.numel()

else:

    # ========================================================
    # 발화가 많이 포함된 구간 찾기
    #
    # 0.5초마다 RMS energy 계산 후
    # 연속 29.5초 구간 중 energy 합이 가장 큰 곳 선택
    # ========================================================

    frame_samples = int(
        ENERGY_FRAME_SECONDS
        * sample_rate
    )

    num_frames = (
        waveform.numel()
        // frame_samples
    )

    frame_energies = []

    for i in range(num_frames):

        start = (
            i * frame_samples
        )

        end = (
            start
            + frame_samples
        )

        frame = waveform[
            start:end
        ]

        rms = torch.sqrt(
            torch.mean(
                frame ** 2
            )
            + 1e-12
        )

        frame_energies.append(
            float(rms)
        )

    frame_energies = np.asarray(
        frame_energies
    )

    window_frames = int(
        TARGET_SECONDS
        / ENERGY_FRAME_SECONDS
    )

    # --------------------------------------------------------
    # Sliding window energy
    # --------------------------------------------------------

    window_energy = np.convolve(
        frame_energies,
        np.ones(
            window_frames
        ),
        mode="valid",
    )

    best_frame = int(
        np.argmax(
            window_energy
        )
    )

    start_sample = (
        best_frame
        * frame_samples
    )

    end_sample = (
        start_sample
        + target_samples
    )

    # 파일 끝을 넘어가지 않도록 보정
    if end_sample > waveform.numel():

        end_sample = waveform.numel()

        start_sample = max(
            0,
            end_sample
            - target_samples
        )

    selected = waveform[
        start_sample:end_sample
    ]


# ============================================================
# Save
# ============================================================

OUTPUT_WAV.parent.mkdir(
    parents=True,
    exist_ok=True,
)

sf.write(
    OUTPUT_WAV,
    selected
    .detach()
    .cpu()
    .numpy(),
    sample_rate,
    subtype="PCM_16",
)


# ============================================================
# 결과
# ============================================================

start_sec = (
    start_sample
    / sample_rate
)

end_sec = (
    end_sample
    / sample_rate
)

duration_sec = (
    selected.numel()
    / sample_rate
)

print()
print("=" * 80)
print("30초 테스트 음성 생성 완료")
print("=" * 80)

print(
    f"선택 구간 : "
    f"{start_sec:.2f} ~ "
    f"{end_sec:.2f} sec"
)

print(
    f"최종 길이 : "
    f"{duration_sec:.2f} sec"
)

print(
    f"Sample rate: "
    f"{sample_rate}"
)

print("Channel    : mono")

print(
    f"저장 경로 : "
    f"{OUTPUT_WAV}"
)

print("=" * 80)