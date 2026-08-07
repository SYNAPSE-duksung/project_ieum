# 1차원 음성 Tensor를 Whisper 방식의 Log-Mel로 변환하는 코드
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from transformers import WhisperFeatureExtractor


class IEUMWhisperFeatureExtractor:
    """
    IEUM ASR용 Whisper 음성 Feature Extractor.

    16 kHz waveform을 Whisper Encoder가 사용하는
    log-Mel spectrogram(input_features)으로 변환한다.

    Whisper Encoder 자체를 포함하지는 않는다.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        sample_rate: int = 16000,
        max_audio_seconds: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds

        self.max_audio_samples = int(
            sample_rate * max_audio_seconds
        )

        self.feature_extractor = (
            WhisperFeatureExtractor.from_pretrained(
                model_name
            )
        )

    @staticmethod
    def _to_numpy(
        waveform: Tensor | np.ndarray,
    ) -> np.ndarray:
        """
        waveform을 float32 NumPy 배열로 변환한다.
        """
        if isinstance(waveform, Tensor):
            waveform = (
                waveform
                .detach()
                .cpu()
                .float()
                .numpy()
            )

        waveform = np.asarray(
            waveform,
            dtype=np.float32,
        )

        if waveform.ndim != 1:
            raise ValueError(
                "waveform은 1차원이어야 합니다.\n"
                f"현재 shape: {waveform.shape}"
            )

        return waveform

    def __call__(
        self,
        waveform: Tensor | np.ndarray,
    ) -> dict[str, Tensor | int]:
        """
        음성 하나를 Whisper input_features로 변환한다.

        Returns:
            {
                "input_features": Tensor [80, 3000],
                "audio_num_samples": int,
            }
        """
        waveform = self._to_numpy(waveform)

        audio_num_samples = len(waveform)

        if audio_num_samples == 0:
            raise ValueError("빈 waveform이 입력되었습니다.")

        if audio_num_samples > self.max_audio_samples:
            raise ValueError(
                "Whisper 최대 입력 길이를 초과했습니다.\n"
                f"현재 샘플 수: {audio_num_samples}\n"
                f"최대 샘플 수: {self.max_audio_samples}"
            )

        features = self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            max_length=self.max_audio_samples,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # [1, 80, 3000] → [80, 3000]
        input_features = features.input_features.squeeze(0)

        return {
            "input_features": input_features,
            "audio_num_samples": audio_num_samples,
        }

    def batch(
        self,
        waveforms: Sequence[Tensor | np.ndarray],
    ) -> dict[str, Tensor]:
        """
        여러 음성을 한 번에 Whisper input_features로 변환한다.

        Returns:
            input_features:
                [batch, 80, 3000]

            audio_num_samples:
                각 음성의 실제 sample 수
        """
        numpy_waveforms = [
            self._to_numpy(waveform)
            for waveform in waveforms
        ]

        audio_lengths = [
            len(waveform)
            for waveform in numpy_waveforms
        ]

        for length in audio_lengths:
            if length == 0:
                raise ValueError(
                    "Batch 안에 빈 waveform이 있습니다."
                )

            if length > self.max_audio_samples:
                raise ValueError(
                    "Batch 안에 30초를 초과하는 "
                    "음성이 있습니다."
                )

        features = self.feature_extractor(
            numpy_waveforms,
            sampling_rate=self.sample_rate,
            max_length=self.max_audio_samples,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_features": features.input_features,
            "audio_num_samples": torch.tensor(
                audio_lengths,
                dtype=torch.long,
            ),
        }