from __future__ import annotations

import io
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.asr.feature_extractor import IEUMWhisperFeatureExtractor
from src.api.model_loader import ModelLoader


class ASRInference:
    """
    IEUM 최종 범용 ASR 모델 추론 클래스.

    처리 흐름:
        audio bytes
            ↓
        waveform 로드
            ↓
        mono 변환
            ↓
        16 kHz resampling
            ↓
        Whisper Log-Mel Feature
            ↓
        Whisper Small Encoder (Last4 Fine-tuned)
            ↓
        BiGRU
            ↓
        CTC Greedy Decoding
            ↓
        최종 문장
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        sample_rate: int = 16000,
    ) -> None:

        if not model_loader.is_loaded:
            raise RuntimeError(
                "ModelLoader가 먼저 모델을 로드해야 합니다."
            )

        self.model_loader = model_loader

        self.model = model_loader.get_model()
        self.tokenizer = model_loader.get_tokenizer()
        self.device = model_loader.get_device()

        self.sample_rate = sample_rate

        # 학습 시 사용한 것과 동일한
        # Whisper Feature Extractor
        self.feature_extractor = IEUMWhisperFeatureExtractor(
            model_name="openai/whisper-small",
            sample_rate=sample_rate,
            max_audio_seconds=30.0,
        )

    def _load_audio(
        self,
        audio_bytes: bytes,
    ) -> torch.Tensor:
        """
        업로드된 WAV bytes를
        16 kHz mono waveform으로 변환한다.
        """

        if not audio_bytes:
            raise ValueError(
                "Audio data is empty."
            )

        audio_buffer = io.BytesIO(
            audio_bytes
        )

        # WAV bytes 직접 로드
        waveform_np, source_sample_rate = sf.read(
            audio_buffer,
            dtype="float32",
            always_2d=False,
        )

        if waveform_np.size == 0:
            raise ValueError(
                "로드된 음성 데이터가 비어 있습니다."
            )

        # --------------------------------------------------
        # Stereo → Mono
        #
        # soundfile stereo shape:
        # [time, channels]
        # --------------------------------------------------

        if waveform_np.ndim == 2:
            waveform_np = waveform_np.mean(
                axis=1
            )

        if waveform_np.ndim != 1:
            raise ValueError(
                "지원하지 않는 오디오 shape입니다. "
                f"shape={waveform_np.shape}"
            )

        # NumPy → Tensor
        waveform = torch.from_numpy(
            np.asarray(
                waveform_np,
                dtype=np.float32,
            )
        )

        # --------------------------------------------------
        # 16 kHz Resampling
        # --------------------------------------------------

        if source_sample_rate != self.sample_rate:

            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=source_sample_rate,
                new_freq=self.sample_rate,
            )

        return waveform.float()

    def _decode(
        self,
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> str:
        """
        CTC logits를 Greedy CTC Decoding하여
        최종 문자열로 변환한다.
        """

        predicted_ids = torch.argmax(
            logits,
            dim=-1,
        )

        # 현재 API는 한 번에 음성 1개 처리
        valid_length = int(
            input_lengths[0].item()
        )

        token_ids = (
            predicted_ids[
                0,
                :valid_length,
            ]
            .detach()
            .cpu()
            .tolist()
        )

        # CTC:
        # 연속 중복 제거 + blank 제거
        text = self.tokenizer.decode(
            token_ids,
            ctc_decode=True,
        )

        return text

    @torch.inference_mode()
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None,
    ) -> str:
        """
        음성 파일 하나를 실제 ASR 모델로 추론한다.
        """

        # ==================================================
        # 1. bytes → waveform
        # ==================================================

        waveform = self._load_audio(
            audio_bytes
        )

        duration = (
            waveform.numel()
            / self.sample_rate
        )

        if duration < 0.1:
            raise ValueError(
                "음성이 너무 짧습니다. "
                "최소 0.1초 이상의 음성이 필요합니다."
            )

        if duration > 30.0:
            raise ValueError(
                "음성이 30초를 초과했습니다. "
                "현재 서비스 추론은 최대 30초까지 지원합니다."
            )

        # ==================================================
        # 2. Whisper Input Feature
        # ==================================================

        features = self.feature_extractor(
            waveform
        )

        input_features = (
            features["input_features"]
            .unsqueeze(0)
            .to(self.device)
        )

        audio_num_samples = torch.tensor(
            [
                features[
                    "audio_num_samples"
                ]
            ],
            dtype=torch.long,
            device=self.device,
        )

        # ==================================================
        # 3. 최종 ASR 모델 추론
        #
        # Whisper Encoder (Last4)
        #       ↓
        # BiGRU
        #       ↓
        # CTC logits
        # ==================================================

        logits, input_lengths = self.model(
            input_features=input_features,
            audio_num_samples=audio_num_samples,
        )

        # ==================================================
        # 4. CTC Decode
        # ==================================================

        text = self._decode(
            logits=logits,
            input_lengths=input_lengths,
        )

        return text