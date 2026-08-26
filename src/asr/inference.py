from __future__ import annotations

import io
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.asr.feature_extractor import (
    IEUMWhisperFeatureExtractor,
)

from src.asr.tokenizer import (
    CTCCharacterTokenizer,
)

from src.api.model_loader import (
    ModelLoader,
)


class ASRInference:
    """
    IEUM ASR 추론 클래스.

    지원 기능
    ---------
    1. 범용 모델 단독 추론
    2. 특정 화자의 개인화 모델 추론
    3. 동일 음성에 대해
       범용 모델 / 개인화 모델 결과 동시 비교

    공통 처리 흐름
    -------------
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
    ┌────────────────────┐
    │                    │
    ▼                    ▼
    General Model    Personalized Model
    │                    │
    ▼                    ▼
    General Text     Personalized Text
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        sample_rate: int = 16000,
    ) -> None:

        if not model_loader.is_loaded:
            raise RuntimeError(
                "ModelLoader가 먼저 범용 모델을 "
                "로드해야 합니다."
            )

        self.model_loader = (
            model_loader
        )

        # ====================================================
        # General model
        # ====================================================

        self.general_model = (
            model_loader.get_model()
        )

        self.general_tokenizer = (
            model_loader.get_tokenizer()
        )

        self.device = (
            model_loader.get_device()
        )

        self.sample_rate = (
            sample_rate
        )

        # ====================================================
        # Feature extractor
        #
        # 학습 및 기존 API와 동일한 Whisper feature extractor
        # ====================================================

        self.feature_extractor = (
            IEUMWhisperFeatureExtractor(
                model_name=(
                    "openai/whisper-small"
                ),
                sample_rate=sample_rate,
                max_audio_seconds=30.0,
            )
        )


    # ========================================================
    # Audio preprocessing
    # ========================================================

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

        # ----------------------------------------------------
        # WAV bytes
        # ----------------------------------------------------

        waveform_np, source_sample_rate = (
            sf.read(
                audio_buffer,
                dtype="float32",
                always_2d=False,
            )
        )

        if waveform_np.size == 0:

            raise ValueError(
                "로드된 음성 데이터가 "
                "비어 있습니다."
            )

        # ----------------------------------------------------
        # Stereo → Mono
        #
        # soundfile stereo:
        # [time, channels]
        # ----------------------------------------------------

        if waveform_np.ndim == 2:

            waveform_np = (
                waveform_np.mean(
                    axis=1
                )
            )

        if waveform_np.ndim != 1:

            raise ValueError(
                "지원하지 않는 오디오 shape입니다. "
                f"shape={waveform_np.shape}"
            )

        # ----------------------------------------------------
        # NumPy → Tensor
        # ----------------------------------------------------

        waveform = torch.from_numpy(
            np.asarray(
                waveform_np,
                dtype=np.float32,
            )
        )

        # ----------------------------------------------------
        # Resample → 16 kHz
        # ----------------------------------------------------

        if (
            source_sample_rate
            != self.sample_rate
        ):

            waveform = (
                torchaudio.functional.resample(
                    waveform,
                    orig_freq=(
                        source_sample_rate
                    ),
                    new_freq=(
                        self.sample_rate
                    ),
                )
            )

        return waveform.float()


    def _validate_waveform(
        self,
        waveform: torch.Tensor,
    ) -> float:
        """
        음성 길이를 검증하고 duration을 반환한다.
        """

        duration = (
            waveform.numel()
            / self.sample_rate
        )

        if duration < 0.1:

            raise ValueError(
                "음성이 너무 짧습니다. "
                "최소 0.1초 이상의 "
                "음성이 필요합니다."
            )

        if duration > 30.0:

            raise ValueError(
                "음성이 30초를 초과했습니다. "
                "현재 서비스 추론은 "
                "최대 30초까지 지원합니다."
            )

        return float(
            duration
        )


    # ========================================================
    # Feature extraction
    # ========================================================

    def _extract_features(
        self,
        waveform: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        동일 waveform에서 Whisper 입력 feature를
        한 번만 추출한다.

        General / Personalized 모델 모두
        이 동일한 feature를 사용한다.
        """

        features = (
            self.feature_extractor(
                waveform
            )
        )

        input_features = (
            features[
                "input_features"
            ]
            .unsqueeze(0)
            .to(self.device)
        )

        audio_num_samples = (
            torch.tensor(
                [
                    features[
                        "audio_num_samples"
                    ]
                ],
                dtype=torch.long,
                device=self.device,
            )
        )

        return (
            input_features,
            audio_num_samples,
        )


    # ========================================================
    # Decode
    # ========================================================

    @staticmethod
    def _decode(
        logits: torch.Tensor,
        input_lengths: torch.Tensor,
        tokenizer: CTCCharacterTokenizer,
    ) -> str:
        """
        CTC logits를 Greedy CTC Decoding한다.

        tokenizer를 외부에서 받는 이유
        ------------------------------
        General:
            vocab size = 824

        Personalized:
            vocab size = 1095

        두 모델의 tokenizer가 다르므로
        모델에 맞는 tokenizer를 사용해야 한다.
        """

        if logits.ndim != 3:

            raise RuntimeError(
                "예상하지 못한 logits shape입니다.\n"
                f"logits.shape={tuple(logits.shape)}"
            )

        predicted_ids = (
            torch.argmax(
                logits,
                dim=-1,
            )
        )

        if (
            predicted_ids.shape[0]
            != 1
        ):

            raise RuntimeError(
                "현재 API는 한 번에 "
                "음성 1개만 처리합니다."
            )

        valid_length = int(
            input_lengths[
                0
            ].item()
        )

        valid_length = max(
            1,
            min(
                valid_length,
                predicted_ids.shape[1],
            ),
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

        text = tokenizer.decode(
            token_ids,
            ctc_decode=True,
        )

        return text


    # ========================================================
    # General inference
    # ========================================================

    @torch.inference_mode()
    def _infer_general_from_features(
        self,
        input_features: torch.Tensor,
        audio_num_samples: torch.Tensor,
    ) -> str:
        """
        이미 추출된 feature로 범용 모델 추론.
        """

        logits, input_lengths = (
            self.general_model(
                input_features=(
                    input_features
                ),
                audio_num_samples=(
                    audio_num_samples
                ),
            )
        )

        return self._decode(
            logits=logits,
            input_lengths=input_lengths,
            tokenizer=(
                self.general_tokenizer
            ),
        )


    # ========================================================
    # Personalized inference
    # ========================================================

    @torch.inference_mode()
    def _infer_personalized_from_features(
        self,
        input_features: torch.Tensor,
        audio_num_samples: torch.Tensor,
        speaker_id: str,
    ) -> str:
        """
        이미 추출된 동일 feature로
        특정 화자의 개인화 모델을 추론한다.
        """

        personalized_model = (
            self.model_loader
            .get_personalized_model(
                speaker_id
            )
        )

        personalized_tokenizer = (
            self.model_loader
            .get_personalized_tokenizer(
                speaker_id
            )
        )

        logits, input_lengths = (
            personalized_model(
                input_features=(
                    input_features
                ),
                audio_num_samples=(
                    audio_num_samples
                ),
            )
        )

        return self._decode(
            logits=logits,
            input_lengths=input_lengths,
            tokenizer=(
                personalized_tokenizer
            ),
        )


    # ========================================================
    # 기존 General-only API
    # ========================================================

    @torch.inference_mode()
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None,
    ) -> str:
        """
        기존 FastAPI와의 호환성을 위한
        범용 모델 단독 추론.

        기존 main.py에서 transcribe()를 사용하더라도
        동작이 깨지지 않도록 유지한다.
        """

        waveform = self._load_audio(
            audio_bytes
        )

        self._validate_waveform(
            waveform
        )

        (
            input_features,
            audio_num_samples,
        ) = self._extract_features(
            waveform
        )

        text = (
            self._infer_general_from_features(
                input_features=(
                    input_features
                ),
                audio_num_samples=(
                    audio_num_samples
                ),
            )
        )

        return text


    # ========================================================
    # Personalized-only inference
    # ========================================================

    @torch.inference_mode()
    def transcribe_personalized(
        self,
        audio_bytes: bytes,
        speaker_id: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        특정 화자의 개인화 모델만 추론한다.
        """

        waveform = self._load_audio(
            audio_bytes
        )

        self._validate_waveform(
            waveform
        )

        (
            input_features,
            audio_num_samples,
        ) = self._extract_features(
            waveform
        )

        text = (
            self._infer_personalized_from_features(
                input_features=(
                    input_features
                ),
                audio_num_samples=(
                    audio_num_samples
                ),
                speaker_id=(
                    speaker_id
                ),
            )
        )

        return text


    # ========================================================
    # General vs Personalized comparison
    # ========================================================

    @torch.inference_mode()
    def transcribe_compare(
        self,
        audio_bytes: bytes,
        speaker_id: str,
        filename: Optional[str] = None,
    ) -> dict:
        """
        동일한 음성을 범용 모델과
        선택된 개인화 모델에 각각 입력하여
        두 결과를 동시에 반환한다.

        반환 예시
        --------
        {
            "speaker_id": "HYH_M_22",
            "filename": "sample.wav",
            "duration_seconds": 3.2,
            "general_text": "...",
            "personalized_text": "..."
        }
        """

        # ----------------------------------------------------
        # 1. audio bytes → waveform
        # ----------------------------------------------------

        waveform = self._load_audio(
            audio_bytes
        )

        duration = (
            self._validate_waveform(
                waveform
            )
        )

        # ----------------------------------------------------
        # 2. Feature extraction
        #
        # 동일 음성에서 딱 한 번 생성
        # ----------------------------------------------------

        (
            input_features,
            audio_num_samples,
        ) = self._extract_features(
            waveform
        )

        # ----------------------------------------------------
        # 3. General
        # ----------------------------------------------------

        general_text = (
            self._infer_general_from_features(
                input_features=(
                    input_features
                ),
                audio_num_samples=(
                    audio_num_samples
                ),
            )
        )

        # ----------------------------------------------------
        # 4. Personalized
        # ----------------------------------------------------

        personalized_text = (
            self._infer_personalized_from_features(
                input_features=(
                    input_features
                ),
                audio_num_samples=(
                    audio_num_samples
                ),
                speaker_id=(
                    speaker_id
                ),
            )
        )

        # ----------------------------------------------------
        # 5. Response data
        # ----------------------------------------------------

        return {
            "speaker_id": (
                speaker_id
            ),
            "filename": (
                filename
            ),
            "duration_seconds": (
                round(
                    duration,
                    3,
                )
            ),
            "general_text": (
                general_text
            ),
            "personalized_text": (
                personalized_text
            ),
        }
