from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.asr.encoder import WhisperEncoder
from src.asr.models import BiGRUCTC
from src.asr.trainer import CTCASRModel


def load_general_model(
    checkpoint_path: str | Path,
    *,
    vocab_size: int,
    device: torch.device,
    model_name: str = "openai/whisper-small",
    encoder_train_mode: str = "freeze",
    hidden_size: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    sample_rate: int = 16000,
) -> tuple[CTCASRModel, dict[str, Any]]:
    """
    학습된 범용 IEUM ASR 모델을 개인화 학습의
    초기 모델로 불러온다.

    구조:
        Whisper Encoder
            ↓
        BiGRU
            ↓
        CTC classifier

    Parameters
    ----------
    checkpoint_path:
        범용 모델의 best_model.pt 경로.

    vocab_size:
        범용 모델 학습 당시 사용한 vocabulary 크기.

    encoder_train_mode:
        개인화 학습에서 Whisper Encoder의
        어느 부분을 학습할지 지정한다.

        freeze / last2 / last4 / full
    """

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "범용 모델 checkpoint를 찾을 수 없습니다.\n"
            f"경로: {checkpoint_path}"
        )

    # ========================================================
    # Whisper Encoder
    # ========================================================

    encoder = WhisperEncoder(
        model_name=model_name,
        train_mode=encoder_train_mode,
    )

    # ========================================================
    # BiGRU CTC
    # ========================================================

    downstream_model = BiGRUCTC(
        input_dim=encoder.hidden_size,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    # ========================================================
    # 전체 ASR Model
    # ========================================================

    model = CTCASRModel(
        encoder=encoder,
        downstream_model=downstream_model,
        sample_rate=sample_rate,
    )

    # ========================================================
    # General checkpoint
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "checkpoint에 model_state_dict가 없습니다."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        device
    )

    return model, checkpoint