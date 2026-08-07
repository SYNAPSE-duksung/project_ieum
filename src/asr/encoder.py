from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import WhisperModel


class WhisperEncoder(nn.Module):
    """
    Hugging Face Whisper Encoder wrapper.

    역할:
        Whisper input_features
        [B, 80, 3000]

        ↓

        Whisper Encoder hidden states
        [B, T, hidden_size]

    encoder_train_mode:
        freeze : Encoder 전체 고정
        last2  : 마지막 2개 layer만 학습
        last4  : 마지막 4개 layer만 학습
        full   : Encoder 전체 학습
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        train_mode: str = "freeze",
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.train_mode = train_mode

        whisper = WhisperModel.from_pretrained(model_name)

        self.encoder = whisper.encoder

        # Whisper decoder는 사용하지 않으므로 메모리에서 제거
        del whisper.decoder

        self.hidden_size = self.encoder.config.d_model

        self._set_train_mode(train_mode)

    def _set_train_mode(
        self,
        train_mode: str,
    ) -> None:

        # 표현 차이를 모두 허용
        aliases = {
            "last_2": "last2",
            "last_4": "last4",
        }

        train_mode = aliases.get(
            train_mode,
            train_mode,
        )

        valid_modes = {
            "freeze",
            "last2",
            "last4",
            "full",
        }

        if train_mode not in valid_modes:
            raise ValueError(
                f"지원하지 않는 train_mode입니다: {train_mode}\n"
                f"사용 가능: {sorted(valid_modes)}"
            )

        # 우선 Encoder 전체 고정
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        if train_mode == "freeze":
            return

        if train_mode == "full":
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
            return

        num_train_layers = {
            "last2": 2,
            "last4": 4,
        }[train_mode]

        encoder_layers = self.encoder.layers

        if num_train_layers > len(encoder_layers):
            raise ValueError(
                f"Encoder layer 수보다 학습 요청 layer가 많습니다.\n"
                f"전체 layer: {len(encoder_layers)}\n"
                f"요청: {num_train_layers}"
            )

        for layer in encoder_layers[-num_train_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        # 마지막 LayerNorm도 함께 학습
        if hasattr(self.encoder, "layer_norm"):
            for parameter in self.encoder.layer_norm.parameters():
                parameter.requires_grad = True

    def forward(
        self,
        input_features: Tensor,
    ) -> Tensor:
        """
        Args:
            input_features:
                [batch, 80, 3000]

        Returns:
            hidden_states:
                [batch, time, hidden_size]
        """

        outputs = self.encoder(
            input_features=input_features,
            return_dict=True,
        )

        return outputs.last_hidden_state

    @staticmethod
    def get_output_lengths(
        audio_num_samples: Tensor,
        sample_rate: int = 16000,
        hop_length: int = 160,
    ) -> Tensor:
        """
        실제 음성 길이를 Whisper Encoder 시간축 길이로 근사 변환한다.

        Whisper log-Mel:
            약 10 ms당 한 frame

        Whisper Encoder:
            convolution stride=2

        따라서 대략
            audio samples
              ↓ /160
            Mel frames
              ↓ /2
            Encoder frames
        """

        mel_lengths = torch.div(
            audio_num_samples + hop_length - 1,
            hop_length,
            rounding_mode="floor",
        )

        encoder_lengths = torch.div(
            mel_lengths + 1,
            2,
            rounding_mode="floor",
        )

        return encoder_lengths.long()

    def trainable_parameter_summary(
        self,
    ) -> dict[str, int]:

        total = sum(
            parameter.numel()
            for parameter in self.encoder.parameters()
        )

        trainable = sum(
            parameter.numel()
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        )

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }