from __future__ import annotations

import torch
from torch import Tensor, nn


class TransformerCTC(nn.Module):
    """
    Whisper Encoder 출력 위에 추가 Transformer Encoder를
    배치한 CTC 모델.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_size: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size는 num_heads로 "
                "나누어 떨어져야 합니다."
            )

        self.input_projection = nn.Linear(
            input_dim,
            hidden_size,
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=feedforward_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.layer_norm = nn.LayerNorm(
            hidden_size
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            hidden_size,
            vocab_size,
        )

    @staticmethod
    def _create_padding_mask(
        sequence_length: int,
        lengths: Tensor,
        device: torch.device,
    ) -> Tensor:
        """
        True인 위치가 padding 위치.

        shape:
            [B, T]
        """

        positions = torch.arange(
            sequence_length,
            device=device,
        ).unsqueeze(0)

        return positions >= lengths.unsqueeze(1)

    def forward(
        self,
        hidden_states: Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:

        output = self.input_projection(
            hidden_states
        )

        padding_mask = None

        if lengths is not None:
            lengths = lengths.to(
                hidden_states.device
            )

            lengths = lengths.clamp(
                max=output.shape[1]
            )

            padding_mask = (
                self._create_padding_mask(
                    sequence_length=output.shape[1],
                    lengths=lengths,
                    device=output.device,
                )
            )

        output = self.transformer(
            output,
            src_key_padding_mask=padding_mask,
        )

        output = self.layer_norm(
            output
        )

        output = self.dropout(
            output
        )

        logits = self.classifier(
            output
        )

        return logits