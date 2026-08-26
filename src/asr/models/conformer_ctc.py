from __future__ import annotations

import torch
from torch import Tensor, nn
from torchaudio.models import Conformer


class ConformerCTC(nn.Module):
    """
    Whisper Encoder 출력 위에 Conformer를 추가한 CTC 모델.

    입력:
        hidden_states: [B, T, input_dim]

    출력:
        logits: [B, T, vocab_size]
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        feedforward_size: int = 2048,
        convolution_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size는 num_heads로 나누어 떨어져야 합니다."
            )

        if convolution_kernel_size % 2 == 0:
            raise ValueError(
                "Conformer convolution kernel size는 홀수여야 합니다."
            )

        self.input_projection = nn.Linear(
            input_dim,
            hidden_size,
        )

        self.conformer = Conformer(
            input_dim=hidden_size,
            num_heads=num_heads,
            ffn_dim=feedforward_size,
            num_layers=num_layers,
            depthwise_conv_kernel_size=convolution_kernel_size,
            dropout=dropout,
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

    def forward(
        self,
        hidden_states: Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            hidden_states:
                Whisper Encoder 출력.
                shape = [B, T, input_dim]

            lengths:
                각 샘플의 실제 Encoder time length.
                shape = [B]

        Returns:
            logits:
                CTC vocabulary logits.
                shape = [B, T, vocab_size]
        """

        output = self.input_projection(
            hidden_states
        )

        # lengths가 없으면 전체 sequence length를 사용
        if lengths is None:
            lengths = torch.full(
                size=(output.shape[0],),
                fill_value=output.shape[1],
                dtype=torch.long,
                device=output.device,
            )

        lengths = lengths.to(
            device=output.device,
            dtype=torch.long,
        )

        # sequence length보다 큰 값이 들어오는 것을 방지
        lengths = lengths.clamp(
            min=1,
            max=output.shape[1],
        )

        output, _ = self.conformer(
            output,
            lengths,
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