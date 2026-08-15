from __future__ import annotations

from torch import Tensor, nn


class BiGRUCTC(nn.Module):
    """
    Whisper Encoder 출력에 Bidirectional GRU를 추가한 CTC 모델.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                "num_layers는 1 이상이어야 합니다."
            )

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,

            # PyTorch GRU에서는 num_layers=1일 때
            # dropout이 적용되지 않음
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # Bidirectional이므로
        # 출력 dimension = hidden_size * 2
        self.classifier = nn.Linear(
            hidden_size * 2,
            vocab_size,
        )

    def forward(
        self,
        hidden_states: Tensor,
        lengths: Tensor | None = None,
    ) -> Tensor:

        output, _ = self.gru(
            hidden_states
        )

        output = self.dropout(
            output
        )

        logits = self.classifier(
            output
        )

        return logits