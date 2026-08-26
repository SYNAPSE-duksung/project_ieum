from __future__ import annotations

from torch import Tensor, nn


class LinearCTC(nn.Module):
    """
    가장 단순한 CTC baseline.

    Whisper Encoder의 각 시간 frame을
    vocabulary 확률 공간으로 직접 투영한다.
    """

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            input_dim,
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
                [B, T, input_dim]

        Returns:
            logits:
                [B, T, vocab_size]
        """

        hidden_states = self.dropout(
            hidden_states
        )

        logits = self.classifier(
            hidden_states
        )

        return logits