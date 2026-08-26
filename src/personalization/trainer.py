from __future__ import annotations

from typing import Any

import torch

from torch import Tensor, nn

from src.asr.tokenizer import (
    CTCCharacterTokenizer,
)

from src.asr.trainer import (
    CTCTrainer,
)


class PersonalizationTrainer(
    CTCTrainer
):
    """
    IEUM 화자 개인화 학습용 Trainer.

    기존 CTCTrainer의 기능을 그대로 사용하면서
    개인화 실험을 위한 loss 계산을 확장한다.

    지원 모드
    ---------
    baseline:
        일반 CTC loss를 사용한
        speaker-specific fine-tuning.

    error_profile:
        각 학습 샘플에 Error Profile 기반
        weight를 적용한 weighted CTC loss.

    현재 구조에서는 동일한 모델 architecture를 사용하고
    학습 loss 방식만 변경한다.
    """

    VALID_MODES = {
        "baseline",
        "error_profile",
    }

    def __init__(
        self,
        model: nn.Module,
        tokenizer: CTCCharacterTokenizer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        mode: str = "baseline",
        use_amp: bool = True,
        gradient_clip_norm: float = 1.0,
    ) -> None:

        mode = str(
            mode
        ).strip().lower()

        if mode not in self.VALID_MODES:
            raise ValueError(
                "지원하지 않는 개인화 학습 mode입니다.\n"
                f"입력: {mode}\n"
                f"사용 가능: {sorted(self.VALID_MODES)}"
            )

        self.mode = mode

        # ====================================================
        # 기존 CTCTrainer 초기화
        # ====================================================

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            gradient_clip_norm=(
                gradient_clip_norm
            ),
            use_cached_features=False,
        )

        # ====================================================
        # 개인화용 CTC loss
        #
        # sample별 loss가 필요하기 때문에
        # reduction="none" 사용
        # ====================================================

        self.sample_ctc_loss = nn.CTCLoss(
            blank=tokenizer.blank_id,
            reduction="none",
            zero_infinity=True,
        )

    # ========================================================
    # Batch
    # ========================================================

    def _move_batch(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:

        batch = super()._move_batch(
            batch
        )

        # Error Profile 실험에서 사용할
        # sample weight가 존재하면 device로 이동
        if "sample_weights" in batch:

            batch[
                "sample_weights"
            ] = (
                batch[
                    "sample_weights"
                ]
                .to(
                    self.device,
                    non_blocking=True,
                )
                .float()
            )

        return batch

    # ========================================================
    # Sample-level CTC loss
    # ========================================================

    def _calculate_sample_losses(
        self,
        logits: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        """
        각 sample의 길이 정규화된 CTC loss를 계산한다.

        기존 CTCTrainer의
        CTCLoss(reduction="mean")과 동일하게
        각 sample loss를 target length로 정규화한다.

        Returns
        -------
        Tensor
            shape: [batch_size]
        """

        log_probs = logits.log_softmax(
            dim=-1
        )

        # CTCLoss 입력 형식
        # [B, T, C] -> [T, B, C]
        log_probs = log_probs.transpose(
            0,
            1,
        )

        losses = self.sample_ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        # 기존 CTCLoss(reduction="mean")의
        # sample별 길이 정규화 방식 유지
        
        normalized_losses = (
            losses
            / target_lengths
            .to(
                losses.dtype
            )
            .clamp_min(1)
        )

        return normalized_losses

    # ========================================================
    # Loss
    # ========================================================

    def _calculate_loss(
        self,
        logits: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
        sample_weights: Tensor | None = None,
    ) -> Tensor:
        """
        개인화 실험 mode에 따라 loss를 계산한다.

        baseline
            sample loss의 단순 평균.

        error_profile
            sample loss에 Error Profile weight를 곱한 뒤
            weighted mean을 계산.
        """

        sample_losses = (
            self._calculate_sample_losses(
                logits=logits,
                input_lengths=input_lengths,
                targets=targets,
                target_lengths=target_lengths,
            )
        )

        # ====================================================
        # Baseline personalization
        # ====================================================

        if self.mode == "baseline":

            return sample_losses.mean()

        # ====================================================
        # Error Profile personalization
        # ====================================================

        if sample_weights is None:
            raise ValueError(
                "mode='error_profile'에서는 "
                "batch에 sample_weights가 필요합니다."
            )

        if (
            sample_weights.ndim != 1
        ):
            raise ValueError(
                "sample_weights는 "
                "1차원 Tensor여야 합니다."
            )

        if (
            sample_weights.shape[0]
            != sample_losses.shape[0]
        ):
            raise ValueError(
                "sample_weights 개수와 "
                "batch sample 수가 일치하지 않습니다.\n"
                f"sample_weights: "
                f"{sample_weights.shape[0]}\n"
                f"sample_losses: "
                f"{sample_losses.shape[0]}"
            )

        if not torch.all(
            torch.isfinite(
                sample_weights
            )
        ):
            raise ValueError(
                "sample_weights에 "
                "NaN 또는 Inf가 포함되어 있습니다."
            )

        if torch.any(
            sample_weights <= 0
        ):
            raise ValueError(
                "sample_weights는 모두 "
                "0보다 커야 합니다."
            )

        weighted_loss = (
            sample_losses
            * sample_weights
        ).sum()

        weight_sum = (
            sample_weights.sum()
            .clamp_min(
                1e-8
            )
        )

        return (
            weighted_loss
            / weight_sum
        )

    # ========================================================
    # Train epoch
    # ========================================================

    def train_epoch(
        self,
        dataloader,
    ) -> float:
        """
        기존 CTCTrainer의 train loop와 동일하되,
        개인화 mode에 따라 sample weight를 전달한다.
        """

        self.model.train()

        total_loss = 0.0
        total_batches = 0

        from tqdm.auto import tqdm

        progress = tqdm(
            dataloader,
            desc=(
                f"Personalization "
                f"[{self.mode}]"
            ),
            leave=False,
        )

        for batch in progress:

            batch = self._move_batch(
                batch
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.use_amp,
            ):

                logits, input_lengths = (
                    self._forward(
                        batch
                    )
                )

                loss = (
                    self._calculate_loss(
                        logits=logits,
                        input_lengths=(
                            input_lengths
                        ),
                        targets=(
                            batch[
                                "targets"
                            ]
                        ),
                        target_lengths=(
                            batch[
                                "target_lengths"
                            ]
                        ),
                        sample_weights=(
                            batch.get(
                                "sample_weights"
                            )
                        ),
                    )
                )

            self.scaler.scale(
                loss
            ).backward()

            if (
                self.gradient_clip_norm
                is not None
            ):

                self.scaler.unscale_(
                    self.optimizer
                )

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

            self.scaler.step(
                self.optimizer
            )

            self.scaler.update()

            total_loss += float(
                loss.item()
            )

            total_batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        return (
            total_loss
            / max(
                total_batches,
                1,
            )
        )

    # ========================================================
    # Validation / Test
    # ========================================================

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
    ) -> dict[str, Any]:
        """
        Validation / Test에서는 Error Profile weight를
        사용하지 않는다.

        평가 지표는 모든 실험에서 동일한
        일반 CTC loss / CER / WER로 계산한다.
        """

        original_mode = (
            self.mode
        )

        # 평가에서는 항상 baseline loss 사용
        self.mode = "baseline"

        try:

            result = super().evaluate(
                dataloader
            )

        finally:

            self.mode = (
                original_mode
            )

        return result