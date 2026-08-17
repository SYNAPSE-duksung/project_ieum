from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from jiwer import cer, wer
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.asr.encoder import WhisperEncoder
from src.asr.tokenizer import CTCCharacterTokenizer


# ============================================================
# Model wrappers
# ============================================================


class CTCASRModel(nn.Module):
    """
    Whisper Encoder + Downstream CTC 모델.

    Encoder를 직접 실행하는 일반 학습용 모델.
    Encoder fine-tuning 단계에서도 사용할 수 있다.
    """

    def __init__(
        self,
        encoder: WhisperEncoder,
        downstream_model: nn.Module,
        sample_rate: int = 16000,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.downstream_model = downstream_model
        self.sample_rate = int(sample_rate)

    def forward(
        self,
        input_features: Tensor,
        audio_num_samples: Tensor,
    ) -> tuple[Tensor, Tensor]:

        hidden_states = self.encoder(
            input_features
        )

        input_lengths = (
            self.encoder.get_output_lengths(
                audio_num_samples=audio_num_samples,
                sample_rate=self.sample_rate,
            )
        )

        input_lengths = input_lengths.clamp(
            min=1,
            max=hidden_states.shape[1],
        )

        logits = self.downstream_model(
            hidden_states,
            lengths=input_lengths,
        )

        return logits, input_lengths


class CachedCTCModel(nn.Module):
    """
    미리 저장된 Whisper Encoder hidden state를 입력받는 모델.

    Encoder Freeze 상태의 downstream 구조 비교에서 사용한다.
    """

    def __init__(
        self,
        downstream_model: nn.Module,
    ) -> None:
        super().__init__()

        self.downstream_model = (
            downstream_model
        )

    def forward(
        self,
        hidden_states: Tensor,
        input_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:

        logits = self.downstream_model(
            hidden_states,
            lengths=input_lengths,
        )

        return logits, input_lengths


# ============================================================
# Trainer
# ============================================================


class CTCTrainer:
    """
    IEUM ASR CTC 공통 Trainer.

    지원:
    - Linear CTC
    - BiGRU CTC
    - Transformer CTC
    - Conformer CTC

    추가 기능:
    - Feature cache 학습
    - AMP
    - Gradient clipping
    - Early stopping
    - Best checkpoint
    - Epoch resume checkpoint
    - 완료된 모델 자동 skip
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: CTCCharacterTokenizer,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        *,
        use_amp: bool = True,
        gradient_clip_norm: float = 1.0,
        use_cached_features: bool = False,
    ) -> None:

        self.model = model.to(
            device
        )

        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device

        self.use_cached_features = bool(
            use_cached_features
        )

        self.use_amp = (
            bool(use_amp)
            and device.type == "cuda"
        )

        self.gradient_clip_norm = (
            gradient_clip_norm
        )

        self.ctc_loss = nn.CTCLoss(
            blank=tokenizer.blank_id,
            reduction="mean",
            zero_infinity=True,
        )

        # PyTorch 최신 방식
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

    # ========================================================
    # Batch
    # ========================================================

    def _move_batch(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:

        if self.use_cached_features:

            batch["hidden_states"] = (
                batch["hidden_states"]
                .to(
                    self.device,
                    non_blocking=True,
                )
                .float()
            )

            batch["input_lengths"] = (
                batch["input_lengths"]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

        else:

            batch["input_features"] = (
                batch["input_features"]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

            batch["audio_num_samples"] = (
                batch["audio_num_samples"]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

        batch["targets"] = (
            batch["targets"]
            .to(
                self.device,
                non_blocking=True,
            )
        )

        batch["target_lengths"] = (
            batch["target_lengths"]
            .to(
                self.device,
                non_blocking=True,
            )
        )

        return batch

    def _forward(
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, Tensor]:

        if self.use_cached_features:

            return self.model(
                hidden_states=(
                    batch[
                        "hidden_states"
                    ]
                ),
                input_lengths=(
                    batch[
                        "input_lengths"
                    ]
                ),
            )

        return self.model(
            input_features=(
                batch[
                    "input_features"
                ]
            ),
            audio_num_samples=(
                batch[
                    "audio_num_samples"
                ]
            ),
        )

    # ========================================================
    # Loss
    # ========================================================

    def _calculate_loss(
        self,
        logits: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:

        log_probs = logits.log_softmax(
            dim=-1
        )

        # CTCLoss:
        # [B, T, C] -> [T, B, C]
        log_probs = log_probs.transpose(
            0,
            1,
        )

        loss = self.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        return loss

    # ========================================================
    # Train epoch
    # ========================================================

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:

        self.model.train()

        total_loss = 0.0
        total_batches = 0

        progress = tqdm(
            dataloader,
            desc="Train",
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
        dataloader: DataLoader,
    ) -> dict[str, Any]:

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        predictions: list[str] = []
        references: list[str] = []

        progress = tqdm(
            dataloader,
            desc="Eval",
            leave=False,
        )

        for batch in progress:

            batch = self._move_batch(
                batch
            )

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
                )
            )

            total_loss += float(
                loss.item()
            )

            total_batches += 1

            predicted_ids = (
                logits.argmax(
                    dim=-1
                )
            )

            for index in range(
                predicted_ids.shape[0]
            ):

                length = int(
                    input_lengths[
                        index
                    ].item()
                )

                token_ids = (
                    predicted_ids[
                        index,
                        :length,
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )

                prediction = (
                    self.tokenizer.decode(
                        token_ids,
                        ctc_decode=True,
                    )
                )

                predictions.append(
                    prediction
                )

            references.extend(
                batch["references"]
            )

        average_loss = (
            total_loss
            / max(
                total_batches,
                1,
            )
        )

        current_cer = cer(
            references,
            predictions,
        )

        current_wer = wer(
            references,
            predictions,
        )

        return {
            "loss": (
                average_loss
            ),
            "cer": (
                current_cer
            ),
            "wer": (
                current_wer
            ),
            "predictions": (
                predictions
            ),
            "references": (
                references
            ),
        }

    # ========================================================
    # Resume state directory
    # ========================================================

    def _get_resume_dir(
        self,
        output_dir: Path,
        resume_key: str | None = None,
    ) -> Path:
        """
        Resume checkpoint 저장 폴더를 반환한다.

        기본 학습:
            기존 방식 그대로 사용

        개인화/하이퍼파라미터 실험:
            resume_key를 지정하여
            서로 다른 실험의 resume가 섞이지 않게 한다.
        """

        downstream = getattr(
            self.model,
            "downstream_model",
            self.model,
        )

        model_name = (
            downstream
            .__class__
            .__name__
        )

        resume_dir = (
            output_dir.parent
            / "_resume"
        )

        if resume_key is not None:
            resume_dir = (
                resume_dir
                / resume_key
            )

        resume_dir = (
            resume_dir
            / model_name
        )

        resume_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return resume_dir

    # ========================================================
    # Checkpoint save
    # ========================================================

    def _save_last_checkpoint(
        self,
        *,
        checkpoint_path: Path,
        epoch: int,
        history: list[dict[str, Any]],
        best_cer: float,
        best_epoch: int,
        patience_counter: int,
        total_training_seconds: float,
    ) -> None:

        checkpoint = {
            "epoch": int(
                epoch
            ),

            "model_state_dict": (
                self.model.state_dict()
            ),

            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),

            "scaler_state_dict": (
                self.scaler.state_dict()
            ),

            "history": (
                history
            ),

            "best_cer": float(
                best_cer
            ),

            "best_epoch": int(
                best_epoch
            ),

            "patience_counter": int(
                patience_counter
            ),

            "total_training_seconds": float(
                total_training_seconds
            ),
        }

        temp_path = (
            checkpoint_path
            .with_suffix(
                ".tmp"
            )
        )

        torch.save(
            checkpoint,
            temp_path,
        )

        temp_path.replace(
            checkpoint_path
        )

    # ========================================================
    # Resume load
    # ========================================================

    def _load_resume_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> dict[str, Any] | None:

        if not checkpoint_path.exists():

            return None

        print()
        print("=" * 70)
        print(
            "이전 학습 Checkpoint 발견"
        )
        print("=" * 70)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if (
            "scaler_state_dict"
            in checkpoint
        ):

            self.scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        completed_epoch = int(
            checkpoint[
                "epoch"
            ]
        )

        print(
            f"완료된 마지막 Epoch: "
            f"{completed_epoch}"
        )

        print(
            f"다음 시작 Epoch: "
            f"{completed_epoch + 1}"
        )

        return checkpoint

    # ========================================================
    # Fit
    # ========================================================

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        *,
        epochs: int,
        output_dir: str | Path,
        early_stopping_patience: int = 5,
        resume_key: str | None = None,
    ) -> dict[str, Any]:

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        resume_dir = (
            self._get_resume_dir(
                output_dir,
                resume_key=resume_key,
            )
        )

        checkpoint_path = (
            resume_dir
            / "last_checkpoint.pt"
        )

        completed_path = (
            resume_dir
            / "completed.json"
        )

        # ====================================================
        # 이미 완료된 모델이면 자동 Skip
        # ====================================================

        if completed_path.exists():

            with completed_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                completed_summary = (
                    json.load(
                        file
                    )
                )

            print()
            print("=" * 70)
            print(
                "이미 완료된 모델입니다."
            )
            print("=" * 70)

            print(
                "학습을 건너뜁니다."
            )

            print(
                f"Best epoch: "
                f"{completed_summary['best_epoch']}"
            )

            print(
                f"Best CER: "
                f"{completed_summary['best_valid_cer']:.4f}"
            )

            return (
                completed_summary
            )

        # ====================================================
        # 기본 상태
        # ====================================================

        history: list[
            dict[str, Any]
        ] = []

        best_cer = float(
            "inf"
        )

        best_epoch = 0

        patience_counter = 0

        start_epoch = 1

        previous_training_seconds = (
            0.0
        )

        # ====================================================
        # Resume
        # ====================================================

        checkpoint = (
            self._load_resume_checkpoint(
                checkpoint_path
            )
        )

        if checkpoint is not None:

            start_epoch = (
                int(
                    checkpoint[
                        "epoch"
                    ]
                )
                + 1
            )

            history = list(
                checkpoint.get(
                    "history",
                    [],
                )
            )

            best_cer = float(
                checkpoint.get(
                    "best_cer",
                    float("inf"),
                )
            )

            best_epoch = int(
                checkpoint.get(
                    "best_epoch",
                    0,
                )
            )

            patience_counter = int(
                checkpoint.get(
                    "patience_counter",
                    0,
                )
            )

            previous_training_seconds = float(
                checkpoint.get(
                    "total_training_seconds",
                    0.0,
                )
            )

        # ====================================================
        # 이미 필요한 epoch까지 끝났지만
        # completed 파일만 없는 경우
        # ====================================================

        if start_epoch > epochs:

            print(
                "Checkpoint 기준으로 "
                "모든 Epoch가 이미 완료되었습니다."
            )

            summary = {
                "best_epoch": (
                    best_epoch
                ),

                "best_valid_cer": (
                    best_cer
                ),

                "epochs_completed": (
                    len(history)
                ),

                "total_training_seconds": (
                    previous_training_seconds
                ),

                "completed": True,
            }

            with completed_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    summary,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            return summary

        # ====================================================
        # Train
        # ====================================================

        run_start = (
            time.perf_counter()
        )

        stopped_early = False

        for epoch in range(
            start_epoch,
            epochs + 1,
        ):

            print()
            print("=" * 70)
            print(
                f"Epoch {epoch}/{epochs}"
            )
            print("=" * 70)

            epoch_start = (
                time.perf_counter()
            )

            train_loss = (
                self.train_epoch(
                    train_loader
                )
            )

            valid_result = (
                self.evaluate(
                    valid_loader
                )
            )

            epoch_seconds = (
                time.perf_counter()
                - epoch_start
            )

            epoch_result = {
                "epoch": (
                    epoch
                ),

                "train_loss": (
                    train_loss
                ),

                "valid_loss": (
                    valid_result[
                        "loss"
                    ]
                ),

                "valid_cer": (
                    valid_result[
                        "cer"
                    ]
                ),

                "valid_wer": (
                    valid_result[
                        "wer"
                    ]
                ),

                "epoch_seconds": (
                    epoch_seconds
                ),

                "epoch_minutes": (
                    epoch_seconds
                    / 60.0
                ),
            }

            history.append(
                epoch_result
            )

            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Valid Loss: "
                f"{valid_result['loss']:.4f}"
            )

            print(
                f"Valid CER: "
                f"{valid_result['cer']:.4f}"
            )

            print(
                f"Valid WER: "
                f"{valid_result['wer']:.4f}"
            )

            print(
                f"Epoch Time: "
                f"{epoch_seconds / 60.0:.2f}분"
            )

            # ================================================
            # History
            # ================================================

            pd.DataFrame(
                history
            ).to_csv(
                output_dir
                / "history.csv",
                index=False,
                encoding="utf-8-sig",
            )

            # ================================================
            # Best model
            # ================================================

            if (
                valid_result[
                    "cer"
                ]
                < best_cer
            ):

                best_cer = float(
                    valid_result[
                        "cer"
                    ]
                )

                best_epoch = (
                    epoch
                )

                patience_counter = (
                    0
                )

                torch.save(
                    {
                        "epoch": (
                            epoch
                        ),

                        "model_state_dict": (
                            self.model
                            .state_dict()
                        ),

                        "optimizer_state_dict": (
                            self.optimizer
                            .state_dict()
                        ),

                        "valid_cer": (
                            valid_result[
                                "cer"
                            ]
                        ),

                        "valid_wer": (
                            valid_result[
                                "wer"
                            ]
                        ),
                    },

                    output_dir
                    / "best_model.pt",
                )

                prediction_dataframe = (
                    pd.DataFrame(
                        {
                            "reference": (
                                valid_result[
                                    "references"
                                ]
                            ),

                            "prediction": (
                                valid_result[
                                    "predictions"
                                ]
                            ),
                        }
                    )
                )

                prediction_dataframe.to_csv(
                    output_dir
                    / "best_valid_predictions.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                print(
                    "Best model 저장 완료"
                )

            else:

                patience_counter += 1

                print(
                    "CER 개선 없음 "
                    f"({patience_counter}/"
                    f"{early_stopping_patience})"
                )

            # ================================================
            # Resume checkpoint
            #
            # 매 Epoch 종료 후 반드시 저장
            # ================================================

            current_training_seconds = (
                previous_training_seconds
                + (
                    time.perf_counter()
                    - run_start
                )
            )

            self._save_last_checkpoint(
                checkpoint_path=(
                    checkpoint_path
                ),
                epoch=epoch,
                history=history,
                best_cer=best_cer,
                best_epoch=best_epoch,
                patience_counter=(
                    patience_counter
                ),
                total_training_seconds=(
                    current_training_seconds
                ),
            )

            print(
                "Resume checkpoint 저장 완료"
            )

            # ================================================
            # Early stopping
            # ================================================

            if (
                patience_counter
                >= early_stopping_patience
            ):

                print(
                    "Early stopping"
                )

                stopped_early = (
                    True
                )

                break

        # ====================================================
        # Final summary
        # ====================================================

        total_training_seconds = (
            previous_training_seconds
            + (
                time.perf_counter()
                - run_start
            )
        )

        summary = {
            "best_epoch": (
                best_epoch
            ),

            "best_valid_cer": (
                best_cer
            ),

            "epochs_completed": (
                len(history)
            ),

            "total_training_seconds": (
                total_training_seconds
            ),

            "total_training_minutes": (
                total_training_seconds
                / 60.0
            ),

            "early_stopped": (
                stopped_early
            ),

            "completed": True,
        }

        # 기존 결과 폴더
        with (
            output_dir
            / "summary.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # 고정 resume 폴더
        with completed_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print()
        print("=" * 70)
        print(
            "학습 완료"
        )
        print("=" * 70)

        print(
            f"Best Epoch: "
            f"{best_epoch}"
        )

        print(
            f"Best CER: "
            f"{best_cer:.4f}"
        )

        print(
            f"총 학습 시간: "
            f"{total_training_seconds / 60.0:.2f}분"
        )

        return summary