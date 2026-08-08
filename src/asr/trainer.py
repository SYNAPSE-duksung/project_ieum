from __future__ import annotations

import json
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


class CTCASRModel(nn.Module):
    """
    Whisper Encoder + 후속 CTC 모델.

    Encoder fine-tuning 실험에서 사용한다.
    """

    def __init__(
        self,
        encoder: WhisperEncoder,
        downstream_model: nn.Module,
        sample_rate: int = 16000,
    ) -> None:

        super().__init__()

        self.encoder = encoder
        self.downstream_model = (
            downstream_model
        )

        self.sample_rate = (
            sample_rate
        )

    def forward(
        self,
        input_features: Tensor,
        audio_num_samples: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:

        hidden_states = (
            self.encoder(
                input_features
            )
        )

        input_lengths = (
            self.encoder
            .get_output_lengths(
                audio_num_samples=(
                    audio_num_samples
                ),
                sample_rate=(
                    self.sample_rate
                ),
            )
        )

        input_lengths = (
            input_lengths.clamp(
                min=1,
                max=(
                    hidden_states.shape[
                        1
                    ]
                ),
            )
        )

        logits = (
            self.downstream_model(
                hidden_states,
                lengths=(
                    input_lengths
                ),
            )
        )

        return (
            logits,
            input_lengths,
        )


class CachedCTCModel(nn.Module):
    """
    미리 계산된 Whisper Encoder hidden state를
    직접 후속 CTC 모델에 전달한다.

    Encoder Freeze 구조 비교에서 사용한다.
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
    ) -> tuple[
        Tensor,
        Tensor,
    ]:

        logits = (
            self.downstream_model(
                hidden_states,
                lengths=input_lengths,
            )
        )

        return (
            logits,
            input_lengths,
        )


class CTCTrainer:
    """
    IEUM ASR CTC 공통 Trainer.
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

        self.tokenizer = (
            tokenizer
        )

        self.optimizer = (
            optimizer
        )

        self.device = (
            device
        )

        self.use_cached_features = (
            use_cached_features
        )

        self.use_amp = (
            use_amp
            and device.type
            == "cuda"
        )

        self.gradient_clip_norm = (
            gradient_clip_norm
        )

        self.ctc_loss = (
            nn.CTCLoss(
                blank=(
                    tokenizer.blank_id
                ),
                reduction="mean",
                zero_infinity=True,
            )
        )

        # 최신 PyTorch 방식
        self.scaler = (
            torch.amp.GradScaler(
                "cuda",
                enabled=(
                    self.use_amp
                ),
            )
        )

    def _move_batch(
        self,
        batch: dict[
            str,
            Any,
        ],
    ) -> dict[
        str,
        Any,
    ]:

        if self.use_cached_features:

            batch[
                "hidden_states"
            ] = (
                batch[
                    "hidden_states"
                ]
                .to(
                    self.device,
                    non_blocking=True,
                )
                .float()
            )

            batch[
                "input_lengths"
            ] = (
                batch[
                    "input_lengths"
                ]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

        else:

            batch[
                "input_features"
            ] = (
                batch[
                    "input_features"
                ]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

            batch[
                "audio_num_samples"
            ] = (
                batch[
                    "audio_num_samples"
                ]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

        batch[
            "targets"
        ] = (
            batch[
                "targets"
            ]
            .to(
                self.device,
                non_blocking=True,
            )
        )

        batch[
            "target_lengths"
        ] = (
            batch[
                "target_lengths"
            ]
            .to(
                self.device,
                non_blocking=True,
            )
        )

        return batch

    def _forward(
        self,
        batch: dict[
            str,
            Any,
        ],
    ) -> tuple[
        Tensor,
        Tensor,
    ]:

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

    def _calculate_loss(
        self,
        logits: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:

        log_probs = (
            logits
            .log_softmax(
                dim=-1
            )
            .transpose(
                0,
                1,
            )
        )

        return self.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

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

            batch = (
                self._move_batch(
                    batch
                )
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=(
                    self.device.type
                ),
                dtype=(
                    torch.float16
                ),
                enabled=(
                    self.use_amp
                ),
            ):

                (
                    logits,
                    input_lengths,
                ) = self._forward(
                    batch
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

            current_loss = float(
                loss.item()
            )

            total_loss += (
                current_loss
            )

            total_batches += 1

            progress.set_postfix(
                loss=(
                    f"{current_loss:.4f}"
                )
            )

        return (
            total_loss
            / max(
                total_batches,
                1,
            )
        )

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> dict[
        str,
        Any,
    ]:

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        predictions: list[
            str
        ] = []

        references: list[
            str
        ] = []

        progress = tqdm(
            dataloader,
            desc="Valid",
            leave=False,
        )

        for batch in progress:

            batch = (
                self._move_batch(
                    batch
                )
            )

            (
                logits,
                input_lengths,
            ) = self._forward(
                batch
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

            current_loss = float(
                loss.item()
            )

            total_loss += (
                current_loss
            )

            total_batches += 1

            progress.set_postfix(
                loss=(
                    f"{current_loss:.4f}"
                )
            )

            predicted_ids = (
                logits.argmax(
                    dim=-1
                )
            )

            for index in range(
                predicted_ids.shape[
                    0
                ]
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
                batch[
                    "references"
                ]
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

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        *,
        epochs: int,
        output_dir: str | Path,
        early_stopping_patience: int = 5,
    ) -> dict[
        str,
        Any,
    ]:

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        history: list[
            dict[
                str,
                Any,
            ]
        ] = []

        best_cer = float(
            "inf"
        )

        best_epoch = 0
        patience_counter = 0

        for epoch in range(
            1,
            epochs + 1,
        ):

            print()
            print("=" * 70)
            print(
                f"Epoch {epoch}/{epochs}"
            )
            print("=" * 70)

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

            epoch_result = {
                "epoch": epoch,
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

            pd.DataFrame(
                history
            ).to_csv(
                output_dir
                / "history.csv",
                index=False,
                encoding="utf-8-sig",
            )

            if (
                valid_result[
                    "cer"
                ]
                < best_cer
            ):

                best_cer = (
                    valid_result[
                        "cer"
                    ]
                )

                best_epoch = (
                    epoch
                )

                patience_counter = 0

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
                ).to_csv(
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

            if (
                patience_counter
                >= early_stopping_patience
            ):

                print(
                    "Early stopping"
                )

                break

        summary = {
            "best_epoch": (
                best_epoch
            ),
            "best_valid_cer": (
                best_cer
            ),
            "epochs_completed": (
                len(
                    history
                )
            ),
        }

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

        return summary