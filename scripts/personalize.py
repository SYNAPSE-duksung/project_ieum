from __future__ import annotations

import argparse
import json
import random
import sys

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# IEUM modules
# ============================================================

from src.asr.feature_extractor import (
    IEUMWhisperFeatureExtractor,
)

from src.asr.tokenizer import (
    CTCCharacterTokenizer,
)

from src.personalization.dataset import (
    PersonalizationIEUMDataset,
)

from src.personalization.model_loader import (
    load_general_model,
)

from src.personalization.trainer import (
    PersonalizationTrainer,
)


# ============================================================
# Seed
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# CTC target encoding
# ============================================================

def encode_targets(
    references: list[str],
    tokenizer: CTCCharacterTokenizer,
) -> dict[str, torch.Tensor]:

    encoded_targets = [
        tokenizer.encode(
            text
        )
        for text in references
    ]

    target_lengths = torch.tensor(
        [
            len(target)
            for target in encoded_targets
        ],
        dtype=torch.long,
    )

    flattened_targets = torch.tensor(
        [
            token
            for target in encoded_targets
            for token in target
        ],
        dtype=torch.long,
    )

    return {
        "targets": flattened_targets,
        "target_lengths": target_lengths,
    }


# ============================================================
# Raw CTC Collator
# ============================================================

class PersonalizationCollator:
    """
    개인화 학습용 Raw Audio Collator.

    waveform
        ↓
    Whisper Log-Mel Feature
        ↓
    CTC target encoding
    """

    def __init__(
        self,
        feature_extractor: IEUMWhisperFeatureExtractor,
        tokenizer: CTCCharacterTokenizer,
    ) -> None:

        self.feature_extractor = (
            feature_extractor
        )

        self.tokenizer = (
            tokenizer
        )

    def __call__(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:

        waveforms = [
            sample["waveform"]
            for sample in samples
        ]

        references = [
            sample["transcript"]
            for sample in samples
        ]

        feature_batch = (
            self.feature_extractor.batch(
                waveforms
            )
        )

        target_data = encode_targets(
            references=references,
            tokenizer=self.tokenizer,
        )

        batch = {
            "input_features": (
                feature_batch[
                    "input_features"
                ]
            ),
            "audio_num_samples": (
                feature_batch[
                    "audio_num_samples"
                ]
            ),
            "targets": (
                target_data[
                    "targets"
                ]
            ),
            "target_lengths": (
                target_data[
                    "target_lengths"
                ]
            ),
            "references": references,
        }

        # ----------------------------------------------------
        # Error Profile 단계에서 sample_weights가
        # dataset에 추가되면 자동으로 batch에 포함
        # ----------------------------------------------------

        if all(
            "sample_weight" in sample
            for sample in samples
        ):
            batch["sample_weights"] = (
                torch.tensor(
                    [
                        float(
                            sample[
                                "sample_weight"
                            ]
                        )
                        for sample in samples
                    ],
                    dtype=torch.float32,
                )
            )

        return batch


# ============================================================
# Dataset
# ============================================================

def create_dataset(
    csv_path: Path,
    audio_root: Path,
    speaker_id: str,
    split: str,
    sample_rate: int,
    max_audio_seconds: float,
) -> PersonalizationIEUMDataset:

    return PersonalizationIEUMDataset(
        csv_path=csv_path,
        audio_root=audio_root,
        speaker_id=speaker_id,
        split=split,
        split_column="personal_split",
        sample_rate=sample_rate,
        max_audio_seconds=max_audio_seconds,
        load_audio=True,
    )


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "IEUM speaker personalization"
        )
    )

    # --------------------------------------------------------
    # Speaker / data
    # --------------------------------------------------------

    parser.add_argument(
        "--speaker",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--audio_root",
        type=str,
        required=True,
    )

    # --------------------------------------------------------
    # General model
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="범용 모델 vocab.json",
    )

    parser.add_argument(
        "--extended_vocab",
        type=str,
        required=True,
    )

    # --------------------------------------------------------
    # Personalization
    # --------------------------------------------------------

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "baseline",
            "error_profile",
        ],
        default="baseline",
    )

    parser.add_argument(
        "--encoder_mode",
        type=str,
        choices=[
            "freeze",
            "last2",
            "last4",
            "full",
        ],
        default="freeze",
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
    )

    parser.add_argument(
        "--max_audio_seconds",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/personalization"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    # ========================================================
    # Seed / Device
    # ========================================================

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("IEUM 개인화 학습")
    print("=" * 70)

    print(
        f"Speaker      : {args.speaker}"
    )

    print(
        f"Mode         : {args.mode}"
    )

    print(
        f"Encoder mode : {args.encoder_mode}"
    )

    print(
        f"Device       : {device}"
    )

    # ========================================================
    # Paths
    # ========================================================

    csv_path = Path(
        args.csv
    )

    audio_root = Path(
        args.audio_root
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    vocab_path = Path(
        args.vocab
    )

    extended_vocab_path = Path(
        args.extended_vocab
    )

    # ========================================================
    # General → Personalization model
    # ========================================================

    print()
    print(
        "범용 모델 불러오는 중..."
    )

    model, tokenizer, _ = (
        load_general_model(
            checkpoint_path=(
                checkpoint_path
            ),
            original_vocab_path=(
                vocab_path
            ),
            extended_vocab_path=(
                extended_vocab_path
            ),
            device=device,
            encoder_train_mode=(
                args.encoder_mode
            ),
            sample_rate=(
                args.sample_rate
            ),
        )
    )

    print(
        f"Personalization vocab size: "
        f"{tokenizer.vocab_size}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    print()
    print(
        "화자 Dataset 생성 중..."
    )

    train_dataset = create_dataset(
        csv_path=csv_path,
        audio_root=audio_root,
        speaker_id=args.speaker,
        split="train",
        sample_rate=args.sample_rate,
        max_audio_seconds=(
            args.max_audio_seconds
        ),
    )

    valid_dataset = create_dataset(
        csv_path=csv_path,
        audio_root=audio_root,
        speaker_id=args.speaker,
        split="valid",
        sample_rate=args.sample_rate,
        max_audio_seconds=(
            args.max_audio_seconds
        ),
    )

    test_dataset = create_dataset(
        csv_path=csv_path,
        audio_root=audio_root,
        speaker_id=args.speaker,
        split="test",
        sample_rate=args.sample_rate,
        max_audio_seconds=(
            args.max_audio_seconds
        ),
    )

    print(
        f"Train chunks : "
        f"{len(train_dataset)}"
    )

    print(
        f"Valid chunks : "
        f"{len(valid_dataset)}"
    )

    print(
        f"Test chunks  : "
        f"{len(test_dataset)}"
    )

    if len(train_dataset) == 0:
        raise ValueError(
            "Train dataset이 비어 있습니다."
        )

    if len(valid_dataset) == 0:
        raise ValueError(
            "Valid dataset이 비어 있습니다."
        )

    if len(test_dataset) == 0:
        raise ValueError(
            "Test dataset이 비어 있습니다."
        )

    # ========================================================
    # Feature Extractor
    # ========================================================

    feature_extractor = (
        IEUMWhisperFeatureExtractor(
            model_name=(
                "openai/whisper-small"
            ),
            sample_rate=(
                args.sample_rate
            ),
            max_audio_seconds=(
                args.max_audio_seconds
            ),
        )
    )

    collator = (
        PersonalizationCollator(
            feature_extractor=(
                feature_extractor
            ),
            tokenizer=(
                tokenizer
            ),
        )
    )

    # ========================================================
    # DataLoader
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ========================================================
    # Trainable parameters
    # ========================================================

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    trainable_count = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    print()
    print(
        f"Trainable parameters: "
        f"{trainable_count:,}"
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    # ========================================================
    # Output
    # ========================================================

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir = (
        PROJECT_ROOT
        / args.output_root
        / args.mode
        / args.speaker
        / timestamp
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        output_dir
        / "extended_vocab.json"
    )

    # 실행 조건 저장
    run_config = vars(
        args
    ).copy()

    with (
        output_dir
        / "run_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            run_config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Output       : {output_dir}"
    )

    # ========================================================
    # Trainer
    # ========================================================

    trainer = PersonalizationTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        mode=args.mode,
        use_amp=True,
        gradient_clip_norm=1.0,
    )

    # ========================================================
    # Train
    # ========================================================

    summary = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        output_dir=output_dir,
        early_stopping_patience=(
            args.patience
        ),
    )

    # ========================================================
    # Best checkpoint 다시 로드
    # ========================================================

    best_model_path = (
        output_dir
        / "best_model.pt"
    )

    if not best_model_path.exists():
        raise FileNotFoundError(
            "개인화 best_model.pt가 "
            "생성되지 않았습니다."
        )

    best_checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # Test
    # ========================================================

    print()
    print("=" * 70)
    print("Best model Test 평가")
    print("=" * 70)

    test_result = trainer.evaluate(
        test_loader
    )

    print(
        f"Test CER: "
        f"{test_result['cer']:.4f}"
    )

    print(
        f"Test WER: "
        f"{test_result['wer']:.4f}"
    )

    # ========================================================
    # Test 결과 저장
    # ========================================================

    test_summary = {
        "speaker_id": args.speaker,
        "mode": args.mode,
        "best_epoch": (
            summary[
                "best_epoch"
            ]
        ),
        "best_valid_cer": (
            summary[
                "best_valid_cer"
            ]
        ),
        "test_loss": (
            test_result[
                "loss"
            ]
        ),
        "test_cer": (
            test_result[
                "cer"
            ]
        ),
        "test_wer": (
            test_result[
                "wer"
            ]
        ),
    }

    with (
        output_dir
        / "test_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            test_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(
        {
            "reference": (
                test_result[
                    "references"
                ]
            ),
            "prediction": (
                test_result[
                    "predictions"
                ]
            ),
        }
    ).to_csv(
        output_dir
        / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("개인화 학습 및 평가 완료")
    print("=" * 70)

    print(
        f"Speaker : {args.speaker}"
    )

    print(
        f"Best epoch : "
        f"{summary['best_epoch']}"
    )

    print(
        f"Valid CER  : "
        f"{summary['best_valid_cer']:.4f}"
    )

    print(
        f"Test CER   : "
        f"{test_result['cer']:.4f}"
    )

    print(
        f"Test WER   : "
        f"{test_result['wer']:.4f}"
    )

    print(
        f"Results    : "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()