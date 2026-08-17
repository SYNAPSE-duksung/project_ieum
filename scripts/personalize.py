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

from src.personalization.error_profile import (
    build_raw_error_profile,
    filter_error_profile,
)

from src.personalization.weighting import (
    calculate_sample_weights,
)


# ============================================================
# Seed
# ============================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CTC target encoding
# ============================================================

def encode_targets(
    references: list[str],
    tokenizer: CTCCharacterTokenizer,
) -> dict[str, torch.Tensor]:

    encoded_targets = [
        tokenizer.encode(text)
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

    Error Profile mode에서는 sample_weight가 존재할 경우
    batch["sample_weights"]로 함께 전달한다.
    """

    def __init__(
        self,
        feature_extractor: IEUMWhisperFeatureExtractor,
        tokenizer: CTCCharacterTokenizer,
    ) -> None:

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

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

        feature_batch = self.feature_extractor.batch(
            waveforms
        )

        target_data = encode_targets(
            references=references,
            tokenizer=self.tokenizer,
        )

        batch = {
            "input_features": (
                feature_batch["input_features"]
            ),
            "audio_num_samples": (
                feature_batch["audio_num_samples"]
            ),
            "targets": (
                target_data["targets"]
            ),
            "target_lengths": (
                target_data["target_lengths"]
            ),
            "references": references,
        }

        # Error Profile mode에서만 존재한다.
        if all(
            "sample_weight" in sample
            for sample in samples
        ):
            batch["sample_weights"] = torch.tensor(
                [
                    float(
                        sample["sample_weight"]
                    )
                    for sample in samples
                ],
                dtype=torch.float32,
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
# General model prediction
# ============================================================

@torch.no_grad()
def predict_train_dataset(
    model: torch.nn.Module,
    data_loader: DataLoader,
    tokenizer: CTCCharacterTokenizer,
    device: torch.device,
    original_vocab_size: int,
) -> list[dict[str, Any]]:
    """
    개인화 Fine-tuning 전에 현재 범용 모델로
    화자의 train 데이터 prediction을 생성한다.

    중요
    ----
    개인화 모델 자체는 extended vocab 크기로 생성되어 있지만,
    Error Profile은 '개인화 전 범용 모델의 오류'를 분석하는 것이므로
    prediction에서는 기존 범용 vocab 범위의 logits만 사용한다.

    즉:
        Error Profile prediction → original vocab
        Personalization training → extended vocab
    """

    if original_vocab_size <= 0:
        raise ValueError(
            "original_vocab_size는 1 이상이어야 합니다."
        )

    model.eval()

    results: list[dict[str, Any]] = []

    for batch in data_loader:

        input_features = batch[
            "input_features"
        ].to(
            device,
            non_blocking=True,
        )

        audio_num_samples = batch[
            "audio_num_samples"
        ].to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # CTCASRModel.forward()
        #
        # 반환:
        #   logits, input_lengths
        # ----------------------------------------------------

        logits, input_lengths = model(
            input_features=input_features,
            audio_num_samples=audio_num_samples,
        )

        if logits.ndim != 3:
            raise RuntimeError(
                "예상하지 못한 logits shape입니다.\n"
                f"logits.shape = {tuple(logits.shape)}"
            )

        if logits.shape[-1] < original_vocab_size:
            raise RuntimeError(
                "모델 classifier 크기가 기존 vocab보다 작습니다.\n"
                f"classifier size   : {logits.shape[-1]}\n"
                f"original vocab   : {original_vocab_size}"
            )

        # ----------------------------------------------------
        # Error Profile 생성에서는 기존 범용 vocab만 사용
        #
        # 개인화 모델:
        #   0 ~ 823     : 범용 모델에서 학습된 weight
        #   824 ~ 1094 : 새로 초기화된 weight
        #
        # 새 classifier row는 아직 학습되지 않았으므로
        # 범용 모델 Error Profile prediction에는 포함하지 않는다.
        # ----------------------------------------------------

        general_logits = logits[
            ...,
            :original_vocab_size,
        ]

        predicted_ids = torch.argmax(
            general_logits,
            dim=-1,
        )

        references = batch[
            "references"
        ]

        if len(references) != predicted_ids.shape[0]:
            raise RuntimeError(
                "Reference 수와 prediction batch 크기가 다릅니다."
            )

        # ----------------------------------------------------
        # CTC decode
        #
        # padding 영역까지 decode하지 않고
        # 실제 encoder output length까지만 사용한다.
        # ----------------------------------------------------

        for index, reference in enumerate(
            references
        ):

            length = int(
                input_lengths[
                    index
                ].item()
            )

            length = max(
                1,
                min(
                    length,
                    predicted_ids.shape[1],
                ),
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

            prediction = tokenizer.decode(
                token_ids,
                ctc_decode=True,
            )

            results.append(
                {
                    "reference_text": (
                        reference
                    ),
                    "prediction_text": (
                        prediction
                    ),
                }
            )

    return results


# ============================================================
# Error Profile preparation
# ============================================================

def prepare_error_profile(
    *,
    model: torch.nn.Module,
    train_dataset: PersonalizationIEUMDataset,
    feature_extractor: IEUMWhisperFeatureExtractor,
    tokenizer: CTCCharacterTokenizer,
    device: torch.device,
    original_vocab_size: int,
    batch_size: int,
    num_workers: int,
    min_count: int,
    min_ratio: float,
    alpha: float,
    max_weight: float | None,
    output_dir: Path,
) -> None:
    """
    train 데이터만 사용하여

    1. 범용 모델 prediction 생성
    2. Raw Error Profile 생성
    3. threshold 적용
    4. train sample별 weight 계산
    5. Dataset에 weight 연결

    을 수행한다.

    valid/test 데이터는 Error Profile 생성에 사용하지 않는다.
    """

    print()
    print("=" * 70)
    print("Error Profile 생성")
    print("=" * 70)

    print(
        f"min_count : {min_count}"
    )

    print(
        f"min_ratio : {min_ratio}"
    )

    print(
        f"alpha     : {alpha}"
    )

    print(
        f"max_weight: {max_weight}"
    )

    # --------------------------------------------------------
    # Prediction loader
    #
    # 반드시 shuffle=False.
    # Dataset index와 prediction 순서를 유지해야
    # weight를 정확한 sample에 연결할 수 있다.
    # --------------------------------------------------------

    prediction_collator = (
        PersonalizationCollator(
            feature_extractor=(
                feature_extractor
            ),
            tokenizer=tokenizer,
        )
    )

    prediction_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=prediction_collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # General model prediction
    # --------------------------------------------------------

    print()
    print(
        "범용 모델로 train 데이터 추론 중..."
    )

    prediction_rows = (
        predict_train_dataset(
            model=model,
            data_loader=prediction_loader,
            tokenizer=tokenizer,
            device=device,
            original_vocab_size=(
                original_vocab_size
            ),
        )
    )

    if (
        len(prediction_rows)
        != len(train_dataset)
    ):
        raise RuntimeError(
            "Train prediction 수와 "
            "Dataset sample 수가 다릅니다.\n"
            f"Predictions: {len(prediction_rows)}\n"
            f"Dataset: {len(train_dataset)}"
        )

    sample_ids = (
        train_dataset.samples[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Prediction 저장
    # --------------------------------------------------------

    prediction_save_rows = []

    for sample_id, row in zip(
        sample_ids,
        prediction_rows,
    ):
        prediction_save_rows.append(
            {
                "sample_id": sample_id,
                "reference_text": (
                    row["reference_text"]
                ),
                "prediction_text": (
                    row["prediction_text"]
                ),
            }
        )

    pd.DataFrame(
        prediction_save_rows
    ).to_csv(
        output_dir
        / "train_general_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Reference / Prediction pairs
    # --------------------------------------------------------

    reference_prediction_pairs = [
        (
            row["reference_text"],
            row["prediction_text"],
        )
        for row in prediction_rows
    ]

    # --------------------------------------------------------
    # Raw Error Profile
    # --------------------------------------------------------

    print(
        "Raw Error Profile 생성 중..."
    )

    raw_profile = (
        build_raw_error_profile(
            reference_prediction_pairs
        )
    )

    raw_profile_df = pd.DataFrame(
        raw_profile
    )

    raw_profile_df.to_csv(
        output_dir
        / "raw_error_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    filtered_profile = (
        filter_error_profile(
            raw_profile,
            min_count=min_count,
            min_ratio=min_ratio,
        )
    )

    filtered_profile_df = pd.DataFrame(
        filtered_profile
    )

    filtered_profile_df.to_csv(
        output_dir
        / "filtered_error_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Raw profile errors      : "
        f"{len(raw_profile)}"
    )

    print(
        f"Filtered profile errors : "
        f"{len(filtered_profile)}"
    )

    # --------------------------------------------------------
    # Sample weights
    # --------------------------------------------------------

    print(
        "Train sample weight 계산 중..."
    )

    weight_rows = (
        calculate_sample_weights(
            reference_prediction_pairs,
            filtered_profile,
            alpha=alpha,
            base_weight=1.0,
            max_weight=max_weight,
        )
    )

    if (
        len(weight_rows)
        != len(train_dataset)
    ):
        raise RuntimeError(
            "계산된 sample weight 수와 "
            "Dataset sample 수가 다릅니다.\n"
            f"Weights: {len(weight_rows)}\n"
            f"Dataset: {len(train_dataset)}"
        )

    sample_weights = [
        float(
            row["sample_weight"]
        )
        for row in weight_rows
    ]

    # --------------------------------------------------------
    # Dataset에 weight 연결
    # --------------------------------------------------------

    train_dataset.set_sample_weights(
        sample_weights
    )

    # --------------------------------------------------------
    # Weight 결과 저장
    # --------------------------------------------------------

    weight_save_rows = []

    for sample_id, row in zip(
        sample_ids,
        weight_rows,
    ):

        weight_save_rows.append(
            {
                "sample_id": (
                    sample_id
                ),
                "reference_text": (
                    row["reference_text"]
                ),
                "prediction_text": (
                    row["prediction_text"]
                ),
                "num_profile_errors": (
                    row[
                        "num_profile_errors"
                    ]
                ),
                "profile_ratio_sum": (
                    row[
                        "profile_ratio_sum"
                    ]
                ),
                "sample_weight": (
                    row[
                        "sample_weight"
                    ]
                ),
                "matched_errors": (
                    json.dumps(
                        row[
                            "matched_errors"
                        ],
                        ensure_ascii=False,
                    )
                ),
            }
        )

    pd.DataFrame(
        weight_save_rows
    ).to_csv(
        output_dir
        / "train_sample_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Weight summary
    # --------------------------------------------------------

    weights_array = np.asarray(
        sample_weights,
        dtype=float,
    )

    weighted_sample_count = int(
        (
            weights_array > 1.0
        ).sum()
    )

    print()
    print("-" * 70)
    print("Error Profile 요약")
    print("-" * 70)

    print(
        f"Train samples    : "
        f"{len(train_dataset)}"
    )

    print(
        f"Profile errors   : "
        f"{len(filtered_profile)}"
    )

    print(
        f"Weighted samples : "
        f"{weighted_sample_count}"
        f" / {len(train_dataset)}"
    )

    print(
        f"Weight min       : "
        f"{weights_array.min():.4f}"
    )

    print(
        f"Weight mean      : "
        f"{weights_array.mean():.4f}"
    )

    print(
        f"Weight max       : "
        f"{weights_array.max():.4f}"
    )

    print("-" * 70)


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
    # Error Profile
    # --------------------------------------------------------

    parser.add_argument(
        "--min_count",
        type=int,
        default=2,
        help=(
            "Error Profile에 포함할 "
            "최소 오류 반복 횟수"
        ),
    )

    parser.add_argument(
        "--min_ratio",
        type=float,
        default=0.0,
        help=(
            "Error Profile에 포함할 "
            "최소 오류 비율"
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Error Profile sample "
            "weight 강도"
        ),
    )

    parser.add_argument(
        "--max_weight",
        type=float,
        default=None,
        help=(
            "sample weight 최대값. "
            "미지정 시 제한 없음"
        ),
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
    # Argument validation
    # ========================================================

    if args.min_count < 1:
        raise ValueError(
            "--min_count는 1 이상이어야 합니다."
        )

    if not (
        0.0
        <= args.min_ratio
        <= 1.0
    ):
        raise ValueError(
            "--min_ratio는 0~1 사이여야 합니다."
        )

    if args.alpha < 0:
        raise ValueError(
            "--alpha는 0 이상이어야 합니다."
        )

    if (
        args.max_weight is not None
        and args.max_weight < 1.0
    ):
        raise ValueError(
            "--max_weight는 1.0 이상이어야 합니다."
        )

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
    # Original vocab size
    #
    # Error Profile 생성에서는 범용 모델이 실제로
    # 학습했던 기존 vocab 범위만 사용한다.
    # ========================================================

    with vocab_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        original_vocab = json.load(
            file
        )

    original_vocab_size = len(
        original_vocab
    )

    print()
    print(
        f"Original vocab size for "
        f"Error Profile: "
        f"{original_vocab_size}"
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
    # Feature Extractor / Collator
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
            tokenizer=tokenizer,
        )
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

    print(
        f"Output       : {output_dir}"
    )

    # ========================================================
    # Error Profile
    #
    # 중요:
    # Fine-tuning 전에 실행한다.
    # ========================================================

    if args.mode == "error_profile":

        prepare_error_profile(
            model=model,
            train_dataset=train_dataset,
            feature_extractor=(
                feature_extractor
            ),
            tokenizer=tokenizer,
            device=device,
            original_vocab_size=(
                original_vocab_size
            ),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            min_count=args.min_count,
            min_ratio=args.min_ratio,
            alpha=args.alpha,
            max_weight=args.max_weight,
            output_dir=output_dir,
        )

    # ========================================================
    # Tokenizer / Config 저장
    # ========================================================

    tokenizer.save(
        output_dir
        / "extended_vocab.json"
    )

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

    # ========================================================
    # DataLoader
    #
    # Error Profile mode라면 이 시점에는
    # train_dataset에 sample_weight가 이미 들어 있다.
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
    #
    # Error Profile prediction은 optimizer 생성 전에
    # 수행했기 때문에 모델 parameter는 변경되지 않는다.
    # ========================================================

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
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
            summary["best_epoch"]
        ),
        "best_valid_cer": (
            summary["best_valid_cer"]
        ),
        "test_loss": (
            test_result["loss"]
        ),
        "test_cer": (
            test_result["cer"]
        ),
        "test_wer": (
            test_result["wer"]
        ),
    }

    # Error Profile mode에서는
    # 실험 파라미터도 결과에 함께 기록한다.
    if args.mode == "error_profile":

        test_summary.update(
            {
                "min_count": (
                    args.min_count
                ),
                "min_ratio": (
                    args.min_ratio
                ),
                "alpha": (
                    args.alpha
                ),
                "max_weight": (
                    args.max_weight
                ),
            }
        )

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

    # ========================================================
    # Complete
    # ========================================================

    print()
    print("=" * 70)
    print("개인화 학습 및 평가 완료")
    print("=" * 70)

    print(
        f"Speaker : {args.speaker}"
    )

    print(
        f"Mode    : {args.mode}"
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