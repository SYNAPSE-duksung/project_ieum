from __future__ import annotations

import argparse
import random
import sys

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch.utils.data import DataLoader


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.asr.config import (
    load_config,
    resolve_data_paths,
)

from src.asr.dataset import (
    IEUMDataset,
)

from src.asr.encoder import (
    WhisperEncoder,
)

from src.asr.feature_cache import (
    CachedFeatureDataset,
    build_feature_cache,
)

from src.asr.feature_extractor import (
    IEUMWhisperFeatureExtractor,
)

from src.asr.models import (
    BiGRUCTC,
    ConformerCTC,
    LinearCTC,
    TransformerCTC,
)

from src.asr.tokenizer import (
    CTCCharacterTokenizer,
)

from src.asr.trainer import (
    CachedCTCModel,
    CTCASRModel,
    CTCTrainer,
)


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


class RawCTCCollator:
    """
    Encoder fine-tuning용 collator.
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
        samples: list[
            dict[
                str,
                Any,
            ]
        ],
    ) -> dict[
        str,
        Any,
    ]:

        waveforms = [
            sample[
                "waveform"
            ]
            for sample in samples
        ]

        references = [
            sample[
                "transcript"
            ]
            for sample in samples
        ]

        feature_batch = (
            self.feature_extractor
            .batch(
                waveforms
            )
        )

        target_data = (
            encode_targets(
                references,
                self.tokenizer,
            )
        )

        return {
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
            "references": (
                references
            ),
        }


class CachedCTCCollator:
    """
    Whisper Encoder feature cache용 collator.
    """

    def __init__(
        self,
        tokenizer: CTCCharacterTokenizer,
    ) -> None:

        self.tokenizer = (
            tokenizer
        )

    def __call__(
        self,
        samples: list[
            dict[
                str,
                Any,
            ]
        ],
    ) -> dict[
        str,
        Any,
    ]:

        references = [
            sample[
                "transcript"
            ]
            for sample in samples
        ]

        input_lengths = (
            torch.tensor(
                [
                    sample[
                        "input_length"
                    ]
                    for sample
                    in samples
                ],
                dtype=(
                    torch.long
                ),
            )
        )

        max_length = int(
            input_lengths.max()
            .item()
        )

        hidden_size = int(
            samples[
                0
            ][
                "hidden_states"
            ].shape[
                -1
            ]
        )

        hidden_states = (
            torch.zeros(
                (
                    len(samples),
                    max_length,
                    hidden_size,
                ),
                dtype=(
                    torch.float32
                ),
            )
        )

        for index, sample in enumerate(
            samples
        ):

            length = int(
                sample[
                    "input_length"
                ]
            )

            hidden_states[
                index,
                :length,
            ] = (
                sample[
                    "hidden_states"
                ]
                .float()
            )

        target_data = (
            encode_targets(
                references,
                self.tokenizer,
            )
        )

        return {
            "hidden_states": (
                hidden_states
            ),
            "input_lengths": (
                input_lengths
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
            "references": (
                references
            ),
        }


def encode_targets(
    references: list[
        str
    ],
    tokenizer: CTCCharacterTokenizer,
) -> dict[
    str,
    torch.Tensor,
]:

    encoded_targets = [
        tokenizer.encode(
            text
        )
        for text
        in references
    ]

    target_lengths = (
        torch.tensor(
            [
                len(target)
                for target
                in encoded_targets
            ],
            dtype=(
                torch.long
            ),
        )
    )

    flattened_targets = (
        torch.tensor(
            [
                token
                for target
                in encoded_targets
                for token
                in target
            ],
            dtype=(
                torch.long
            ),
        )
    )

    return {
        "targets": (
            flattened_targets
        ),
        "target_lengths": (
            target_lengths
        ),
    }


def create_dataset(
    config: dict[
        str,
        Any,
    ],
    split: str,
) -> IEUMDataset:

    paths = (
        resolve_data_paths(
            config
        )
    )

    data = (
        config[
            "data"
        ]
    )

    return IEUMDataset(
        csv_path=(
            paths[
                "csv_path"
            ]
        ),
        audio_root=(
            paths[
                "audio_root"
            ]
        ),
        split=(
            split
        ),
        audio_filename_column=(
            data[
                "audio_filename_column"
            ]
        ),
        audio_path_column=(
            data[
                "audio_path_column"
            ]
        ),
        segment_id_column=(
            data[
                "segment_id_column"
            ]
        ),
        transcript_column=(
            data[
                "transcript_column"
            ]
        ),
        word_column=(
            data[
                "word_column"
            ]
        ),
        split_column=(
            data[
                "split_column"
            ]
        ),
        speaker_column=(
            data[
                "speaker_column"
            ]
        ),
        segment_start_column=(
            data[
                "segment_start_column"
            ]
        ),
        segment_end_column=(
            data[
                "segment_end_column"
            ]
        ),
        abnormal_duration_column=(
            data[
                "abnormal_duration_column"
            ]
        ),
        sample_rate=(
            data[
                "sample_rate"
            ]
        ),
        min_chunk_seconds=(
            data[
                "min_chunk_seconds"
            ]
        ),
        max_audio_seconds=(
            data[
                "max_audio_seconds"
            ]
        ),
        load_audio=True,
    )


def create_downstream_model(
    architecture: str,
    input_dim: int,
    vocab_size: int,
    config: dict[
        str,
        Any,
    ],
) -> torch.nn.Module:

    model_config = (
        config[
            "model"
        ]
    )

    dropout = (
        model_config.get(
            "dropout",
            0.1,
        )
    )

    if architecture == "linear_ctc":

        return LinearCTC(
            input_dim=input_dim,
            vocab_size=vocab_size,
            dropout=dropout,
        )

    if architecture == "bigru_ctc":

        settings = (
            model_config[
                "bigru"
            ]
        )

        return BiGRUCTC(
            input_dim=input_dim,
            vocab_size=vocab_size,
            hidden_size=(
                settings[
                    "hidden_size"
                ]
            ),
            num_layers=(
                settings[
                    "num_layers"
                ]
            ),
            dropout=dropout,
        )

    if architecture == "transformer_ctc":

        settings = (
            model_config[
                "transformer"
            ]
        )

        return TransformerCTC(
            input_dim=input_dim,
            vocab_size=vocab_size,
            hidden_size=(
                settings[
                    "hidden_size"
                ]
            ),
            num_layers=(
                settings[
                    "num_layers"
                ]
            ),
            num_heads=(
                settings[
                    "num_heads"
                ]
            ),
            feedforward_size=(
                settings[
                    "feedforward_size"
                ]
            ),
            dropout=dropout,
        )

    if architecture == "conformer_ctc":

        settings = (
            model_config[
                "conformer"
            ]
        )

        return ConformerCTC(
            input_dim=input_dim,
            vocab_size=vocab_size,
            hidden_size=(
                settings[
                    "hidden_size"
                ]
            ),
            num_layers=(
                settings[
                    "num_layers"
                ]
            ),
            num_heads=(
                settings[
                    "num_heads"
                ]
            ),
            feedforward_size=(
                settings[
                    "feedforward_size"
                ]
            ),
            convolution_kernel_size=(
                settings[
                    "convolution_kernel_size"
                ]
            ),
            dropout=dropout,
        )

    raise ValueError(
        "지원하지 않는 모델: "
        f"{architecture}"
    )


def parse_args() -> argparse.Namespace:

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/base_config.yaml"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "linear_ctc",
            "bigru_ctc",
            "transformer_ctc",
            "conformer_ctc",
        ],
        default=None,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    config_path = (
        PROJECT_ROOT
        / args.config
    )

    config = (
        load_config(
            config_path
        )
    )

    set_seed(
        config[
            "project"
        ][
            "seed"
        ]
    )

    architecture = (
        args.model
        if args.model
        is not None
        else config[
            "model"
        ][
            "architecture"
        ]
    )

    encoder_mode = (
        config[
            "encoder"
        ][
            "train_mode"
        ]
    )

    requested_cache = bool(
        config[
            "training"
        ].get(
            "use_feature_cache",
            False,
        )
    )

    use_feature_cache = (
        requested_cache
        and encoder_mode
        == "freeze"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("IEUM ASR 학습 시작")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Model: {architecture}"
    )

    print(
        f"Encoder mode: "
        f"{encoder_mode}"
    )

    print(
        f"Feature cache: "
        f"{use_feature_cache}"
    )

    # ============================================================
    # Raw Dataset
    # ============================================================

    print()
    print(
        "Dataset 생성 중..."
    )

    train_raw = (
        create_dataset(
            config,
            "train",
        )
    )

    valid_raw = (
        create_dataset(
            config,
            "valid",
        )
    )

    print(
        f"Train chunks: "
        f"{len(train_raw)}"
    )

    print(
        f"Valid chunks: "
        f"{len(valid_raw)}"
    )

    # ============================================================
    # Tokenizer
    # ============================================================

    tokenizer = (
        CTCCharacterTokenizer
        .build_from_texts(
            train_raw
            .get_metadata()[
                "transcript"
            ]
        )
    )

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    # ============================================================
    # Whisper components
    # ============================================================

    feature_extractor = (
        IEUMWhisperFeatureExtractor(
            model_name=(
                config[
                    "whisper"
                ][
                    "model_name"
                ]
            ),
            sample_rate=(
                config[
                    "data"
                ][
                    "sample_rate"
                ]
            ),
            max_audio_seconds=(
                config[
                    "data"
                ][
                    "max_audio_seconds"
                ]
            ),
        )
    )

    encoder = (
        WhisperEncoder(
            model_name=(
                config[
                    "whisper"
                ][
                    "model_name"
                ]
            ),
            train_mode=(
                encoder_mode
            ),
        )
    )

    print(
        "Encoder parameter summary:",
        encoder
        .trainable_parameter_summary(),
    )

    # ============================================================
    # Cached mode
    # ============================================================

    if use_feature_cache:

        cache_root = Path(
            config[
                "training"
            ][
                "feature_cache_root"
            ]
        )

        train_cache_dir = (
            cache_root
            / "train"
        )

        valid_cache_dir = (
            cache_root
            / "valid"
        )

        build_feature_cache(
            dataset=train_raw,
            feature_extractor=(
                feature_extractor
            ),
            encoder=encoder,
            cache_dir=(
                train_cache_dir
            ),
            device=device,
            batch_size=(
                config[
                    "training"
                ][
                    "batch_size"
                ]
            ),
        )

        build_feature_cache(
            dataset=valid_raw,
            feature_extractor=(
                feature_extractor
            ),
            encoder=encoder,
            cache_dir=(
                valid_cache_dir
            ),
            device=device,
            batch_size=(
                config[
                    "training"
                ][
                    "batch_size"
                ]
            ),
        )

        # Encoder 메모리 해제
        del encoder

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        train_dataset = (
            CachedFeatureDataset(
                train_cache_dir
            )
        )

        valid_dataset = (
            CachedFeatureDataset(
                valid_cache_dir
            )
        )

        collator = (
            CachedCTCCollator(
                tokenizer
            )
        )

    else:

        train_dataset = (
            train_raw
        )

        valid_dataset = (
            valid_raw
        )

        collator = (
            RawCTCCollator(
                feature_extractor=(
                    feature_extractor
                ),
                tokenizer=(
                    tokenizer
                ),
            )
        )

    # ============================================================
    # DataLoader
    # ============================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            config[
                "training"
            ][
                "batch_size"
            ]
        ),
        shuffle=True,
        num_workers=(
            config[
                "data"
            ][
                "num_workers"
            ]
        ),
        collate_fn=(
            collator
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=(
            config[
                "training"
            ][
                "batch_size"
            ]
        ),
        shuffle=False,
        num_workers=(
            config[
                "data"
            ][
                "num_workers"
            ]
        ),
        collate_fn=(
            collator
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    # ============================================================
    # Downstream model
    # ============================================================

    downstream_model = (
        create_downstream_model(
            architecture=(
                architecture
            ),
            input_dim=(
                768
            ),
            vocab_size=(
                tokenizer.vocab_size
            ),
            config=(
                config
            ),
        )
    )

    if use_feature_cache:

        model = (
            CachedCTCModel(
                downstream_model
            )
        )

    else:

        model = (
            CTCASRModel(
                encoder=(
                    encoder
                ),
                downstream_model=(
                    downstream_model
                ),
                sample_rate=(
                    config[
                        "data"
                    ][
                        "sample_rate"
                    ]
                ),
            )
        )

    # ============================================================
    # Optimizer
    # ============================================================

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = (
        torch.optim.AdamW(
            trainable_parameters,
            lr=(
                config[
                    "training"
                ][
                    "learning_rate"
                ]
            ),
            weight_decay=(
                config[
                    "training"
                ][
                    "weight_decay"
                ]
            ),
        )
    )

    # ============================================================
    # Output
    # ============================================================

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir = (
        PROJECT_ROOT
        / config[
            "output"
        ][
            "root_dir"
        ]
        / (
            f"{architecture}_"
            f"{timestamp}"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        output_dir
        / "vocab.json"
    )

    # ============================================================
    # Train
    # ============================================================

    trainer = CTCTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        use_amp=(
            config[
                "training"
            ][
                "use_amp"
            ]
        ),
        gradient_clip_norm=1.0,
        use_cached_features=(
            use_feature_cache
        ),
    )

    summary = trainer.fit(
        train_loader=(
            train_loader
        ),
        valid_loader=(
            valid_loader
        ),
        epochs=(
            config[
                "training"
            ][
                "epochs"
            ]
        ),
        output_dir=(
            output_dir
        ),
        early_stopping_patience=(
            config[
                "training"
            ][
                "early_stopping_patience"
            ]
        ),
    )

    print()
    print("=" * 70)
    print("학습 완료")
    print("=" * 70)

    print(
        f"Best epoch: "
        f"{summary['best_epoch']}"
    )

    print(
        f"Best CER: "
        f"{summary['best_valid_cer']:.4f}"
    )

    print(
        f"결과 경로: "
        f"{output_dir}"
    )


if __name__ == "__main__":

    main()