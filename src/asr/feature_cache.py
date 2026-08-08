from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.asr.encoder import WhisperEncoder
from src.asr.feature_extractor import (
    IEUMWhisperFeatureExtractor,
)


def _safe_filename(
    sample_id: str,
) -> str:
    """
    sample_id를 파일명으로 안전하게 변환한다.
    """

    safe = re.sub(
        r"[^0-9A-Za-z가-힣._-]+",
        "_",
        str(sample_id),
    )

    return safe


class CachedFeatureDataset(Dataset):
    """
    미리 계산된 Whisper Encoder feature를
    불러오는 Dataset.
    """

    def __init__(
        self,
        cache_dir: str | Path,
    ) -> None:

        self.cache_dir = Path(
            cache_dir
        )

        manifest_path = (
            self.cache_dir
            / "manifest.csv"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                "Feature cache manifest를 찾을 수 없습니다.\n"
                f"{manifest_path}"
            )

        self.manifest = pd.read_csv(
            manifest_path
        )

        if self.manifest.empty:
            raise ValueError(
                "Feature cache가 비어 있습니다."
            )

    def __len__(
        self,
    ) -> int:

        return len(
            self.manifest
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        row = self.manifest.iloc[
            index
        ]

        feature_path = (
            self.cache_dir
            / str(
                row[
                    "feature_file"
                ]
            )
        )

        cached = torch.load(
            feature_path,
            map_location="cpu",
            weights_only=True,
        )

        return {
            "hidden_states": (
                cached[
                    "hidden_states"
                ]
            ),
            "input_length": int(
                cached[
                    "input_length"
                ]
            ),
            "transcript": str(
                row[
                    "transcript"
                ]
            ),
            "sample_id": str(
                row[
                    "sample_id"
                ]
            ),
        }


def build_feature_cache(
    *,
    dataset: Dataset,
    feature_extractor: IEUMWhisperFeatureExtractor,
    encoder: WhisperEncoder,
    cache_dir: str | Path,
    device: torch.device,
    batch_size: int = 4,
    force_rebuild: bool = False,
) -> Path:
    """
    Dataset 전체를 Whisper Encoder에 한 번 통과시켜
    hidden state를 저장한다.

    Encoder Freeze 실험에서만 사용한다.
    """

    cache_dir = Path(
        cache_dir
    )

    manifest_path = (
        cache_dir
        / "manifest.csv"
    )

    metadata_path = (
        cache_dir
        / "cache_info.json"
    )

    # 이미 완성된 cache가 있으면 재사용
    if (
        manifest_path.exists()
        and metadata_path.exists()
        and not force_rebuild
    ):
        manifest = pd.read_csv(
            manifest_path
        )

        if len(manifest) == len(dataset):

            print(
                f"기존 feature cache 사용: "
                f"{cache_dir}"
            )

            print(
                f"Cached samples: "
                f"{len(manifest)}"
            )

            return cache_dir

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 이전 incomplete cache 제거
    for feature_file in (
        cache_dir.glob(
            "*.pt"
        )
    ):
        feature_file.unlink()

    encoder = encoder.to(
        device
    )

    encoder.eval()

    records: list[
        dict[str, Any]
    ] = []

    print()
    print("=" * 70)
    print("Whisper Encoder Feature Cache 생성")
    print("=" * 70)

    print(
        f"Samples: {len(dataset)}"
    )

    print(
        f"Cache: {cache_dir}"
    )

    # Dataset에서 batch 단위로 직접 묶는다.
    for start_index in tqdm(
        range(
            0,
            len(dataset),
            batch_size,
        ),
        desc="Whisper feature 추출",
    ):

        end_index = min(
            start_index
            + batch_size,
            len(dataset),
        )

        samples = [
            dataset[index]
            for index in range(
                start_index,
                end_index,
            )
        ]

        waveforms = [
            sample[
                "waveform"
            ]
            for sample in samples
        ]

        feature_batch = (
            feature_extractor.batch(
                waveforms
            )
        )

        input_features = (
            feature_batch[
                "input_features"
            ]
            .to(device)
        )

        audio_num_samples = (
            feature_batch[
                "audio_num_samples"
            ]
            .to(device)
        )

        # Encoder는 완전히 frozen
        with torch.inference_mode():

            with torch.autocast(
                device_type=(
                    device.type
                ),
                dtype=torch.float16,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):

                hidden_states = encoder(
                    input_features
                )

            input_lengths = (
                encoder.get_output_lengths(
                    audio_num_samples=(
                        audio_num_samples
                    ),
                    sample_rate=16000,
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

        hidden_states = (
            hidden_states
            .detach()
            .cpu()
            .to(
                torch.float16
            )
        )

        input_lengths = (
            input_lengths
            .detach()
            .cpu()
        )

        for local_index, sample in enumerate(
            samples
        ):

            valid_length = int(
                input_lengths[
                    local_index
                ].item()
            )

            # Padding 부분은 저장하지 않는다.
            sample_hidden = (
                hidden_states[
                    local_index,
                    :valid_length,
                ]
                .contiguous()
            )

            filename = (
                f"{start_index + local_index:06d}_"
                f"{_safe_filename(sample['sample_id'])}.pt"
            )

            torch.save(
                {
                    "hidden_states": (
                        sample_hidden
                    ),
                    "input_length": (
                        valid_length
                    ),
                },
                cache_dir
                / filename,
            )

            records.append(
                {
                    "sample_id": (
                        sample[
                            "sample_id"
                        ]
                    ),
                    "transcript": (
                        sample[
                            "transcript"
                        ]
                    ),
                    "feature_file": (
                        filename
                    ),
                    "input_length": (
                        valid_length
                    ),
                }
            )

    manifest = pd.DataFrame(
        records
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    cache_info = {
        "sample_count": (
            len(records)
        ),
        "hidden_size": (
            int(
                encoder.hidden_size
            )
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            cache_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Feature cache 생성 완료"
    )

    print(
        f"Samples: {len(records)}"
    )

    return cache_dir