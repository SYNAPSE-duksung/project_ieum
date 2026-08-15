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

    return re.sub(
        r"[^0-9A-Za-z가-힣._-]+",
        "_",
        str(sample_id),
    )


def _make_feature_filename(
    dataset_index: int,
    sample_id: str,
) -> str:
    """
    Dataset index와 sample_id를 이용해
    항상 동일한 feature 파일명을 생성한다.

    기존 feature_cache.py에서 사용하던
    파일명 규칙과 동일하게 유지한다.
    """

    return (
        f"{dataset_index:06d}_"
        f"{_safe_filename(sample_id)}.pt"
    )


def _save_manifest(
    records: dict[str, dict[str, Any]],
    manifest_path: Path,
) -> None:
    """
    현재까지 생성된 cache 정보를 manifest.csv로 저장한다.

    batch가 끝날 때마다 호출하여
    Colab이 중단되더라도 진행 상태를 보존한다.
    """

    manifest = pd.DataFrame(
        list(records.values())
    )

    if not manifest.empty:

        manifest = (
            manifest
            .sort_values(
                "dataset_index"
            )
            .reset_index(drop=True)
        )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )


class CachedFeatureDataset(Dataset):
    """
    미리 계산된 Whisper Encoder feature를
    불러오는 Dataset.

    Encoder Freeze 상태에서
    Linear / BiGRU / Transformer / Conformer가
    동일한 Whisper feature를 재사용하도록 한다.
    """

    def __init__(
        self,
        cache_dir: str | Path,
    ) -> None:

        self.cache_dir = Path(
            cache_dir
        )

        self.manifest_path = (
            self.cache_dir
            / "manifest.csv"
        )

        self.metadata_path = (
            self.cache_dir
            / "cache_info.json"
        )

        if not self.manifest_path.exists():

            raise FileNotFoundError(
                "Feature cache manifest를 찾을 수 없습니다.\n"
                f"{self.manifest_path}"
            )

        self.manifest = pd.read_csv(
            self.manifest_path
        )

        if self.manifest.empty:

            raise ValueError(
                "Feature cache manifest가 비어 있습니다."
            )

        required_columns = [
            "sample_id",
            "transcript",
            "feature_file",
            "input_length",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column
            not in self.manifest.columns
        ]

        if missing_columns:

            raise ValueError(
                "Feature cache manifest에 "
                "필요한 컬럼이 없습니다.\n"
                f"누락 컬럼: {missing_columns}"
            )

        # 실제 feature 파일 존재 여부 확인
        missing_files = []

        for feature_file in (
            self.manifest[
                "feature_file"
            ]
            .astype(str)
        ):

            feature_path = (
                self.cache_dir
                / feature_file
            )

            if not feature_path.exists():

                missing_files.append(
                    feature_file
                )

        if missing_files:

            raise FileNotFoundError(
                "manifest에는 기록되어 있지만 "
                "실제 feature 파일이 없습니다.\n"
                f"누락 수: {len(missing_files)}\n"
                f"예시: {missing_files[:5]}"
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


def _get_dataset_metadata(
    dataset: Dataset,
) -> pd.DataFrame:
    """
    IEUMDataset의 metadata를 이용해
    audio를 실제로 읽지 않고 sample_id와 transcript를 확인한다.

    Resume 여부를 확인하기 위해 dataset[index]를 호출하면
    WAV 파일까지 읽게 되므로 매우 느려질 수 있다.
    """

    if not hasattr(
        dataset,
        "get_metadata",
    ):

        raise TypeError(
            "Feature cache 생성에는 "
            "get_metadata()를 제공하는 Dataset이 필요합니다."
        )

    metadata = (
        dataset
        .get_metadata()
        .reset_index(drop=True)
    )

    required_columns = [
        "sample_id",
        "transcript",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column
        not in metadata.columns
    ]

    if missing_columns:

        raise ValueError(
            "Dataset metadata에 필요한 "
            "컬럼이 없습니다.\n"
            f"누락 컬럼: {missing_columns}"
        )

    if len(metadata) != len(dataset):

        raise ValueError(
            "Dataset 길이와 metadata 길이가 "
            "일치하지 않습니다.\n"
            f"Dataset: {len(dataset)}\n"
            f"Metadata: {len(metadata)}"
        )

    return metadata


def _load_existing_manifest(
    manifest_path: Path,
    cache_dir: Path,
) -> dict[str, dict[str, Any]]:
    """
    기존 manifest에서 실제 .pt 파일이 존재하는
    정상 cache 항목만 읽어온다.
    """

    records: dict[
        str,
        dict[str, Any]
    ] = {}

    if not manifest_path.exists():

        return records

    try:

        manifest = pd.read_csv(
            manifest_path
        )

    except Exception as error:

        print(
            "기존 manifest.csv를 읽지 못했습니다."
        )

        print(
            f"오류: {error}"
        )

        return records

    required_columns = {
        "sample_id",
        "transcript",
        "feature_file",
        "input_length",
    }

    if not required_columns.issubset(
        set(
            manifest.columns
        )
    ):

        print(
            "기존 manifest 형식이 현재 버전과 "
            "일치하지 않습니다."
        )

        return records

    for _, row in (
        manifest.iterrows()
    ):

        sample_id = str(
            row[
                "sample_id"
            ]
        )

        feature_file = str(
            row[
                "feature_file"
            ]
        )

        feature_path = (
            cache_dir
            / feature_file
        )

        if not feature_path.exists():

            continue

        dataset_index = (
            int(
                row[
                    "dataset_index"
                ]
            )
            if (
                "dataset_index"
                in manifest.columns
                and not pd.isna(
                    row[
                        "dataset_index"
                    ]
                )
            )
            else -1
        )

        records[
            sample_id
        ] = {
            "dataset_index": (
                dataset_index
            ),
            "sample_id": (
                sample_id
            ),
            "transcript": str(
                row[
                    "transcript"
                ]
            ),
            "feature_file": (
                feature_file
            ),
            "input_length": int(
                row[
                    "input_length"
                ]
            ),
        }

    return records


def _recover_existing_feature_files(
    *,
    dataset_metadata: pd.DataFrame,
    cache_dir: Path,
    records: dict[str, dict[str, Any]],
) -> int:
    """
    manifest에는 없지만 기존 코드가 이미 생성해 둔
    .pt 파일이 있는 경우 이를 복구한다.

    예:
        000000_SAMPLE_ID.pt

    현재 실행 중이던 CPU cache를 중단했을 때
    이미 저장된 feature를 최대한 재사용하기 위한 기능이다.
    """

    recovered_count = 0

    for dataset_index, row in (
        dataset_metadata.iterrows()
    ):

        sample_id = str(
            row[
                "sample_id"
            ]
        )

        if sample_id in records:

            continue

        filename = (
            _make_feature_filename(
                dataset_index=(
                    int(
                        dataset_index
                    )
                ),
                sample_id=(
                    sample_id
                ),
            )
        )

        feature_path = (
            cache_dir
            / filename
        )

        if not feature_path.exists():

            continue

        try:

            cached = torch.load(
                feature_path,
                map_location="cpu",
                weights_only=True,
            )

            if (
                "hidden_states"
                not in cached
                or "input_length"
                not in cached
            ):

                continue

            input_length = int(
                cached[
                    "input_length"
                ]
            )

        except Exception:

            # 깨진 파일은 cache로 인정하지 않는다.
            continue

        records[
            sample_id
        ] = {
            "dataset_index": int(
                dataset_index
            ),
            "sample_id": (
                sample_id
            ),
            "transcript": str(
                row[
                    "transcript"
                ]
            ),
            "feature_file": (
                filename
            ),
            "input_length": (
                input_length
            ),
        }

        recovered_count += 1

    return recovered_count


def build_feature_cache(
    *,
    dataset: Dataset,
    feature_extractor: IEUMWhisperFeatureExtractor,
    encoder: WhisperEncoder,
    cache_dir: str | Path,
    device: torch.device,
    batch_size: int = 4,
    force_rebuild: bool = False,
    sample_rate: int = 16000,
) -> Path:
    """
    Dataset 전체를 Whisper Encoder에 한 번 통과시켜
    hidden state를 저장한다.

    Encoder Freeze 구조 비교에서 사용한다.

    주요 기능
    ----------
    1. 완성된 cache가 있으면 그대로 재사용
    2. 중간에 중단된 cache가 있으면 이어서 생성
    3. batch마다 manifest.csv 저장
    4. manifest가 없어도 기존 .pt 파일 복구 시도
    5. force_rebuild=True일 때만 기존 cache 전체 삭제
    6. CPU / CUDA 모두 지원
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

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Dataset metadata
    # ============================================================

    dataset_metadata = (
        _get_dataset_metadata(
            dataset
        )
    )

    total_samples = len(
        dataset_metadata
    )

    # ============================================================
    # Force rebuild
    # ============================================================

    if force_rebuild:

        print()
        print(
            "force_rebuild=True"
        )

        print(
            "기존 Feature Cache를 모두 삭제합니다."
        )

        for feature_file in (
            cache_dir.glob(
                "*.pt"
            )
        ):

            feature_file.unlink()

        if manifest_path.exists():

            manifest_path.unlink()

        if metadata_path.exists():

            metadata_path.unlink()

    # ============================================================
    # 기존 cache 확인
    # ============================================================

    records = (
        _load_existing_manifest(
            manifest_path=(
                manifest_path
            ),
            cache_dir=(
                cache_dir
            ),
        )
    )

    # ============================================================
    # manifest가 없는 기존 .pt 복구
    # ============================================================

    recovered_count = (
        _recover_existing_feature_files(
            dataset_metadata=(
                dataset_metadata
            ),
            cache_dir=(
                cache_dir
            ),
            records=(
                records
            ),
        )
    )

    if recovered_count > 0:

        print(
            f"기존 .pt 파일 "
            f"{recovered_count}개를 복구했습니다."
        )

        _save_manifest(
            records=records,
            manifest_path=(
                manifest_path
            ),
        )

    # ============================================================
    # 현재 Dataset에 실제 존재하는 sample만 유지
    # ============================================================

    valid_sample_ids = set(
        dataset_metadata[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    records = {
        sample_id: record
        for sample_id, record
        in records.items()
        if sample_id
        in valid_sample_ids
    }

    # ============================================================
    # cache 완성 여부
    # ============================================================

    if (
        len(records)
        == total_samples
        and not force_rebuild
    ):

        print()
        print("=" * 70)
        print(
            "기존 Feature Cache 사용"
        )
        print("=" * 70)

        print(
            f"Cache: {cache_dir}"
        )

        print(
            f"Cached samples: "
            f"{len(records)}"
        )

        _save_manifest(
            records=records,
            manifest_path=(
                manifest_path
            ),
        )

        return cache_dir

    # ============================================================
    # 아직 처리하지 않은 index 찾기
    # ============================================================

    remaining_indices: list[int] = []

    for dataset_index, row in (
        dataset_metadata.iterrows()
    ):

        sample_id = str(
            row[
                "sample_id"
            ]
        )

        if sample_id not in records:

            remaining_indices.append(
                int(
                    dataset_index
                )
            )

    print()
    print("=" * 70)
    print(
        "Whisper Encoder Feature Cache 생성"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"전체 Samples: "
        f"{total_samples}"
    )

    print(
        f"이미 생성됨: "
        f"{len(records)}"
    )

    print(
        f"남은 Samples: "
        f"{len(remaining_indices)}"
    )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"Cache: {cache_dir}"
    )

    # ============================================================
    # Encoder 준비
    # ============================================================

    encoder = encoder.to(
        device
    )

    encoder.eval()

    # ============================================================
    # Remaining batch 처리
    # ============================================================

    progress = tqdm(
        range(
            0,
            len(remaining_indices),
            batch_size,
        ),
        desc="Whisper feature 추출",
    )

    for batch_start in progress:

        batch_indices = (
            remaining_indices[
                batch_start:
                batch_start
                + batch_size
            ]
        )

        # 여기에서만 실제 WAV를 로딩한다.
        samples = [
            dataset[
                dataset_index
            ]
            for dataset_index
            in batch_indices
        ]

        waveforms = [
            sample[
                "waveform"
            ]
            for sample
            in samples
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
            .to(
                device
            )
        )

        audio_num_samples = (
            feature_batch[
                "audio_num_samples"
            ]
            .to(
                device
            )
        )

        # ========================================================
        # Whisper Encoder
        # ========================================================

        with torch.inference_mode():

            if device.type == "cuda":

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    hidden_states = (
                        encoder(
                            input_features
                        )
                    )

            else:

                # CPU에서는 autocast/float16 연산을 사용하지 않는다.
                hidden_states = (
                    encoder(
                        input_features
                    )
                )

            input_lengths = (
                encoder.get_output_lengths(
                    audio_num_samples=(
                        audio_num_samples
                    ),
                    sample_rate=(
                        sample_rate
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

        # 저장 공간 절약을 위해 cache는 float16으로 저장
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

        # ========================================================
        # 각 sample 저장
        # ========================================================

        for local_index, sample in enumerate(
            samples
        ):

            dataset_index = int(
                batch_indices[
                    local_index
                ]
            )

            sample_id = str(
                sample[
                    "sample_id"
                ]
            )

            valid_length = int(
                input_lengths[
                    local_index
                ].item()
            )

            sample_hidden = (
                hidden_states[
                    local_index,
                    :valid_length,
                ]
                .contiguous()
            )

            filename = (
                _make_feature_filename(
                    dataset_index=(
                        dataset_index
                    ),
                    sample_id=(
                        sample_id
                    ),
                )
            )

            feature_path = (
                cache_dir
                / filename
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
                feature_path,
            )

            records[
                sample_id
            ] = {
                "dataset_index": (
                    dataset_index
                ),
                "sample_id": (
                    sample_id
                ),
                "transcript": str(
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

        # ========================================================
        # 핵심: 매 batch마다 manifest 저장
        # ========================================================

        _save_manifest(
            records=records,
            manifest_path=(
                manifest_path
            ),
        )

        progress.set_postfix(
            cached=(
                f"{len(records)}/"
                f"{total_samples}"
            )
        )

    # ============================================================
    # 최종 검증
    # ============================================================

    _save_manifest(
        records=records,
        manifest_path=(
            manifest_path
        ),
    )

    completed_count = len(
        records
    )

    if completed_count != total_samples:

        print()
        print("=" * 70)
        print(
            "Feature Cache가 아직 완성되지 않았습니다."
        )
        print("=" * 70)

        print(
            f"완료: "
            f"{completed_count}"
            f"/{total_samples}"
        )

        print(
            "다음 실행 시 "
            "현재 위치부터 이어서 생성합니다."
        )

        return cache_dir

    # ============================================================
    # 완료 metadata
    # ============================================================

    cache_info = {
        "sample_count": (
            completed_count
        ),
        "hidden_size": int(
            encoder.hidden_size
        ),
        "sample_rate": int(
            sample_rate
        ),
        "complete": True,
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
    print("=" * 70)
    print(
        "Feature Cache 생성 완료"
    )
    print("=" * 70)

    print(
        f"Samples: "
        f"{completed_count}"
    )

    print(
        f"Cache: "
        f"{cache_dir}"
    )

    return cache_dir