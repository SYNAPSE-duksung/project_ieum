import sys
from pathlib import Path

import pandas as pd


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


def print_section(
    title: str,
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def create_dataset(
    config: dict,
    *,
    load_audio: bool,
) -> IEUMDataset:

    paths = resolve_data_paths(
        config
    )

    data = config[
        "data"
    ]

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
        split=None,
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
        max_audio_seconds=(
            data[
                "max_audio_seconds"
            ]
        ),
        load_audio=(
            load_audio
        ),
    )


def check_speaker_leakage(
    metadata: pd.DataFrame,
) -> None:

    split_speakers = {
        split_name: set(
            group[
                "speaker_id"
            ]
        )
        for split_name, group
        in metadata.groupby(
            "split"
        )
    }

    split_names = list(
        split_speakers.keys()
    )

    leakage_found = False

    for i in range(
        len(split_names)
    ):

        for j in range(
            i + 1,
            len(split_names),
        ):

            split_a = (
                split_names[i]
            )

            split_b = (
                split_names[j]
            )

            overlap = (
                split_speakers[
                    split_a
                ]
                & split_speakers[
                    split_b
                ]
            )

            print(
                f"{split_a} ↔ "
                f"{split_b}: "
                f"{len(overlap)}명"
            )

            if overlap:

                leakage_found = True

                print(
                    "중복 화자: "
                    f"{sorted(overlap)[:10]}"
                )

    print(
        "\nSpeaker leakage 존재 여부: "
        f"{leakage_found}"
    )


def main() -> None:

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "base_config.yaml"
    )

    config = load_config(
        config_path
    )

    paths = resolve_data_paths(
        config
    )

    # ============================================================
    print_section(
        "1. 데이터 경로 확인"
    )

    print(
        f"CSV 경로: "
        f"{paths['csv_path']}"
    )

    print(
        f"오디오 루트: "
        f"{paths['audio_root']}"
    )

    if not paths[
        "csv_path"
    ].exists():

        raise FileNotFoundError(
            "CSV 파일을 찾을 수 없습니다."
        )

    print(
        "CSV 파일 존재: True"
    )

    # ============================================================
    print_section(
        "2. Dataset 생성"
    )

    dataset = create_dataset(
        config,
        load_audio=False,
    )

    print(
        "Dataset 생성 성공"
    )

    metadata = (
        dataset.get_metadata()
    )

    # ============================================================
    print_section(
        "3. Dataset 요약"
    )

    summary = (
        dataset.summary()
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ============================================================
    print_section(
        "4. 원래 segment 보존 확인"
    )

    original_count = (
        metadata[
            "original_sample_id"
        ]
        .nunique()
    )

    final_count = (
        len(metadata)
    )

    print(
        f"원래 segment 수: "
        f"{original_count}"
    )

    print(
        f"최종 chunk 수: "
        f"{final_count}"
    )

    print(
        "추가 생성된 chunk 수: "
        f"{final_count - original_count}"
    )

    chunked_count = (
        metadata[
            metadata[
                "chunk_count"
            ] > 1
        ][
            "original_sample_id"
        ]
        .nunique()
    )

    print(
        "분할된 원래 segment 수: "
        f"{chunked_count}"
    )

    # ============================================================
    print_section(
        "5. Split별 원래 segment 수"
    )

    original_metadata = (
        metadata[
            [
                "original_sample_id",
                "split",
                "speaker_id",
                "audio_filename",
            ]
        ]
        .drop_duplicates(
            subset=[
                "original_sample_id"
            ]
        )
    )

    original_summary = (
        original_metadata.groupby(
            "split"
        )
        .agg(
            segment_count=(
                "original_sample_id",
                "count",
            ),
            speaker_count=(
                "speaker_id",
                "nunique",
            ),
            audio_count=(
                "audio_filename",
                "nunique",
            ),
        )
    )

    print(
        original_summary.to_string()
    )

    # ============================================================
    print_section(
        "6. Split별 최종 chunk 수"
    )

    final_summary = (
        metadata.groupby(
            "split"
        )
        .agg(
            chunk_count=(
                "sample_id",
                "count",
            ),
            speaker_count=(
                "speaker_id",
                "nunique",
            ),
            audio_count=(
                "audio_filename",
                "nunique",
            ),
        )
    )

    print(
        final_summary.to_string()
    )

    # ============================================================
    print_section(
        "7. Chunk 길이 분포"
    )

    duration = metadata[
        "duration_seconds"
    ]

    print(
        duration.describe(
            percentiles=[
                0.50,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    max_audio_seconds = float(
        config["data"]["max_audio_seconds"]
    )

    over_30_count = int(
        (
            duration
            > max_audio_seconds + 1e-6
        ).sum()
    )

    under_01_count = int(
        (
            duration < 0.1
        ).sum()
    )

    print(
        f"\n{max_audio_seconds}초 초과 chunk 수: "
        f"{over_30_count}"
    )

    print(
        "0.1초 미만 chunk 수: "
        f"{under_01_count}"
    )

    if over_30_count > 0:
        raise ValueError(
            "Whisper 최대 입력 길이를 초과하는 "
            "chunk가 존재합니다."
        )
    
    # ============================================================
    print_section(
        "8. 중복 sample_id 확인"
    )

    duplicate_count = int(
        metadata[
            "sample_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"중복 sample_id 수: "
        f"{duplicate_count}"
    )

    # ============================================================
    print_section(
        "9. Speaker Leakage 확인"
    )

    check_speaker_leakage(
        metadata
    )

    # ============================================================
    print_section(
        "10. Chunk 분할 예시"
    )

    examples = (
        metadata[
            metadata[
                "chunk_count"
            ] > 1
        ]
        .head(15)
    )

    if examples.empty:

        print(
            "분할된 segment가 없습니다."
        )

    else:

        print(
            examples[
                [
                    "sample_id",
                    "original_sample_id",
                    "chunk_index",
                    "chunk_count",
                    "duration_seconds",
                    "transcript",
                    "split",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ============================================================
    print_section(
        "11. 실제 음성 1개 로딩 검사"
    )

    audio_dataset = (
        create_dataset(
            config,
            load_audio=True,
        )
    )

    sample = (
        audio_dataset[0]
    )

    print(
        f"sample_id: "
        f"{sample['sample_id']}"
    )

    print(
        f"audio_path: "
        f"{sample['audio_path']}"
    )

    print(
        "waveform shape: "
        f"{tuple(sample['waveform'].shape)}"
    )

    print(
        f"sample_rate: "
        f"{sample['sample_rate']}"
    )

    print(
        "metadata 길이: "
        f"{sample['duration_seconds']:.4f}초"
    )

    print(
        "실제 로딩 길이: "
        f"{sample['loaded_duration_seconds']:.4f}초"
    )

    print(
        f"transcript: "
        f"{sample['transcript']}"
    )

    # ============================================================
    print_section(
        "데이터 검사 완료"
    )


if __name__ == "__main__":
    main()