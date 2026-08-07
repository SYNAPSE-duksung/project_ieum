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

    for first_index in range(
        len(split_names)
    ):
        for second_index in range(
            first_index + 1,
            len(split_names),
        ):

            first_split = (
                split_names[
                    first_index
                ]
            )

            second_split = (
                split_names[
                    second_index
                ]
            )

            overlap = (
                split_speakers[
                    first_split
                ]
                & split_speakers[
                    second_split
                ]
            )

            print(
                f"{first_split} ↔ "
                f"{second_split}: "
                f"{len(overlap)}명"
            )

            if overlap:
                leakage_found = True

                print(
                    "중복 화자 예시: "
                    f"{sorted(overlap)[:10]}"
                )

    print(
        "\nSpeaker leakage 존재 여부: "
        f"{leakage_found}"
    )


def create_dataset(
    config: dict,
    load_audio: bool,
) -> IEUMDataset:

    paths = (
        resolve_data_paths(
            config
        )
    )

    data = config["data"]

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
        word_column=(
            data[
                "word_column"
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
            "CSV 파일을 찾을 수 없습니다.\n"
            f"확인한 경로: "
            f"{paths['csv_path']}"
        )

    print(
        "CSV 파일 존재: True"
    )

    # ---------------------------------------------------------

    print_section(
        "2. Dataset 생성"
    )

    dataset = create_dataset(
        config=config,
        load_audio=False,
    )

    print(
        "Dataset 생성 성공"
    )

    # ---------------------------------------------------------

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

    metadata = (
        dataset.get_metadata()
    )

    # ---------------------------------------------------------

    print_section(
        "4. 원래 segment 보존 확인"
    )

    original_segment_count = (
        metadata[
            "original_sample_id"
        ]
        .nunique()
    )

    final_chunk_count = len(
        metadata
    )

    print(
        "원래 segment 수: "
        f"{original_segment_count}"
    )

    print(
        "최종 학습 chunk 수: "
        f"{final_chunk_count}"
    )

    print(
        "추가 생성 chunk 수: "
        f"{final_chunk_count - original_segment_count}"
    )

    print(
        "30초 초과로 분할된 원래 segment 수: "
        f"{(
            metadata[
                metadata[
                    'chunk_count'
                ] > 1
            ][
                'original_sample_id'
            ]
            .nunique()
        )}"
    )

    # ---------------------------------------------------------

    print_section(
        "5. Split별 원래 segment 수"
    )

    original_split = (
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

    original_split_summary = (
        original_split.groupby(
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
        original_split_summary
        .to_string()
    )

    # ---------------------------------------------------------

    print_section(
        "6. Split별 최종 chunk 수"
    )

    chunk_summary = (
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
        chunk_summary.to_string()
    )

    # ---------------------------------------------------------

    print_section(
        "7. Chunk 길이 분포"
    )

    print(
        metadata[
            "duration_seconds"
        ]
        .describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    max_duration = (
        metadata[
            "duration_seconds"
        ].max()
    )

    print(
        "\n최대 chunk 길이: "
        f"{max_duration:.4f}초"
    )

    if (
        max_duration
        > config[
            "data"
        ][
            "max_audio_seconds"
        ]
        + 1e-6
    ):
        raise ValueError(
            "30초를 초과한 chunk가 존재합니다."
        )

    # ---------------------------------------------------------

    print_section(
        "8. 중복 ID 확인"
    )

    duplicate_sample_ids = (
        metadata[
            "sample_id"
        ]
        .duplicated()
        .sum()
    )

    print(
        "중복 sample_id 수: "
        f"{duplicate_sample_ids}"
    )

    # ---------------------------------------------------------

    print_section(
        "9. Speaker Leakage 확인"
    )

    check_speaker_leakage(
        metadata
    )

    # ---------------------------------------------------------

    print_section(
        "10. 분할된 segment 예시"
    )

    chunked_examples = (
        metadata[
            metadata[
                "chunk_count"
            ] > 1
        ]
        .head(10)
    )

    if chunked_examples.empty:
        print(
            "분할된 segment 없음"
        )
    else:
        print(
            chunked_examples[
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

    # ---------------------------------------------------------

    print_section(
        "11. 실제 음성 1개 로딩 검사"
    )

    audio_dataset = (
        create_dataset(
            config=config,
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
        "Metadata 기준 길이: "
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

    print_section(
        "데이터 검사 완료"
    )


if __name__ == "__main__":
    main()