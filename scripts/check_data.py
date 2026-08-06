import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.asr.config import load_config, resolve_data_paths
from src.asr.dataset import IEUMDataset


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_speaker_leakage(
    metadata: pd.DataFrame,
) -> None:
    """Train/Valid/Test 사이의 화자 중복을 검사한다."""
    split_speakers = {
        split_name: set(group["speaker_id"])
        for split_name, group in metadata.groupby("split")
    }

    split_names = list(split_speakers.keys())
    leakage_found = False

    for first_index in range(len(split_names)):
        for second_index in range(
            first_index + 1,
            len(split_names),
        ):
            first_split = split_names[first_index]
            second_split = split_names[second_index]

            overlap = (
                split_speakers[first_split]
                & split_speakers[second_split]
            )

            print(
                f"{first_split} ↔ {second_split}: "
                f"{len(overlap)}명"
            )

            if overlap:
                leakage_found = True
                print(
                    f"중복 화자 예시: "
                    f"{sorted(overlap)[:10]}"
                )

    print(f"\nSpeaker leakage 존재 여부: {leakage_found}")


def main() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "base_config.yaml"
    )

    config = load_config(config_path)
    paths = resolve_data_paths(config)

    data_config = config["data"]

    print_section("1. 데이터 경로 확인")

    print(f"CSV 경로: {paths['csv_path']}")
    print(f"오디오 루트: {paths['audio_root']}")

    if not paths["csv_path"].exists():
        raise FileNotFoundError(
            "CSV 파일을 찾을 수 없습니다.\n"
            f"확인한 경로: {paths['csv_path']}\n"
            "Colab에서 Google Drive를 마운트했는지 확인하세요."
        )

    print("CSV 파일 존재: True")

    print_section("2. Dataset 생성")

    dataset = IEUMDataset(
        csv_path=paths["csv_path"],
        audio_root=paths["audio_root"],
        split=None,
        audio_filename_column=(
            data_config["audio_filename_column"]
        ),
        audio_path_column=(
            data_config["audio_path_column"]
        ),
        segment_id_column=(
            data_config["segment_id_column"]
        ),
        transcript_column=(
            data_config["transcript_column"]
        ),
        split_column=data_config["split_column"],
        speaker_column=data_config["speaker_column"],
        segment_start_column=(
            data_config["segment_start_column"]
        ),
        segment_end_column=(
            data_config["segment_end_column"]
        ),
        sample_rate=data_config["sample_rate"],
        min_audio_seconds=(
            data_config["min_audio_seconds"]
        ),
        max_audio_seconds=(
            data_config["max_audio_seconds"]
        ),
        load_audio=False,
    )

    print("Dataset 생성 성공")

    print_section("3. Dataset 요약")

    summary = dataset.summary()

    for key, value in summary.items():
        print(f"{key}: {value}")

    metadata = dataset.get_metadata()

    print_section("4. Split별 샘플 및 화자 수")

    split_summary = (
        metadata.groupby("split")
        .agg(
            sample_count=("sample_id", "count"),
            speaker_count=("speaker_id", "nunique"),
            audio_count=("audio_filename", "nunique"),
        )
    )

    print(split_summary.to_string())

    print_section("5. 음성 길이 분포")

    duration_summary = (
        metadata["duration_seconds"]
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
    )

    print(duration_summary.to_string())

    print_section("6. 정답 문장 길이")

    transcript_lengths = (
        metadata["transcript"]
        .astype(str)
        .str.len()
    )

    print(
        transcript_lengths.describe(
            percentiles=[
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print_section("7. 중복 구조 확인")

    repeated_segment_ids = (
        metadata.groupby("segment_id")
        .agg(
            sample_count=("sample_id", "count"),
            speaker_count=("speaker_id", "nunique"),
            audio_count=("audio_filename", "nunique"),
        )
    )

    repeated_segment_ids = repeated_segment_ids[
        repeated_segment_ids["sample_count"] > 1
    ]

    print(
        "여러 음성에서 반복된 segment_id 수: "
        f"{len(repeated_segment_ids)}"
    )

    if not repeated_segment_ids.empty:
        print("\n반복 segment_id 예시:")
        print(
            repeated_segment_ids
            .head(10)
            .to_string()
        )

    duplicate_sample_ids = (
        metadata["sample_id"]
        .duplicated()
        .sum()
    )

    print(
        "\n중복 sample_id 수: "
        f"{duplicate_sample_ids}"
    )

    print_section("8. Speaker Leakage 확인")

    check_speaker_leakage(metadata)

    print_section("9. 샘플 5개 확인")

    preview_columns = [
        "sample_id",
        "audio_filename",
        "segment_id",
        "segment_start_sec",
        "segment_end_sec",
        "duration_seconds",
        "transcript",
        "speaker_id",
        "split",
        "source_row_count",
    ]

    print(
        metadata[preview_columns]
        .head(5)
        .to_string(index=False)
    )

    print_section("10. 실제 음성 1개 로딩 검사")

    audio_dataset = IEUMDataset(
        csv_path=paths["csv_path"],
        audio_root=paths["audio_root"],
        split=None,
        audio_filename_column=(
            data_config["audio_filename_column"]
        ),
        audio_path_column=(
            data_config["audio_path_column"]
        ),
        segment_id_column=(
            data_config["segment_id_column"]
        ),
        transcript_column=(
            data_config["transcript_column"]
        ),
        split_column=data_config["split_column"],
        speaker_column=data_config["speaker_column"],
        segment_start_column=(
            data_config["segment_start_column"]
        ),
        segment_end_column=(
            data_config["segment_end_column"]
        ),
        sample_rate=data_config["sample_rate"],
        min_audio_seconds=(
            data_config["min_audio_seconds"]
        ),
        max_audio_seconds=(
            data_config["max_audio_seconds"]
        ),
        load_audio=True,
    )

    sample = audio_dataset[0]

    print(f"sample_id: {sample['sample_id']}")
    print(f"audio_path: {sample['audio_path']}")
    print(f"waveform shape: {tuple(sample['waveform'].shape)}")
    print(f"sample_rate: {sample['sample_rate']}")
    print(
        "CSV 기준 길이: "
        f"{sample['duration_seconds']:.4f}초"
    )
    print(
        "실제 로딩 길이: "
        f"{sample['loaded_duration_seconds']:.4f}초"
    )
    print(f"transcript: {sample['transcript']}")

    print_section("데이터 검사 완료")


if __name__ == "__main__":
    main()