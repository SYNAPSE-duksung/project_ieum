import sys
from pathlib import Path


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

    data = config["data"]
    training = config["training"]

    print("=" * 60)
    print("설정 파일 로딩 성공")
    print("=" * 60)

    print(
        f"프로젝트명: "
        f"{config['project']['name']}"
    )

    print(
        f"랜덤 시드: "
        f"{config['project']['seed']}"
    )

    print(
        f"프로젝트 데이터 루트: "
        f"{data['project_root']}"
    )

    print(
        f"CSV 경로: "
        f"{paths['csv_path']}"
    )

    print(
        f"오디오 루트: "
        f"{paths['audio_root']}"
    )

    print(
        f"정답 문장 컬럼: "
        f"{data['transcript_column']}"
    )

    print(
        f"단어 컬럼: "
        f"{data['word_column']}"
    )

    print(
        f"샘플링 레이트: "
        f"{data['sample_rate']}"
    )

    print(
        f"Whisper 최대 입력 길이: "
        f"{data['max_audio_seconds']}초"
    )

    print(
        f"Whisper 모델: "
        f"{config['whisper']['model_name']}"
    )

    print(
        f"후속 모델: "
        f"{config['model']['architecture']}"
    )

    print(
        f"Encoder 학습 범위: "
        f"{config['encoder']['train_mode']}"
    )

    print(
        f"Batch size: "
        f"{training['batch_size']}"
    )

    print(
        f"최대 Epoch: "
        f"{training['epochs']}"
    )

    print(
        f"Learning rate: "
        f"{training['learning_rate']}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()