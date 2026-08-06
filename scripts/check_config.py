import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config import load_config, resolve_data_paths


def main() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs"
        / "base_config.yaml"
    )

    config = load_config(config_path)
    paths = resolve_data_paths(config)

    data_config = config["data"]

    print("=" * 60)
    print("설정 파일 로딩 성공")
    print("=" * 60)

    print(f"프로젝트명: {config['project']['name']}")
    print(f"랜덤 시드: {config['project']['seed']}")

    print(f"프로젝트 데이터 루트: {paths['project_root']}")
    print(f"CSV 경로: {paths['csv_path']}")
    print(f"오디오 루트: {paths['audio_root']}")

    print(f"정답 문장 컬럼: {data_config['transcript_column']}")
    print(f"샘플링 레이트: {data_config['sample_rate']}")
    print(
        "음성 길이 범위: "
        f"{data_config['min_audio_seconds']}초"
        f" ~ {data_config['max_audio_seconds']}초"
    )

    print(f"Whisper 모델: {config['whisper']['model_name']}")
    print(f"후속 모델: {config['model']['architecture']}")
    print(f"Encoder 학습 범위: {config['encoder']['train_mode']}")

    print(f"Batch size: {config['training']['batch_size']}")
    print(f"최대 Epoch: {config['training']['epochs']}")
    print(
        f"Learning rate: "
        f"{config['training']['learning_rate']}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()