from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    YAML 설정 파일을 읽어 딕셔너리로 반환한다.

    Args:
        config_path:
            YAML 설정 파일 경로

    Returns:
        설정값이 저장된 딕셔너리
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"설정 파일을 찾을 수 없습니다: {config_path.resolve()}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"설정 파일이 비어 있습니다: {config_path}")

    if not isinstance(config, dict):
        raise ValueError(
            "YAML 설정 파일의 최상위 구조는 key-value 형태여야 합니다."
        )

    return config


def resolve_data_paths(
    config: dict[str, Any],
) -> dict[str, Path]:
    """
    project_root와 상대경로를 결합해 실제 데이터 경로를 만든다.

    예:
        project_root / alignment_csv
        project_root / audio_root
    """
    if "data" not in config:
        raise KeyError("설정 파일에 'data' 항목이 없습니다.")

    data_config = config["data"]

    required_keys = [
        "project_root",
        "alignment_csv",
        "audio_root",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in data_config
    ]

    if missing_keys:
        raise KeyError(
            f"data 설정에 필요한 항목이 없습니다: {missing_keys}"
        )

    project_root = Path(data_config["project_root"])

    return {
        "project_root": project_root,
        "csv_path": project_root / data_config["alignment_csv"],
        "audio_root": project_root / data_config["audio_root"],
    }