"""Local demo configuration. Relative paths resolve from the repository root."""
import os
from dataclasses import dataclass
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    general_model: Path
    personalized_model: Path
    general_processor: Path
    personalized_processor: Path
    device: str = 'auto'
    precision: str = 'auto'
    max_upload_bytes: int = 32 * 1024 * 1024
    piper_executable: str = 'piper'
    piper_model: Path = ROOT/'piper/ko_KR-kss-medium.onnx'
    piper_config: Path = ROOT/'piper/ko_KR-kss-medium.onnx.json'
    tts_output: Path = ROOT/'outputs/e1_demo/tts'

def load_settings():
    path = Path(os.environ.get('IEUM_API_CONFIG', ROOT/'configs/e1_demo_api.yaml'))
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    def local(key):
        value = Path(os.environ.get('IEUM_'+key.upper(), cfg[key])).expanduser()
        return value if value.is_absolute() else ROOT/value
    return Settings(**{k:local(k) for k in ('general_model','personalized_model','general_processor','personalized_processor','piper_model','piper_config','tts_output')},
        device=os.environ.get('IEUM_DEVICE',cfg['device']), precision=os.environ.get('IEUM_PRECISION',cfg['precision']),
        max_upload_bytes=int(cfg['max_upload_bytes']), piper_executable=os.environ.get('IEUM_PIPER_EXECUTABLE',cfg['piper_executable']))
