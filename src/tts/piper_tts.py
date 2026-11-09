"""Piper subprocess contract adapted from integration, with configurable paths."""
import shutil
import subprocess
import uuid
import os
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class PiperTTS:
    def __init__(self,settings):
        self.settings=settings

    @property
    def executable(self):
        value=self.settings.piper_executable
        path=Path(value).expanduser()
        # YAML paths resolve from the project root, not the server working directory.
        if path.is_absolute() or '/' in value or '\\' in value:
            return str(path if path.is_absolute() else PROJECT_ROOT/path)
        return value

    @property
    def ready(self):
        s=self.settings
        return bool(shutil.which(self.executable) and s.piper_model.is_file() and s.piper_config.is_file())

    def synthesize(self,text):
        if not text or not text.strip():
            raise ValueError('TTS text is empty.')
        if not self.ready:
            raise FileNotFoundError('Piper executable or Korean voice model/config is unavailable.')
        s=self.settings
        s.tts_output.mkdir(parents=True,exist_ok=True)
        path=s.tts_output/f'tts_{uuid.uuid4().hex}.wav'
        try:
            env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
            subprocess.run([self.executable,'-m',str(s.piper_model),'-c',str(s.piper_config),'-f',str(path)],input=text,encoding='utf-8',env=env,capture_output=True,check=True,timeout=60)
            if not path.is_file() or path.stat().st_size<=44:
                raise RuntimeError('Piper did not produce a valid WAV file.')
            with wave.open(str(path),'rb') as wav:
                if wav.getnframes()==0 or wav.getsampwidth()!=2 or wav.getnchannels()!=1:
                    raise RuntimeError('Expected a non-empty mono PCM16 Piper WAV.')
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise
