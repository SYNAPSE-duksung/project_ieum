"""Piper subprocess contract adapted from integration, with configurable paths."""
import shutil
import subprocess
import uuid

class PiperTTS:
    def __init__(self,settings):
        self.settings=settings

    @property
    def ready(self):
        s=self.settings
        return bool(shutil.which(s.piper_executable) and s.piper_model.is_file() and s.piper_config.is_file())

    def synthesize(self,text):
        if not self.ready:
            raise FileNotFoundError('Piper executable or Korean voice model/config is unavailable.')
        s=self.settings
        s.tts_output.mkdir(parents=True,exist_ok=True)
        path=s.tts_output/f'tts_{uuid.uuid4().hex}.wav'
        try:
            subprocess.run([s.piper_executable,'-m',str(s.piper_model),'-c',str(s.piper_config),'-f',str(path)],input=text,encoding='utf-8',capture_output=True,check=True,timeout=60)
            if not path.is_file() or path.stat().st_size<=44:
                raise RuntimeError('Piper did not produce a valid WAV file.')
            return path
        except Exception:
            path.unlink(missing_ok=True)
            raise
