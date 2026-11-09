"""Download the approved Korean Piper voice, with revision/hash provenance."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
REPO='https://huggingface.co/rhasspy/piper-voices'
VOICE='ko/ko_KR/kss/medium'

def main():
    target=ROOT/'piper'
    names=['ko_KR-kss-medium.onnx','ko_KR-kss-medium.onnx.json','MODEL_CARD']
    if any((target/n).exists() for n in names):
        raise FileExistsError('Refusing to overwrite existing Korean voice assets')
    with urllib.request.urlopen('https://huggingface.co/api/models/rhasspy/piper-voices',timeout=60) as response:
        revision=json.load(response)['sha']
    target.mkdir(exist_ok=True)
    records=[]
    for name in names:
        url=f'{REPO}/resolve/{revision}/{VOICE}/{name}'
        temporary=target/(name+'.partial')
        try:
            with urllib.request.urlopen(url,timeout=120) as source, temporary.open('wb') as dest:
                while chunk:=source.read(1024*1024):
                    dest.write(chunk)
            temporary.replace(target/name)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        with (target/name).open('rb') as source:
            digest=hashlib.file_digest(source,'sha256').hexdigest()
        records.append(dict(file=name,url=url,bytes=(target/name).stat().st_size,sha256=digest))
        print(name,records[-1]['bytes'],flush=True)
    (target/'voice_provenance.json').write_text(json.dumps(dict(repository=REPO,revision=revision,files=records),indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
