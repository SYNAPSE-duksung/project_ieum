"""Copy evaluated smoke WAVs to a NEW local folder for acoustic/text review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile
from general_test import DEFAULT_ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    args = p.parse_args()
    output = args.run/'review_audio'
    if output.exists():
        p.error('Review folder already exists; no overwrite')
    with sqlite3.connect(args.run/'checkpoint.sqlite3') as db:
        rows = [json.loads(r[0]) for r in db.execute(
            'SELECT a.payload FROM attempts a JOIN (SELECT utterance,MAX(id) id FROM attempts GROUP BY utterance) b ON a.id=b.id')]
    if len(rows)>20:
        p.error('Review export is limited to smoke sets of <=20 utterances')
    output.mkdir()
    index = []
    for n,row in enumerate(rows,1):
        if row['status']!='success':
            continue
        archive_path = args.data_root/'processed_candidate2_archives'/(row['batch']+'.zip')
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(row['audio_member'])
        if hashlib.sha256(payload).hexdigest()!=row['wav_sha256']:
            raise ValueError('Audio changed since inference')
        destination = output/(f'{n:02d}_'+row['speaker_id']+'.wav')
        with destination.open('xb') as stream:
            stream.write(payload)
        index.append(dict(wav=destination.name,speaker_id=row['speaker_id'],
                          disability_group=row['disability_group'],duration_sec=row['duration_sec'],
                          reference=row['reference'],prediction=row['prediction'],wav_sha256=row['wav_sha256']))
    with (output/'review_index.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    print(f'Exported {len(index)} exact inference WAVs with matching SHA256 to {output}')


if __name__=='__main__':
    main()
