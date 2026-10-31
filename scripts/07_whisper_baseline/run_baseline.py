"""Whisper Small Korean zero-shot evaluation with durable per-utterance checkpoints."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import zipfile

from filelock import FileLock
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from huggingface_hub import HfApi

from general_test import DEFAULT_ROOT, GeneralTest
from prepare_inputs import key
from score_predictions import NORMALIZATION, aggregate, counts

BASE = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Journal:
    def __init__(self, path, config):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, utterance TEXT NOT NULL, payload TEXT NOT NULL)')
        saved = self.db.execute('SELECT payload FROM config WHERE id=1').fetchone()
        serialized = json.dumps(config, sort_keys=True, ensure_ascii=False)
        if saved and saved[0] != serialized:
            self.db.close()
            raise ValueError('Resume configuration differs; use a NEW output directory')
        if not saved:
            with self.db:
                self.db.execute('INSERT INTO config VALUES (1, ?)', (serialized,))

    def latest(self):
        return {uid: json.loads(payload) for uid, payload in self.db.execute(
            'SELECT a.utterance,a.payload FROM attempts a JOIN '
            '(SELECT utterance, MAX(id) id FROM attempts GROUP BY utterance) b ON a.id=b.id')}

    def save(self, record):
        with self.db:
            self.db.execute('INSERT INTO attempts (utterance,payload) VALUES (?,?)',
                            (record['utterance_id'], json.dumps(record, ensure_ascii=False)))

    def export(self, output, selected_count):
        rows = list(self.latest().values())
        scores = aggregate(rows)
        scores.update(selected=selected_count, pending=selected_count-len(rows),
                      complete=len(rows) == selected_count and all(r['status']=='success' for r in rows))
        atomic_json(output / 'scores.json', scores)
        for filename, subset in [('predictions.csv', rows), ('failures.csv', [r for r in rows if r['status']=='failed'])]:
            path = output / filename
            temp = path.with_suffix('.csv.tmp')
            fields = ['utterance_id','batch','audio_member','speaker_id','disability_group','duration_sec',
                      'reference','prediction','status','error','stage','loading_sec','inference_sec',
                      'peak_allocated_mib','peak_reserved_mib','transcript_matches','source_sample_rate',
                      'wav_sha256','normalized_cer','normalized_wer','raw_cer','raw_wer']
            with temp.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                for row in subset:
                    flat = dict(row)
                    for mode in ('normalized','raw'):
                        for metric in ('cer','wer'):
                            flat[mode+'_'+metric] = row.get('scores',{}).get(mode,{}).get(metric)
                    writer.writerow(flat)
            os.replace(temp, path)
        return scores


def environment():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no implicit CPU fallback')
    probe = torch.ones((32,32), device='cuda', dtype=torch.float16)
    probe = probe @ probe
    torch.cuda.synchronize()
    if not torch.isfinite(probe).all():
        raise RuntimeError('CUDA compute probe failed')
    prop = torch.cuda.get_device_properties(0)
    return dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                cuda_runtime=torch.version.cuda, cuda_available=True,
                gpu=prop.name, vram_mib=prop.total_memory/2**20,
                capability=list(torch.cuda.get_device_capability(0)), arch_list=torch.cuda.get_arch_list(),
                packages={p:importlib.metadata.version(p) for p in ('transformers','numpy','scipy','soundfile','jiwer')})


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / 'run.lock'), timeout=0):
        return locked_run(args)


def locked_run(args):
    started = time.perf_counter()
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    rows = selection['rows']
    if not rows or len({key(r) for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate selection')
    if len(rows) > 20 and not args.allow_full_test:
        raise ValueError('More than 20 utterances requires --allow-full-test')
    dataset = GeneralTest(args.data_root)
    if (selection['manifest_sha256'],selection['speaker_split_sha256']) != (dataset.manifest_sha256,dataset.speaker_sha256):
        raise ValueError('Source CSV changed since selection')
    source = {key(r): r for r in dataset.test}
    if any(source.get(key(r)) != r for r in rows):
        raise ValueError('Selection differs from authoritative General Test rows')
    env = environment()
    config_path = args.output / 'run_config.json'
    prior = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else None
    revision = args.revision or (prior['revision'] if prior else HfApi().model_info('openai/whisper-small').sha)
    config = dict(schema=1, model='openai/whisper-small', revision=revision,
                  selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
                  manifest_sha256=dataset.manifest_sha256, speaker_split_sha256=dataset.speaker_sha256,
                  language='ko', task='transcribe', do_sample=False, num_beams=1,
                  max_new_tokens=440, dtype='float16', sample_rate=16000,
                  normalization=NORMALIZATION, environment=env,
                  archive_locations=dataset.archive_locations,
                  code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('run_baseline.py','general_test.py','score_predictions.py')})
    if prior and prior != config:
        raise ValueError('Run configuration or environment changed; use a NEW output directory')
    if not prior:
        atomic_json(config_path,config)
    journal = Journal(args.output / 'checkpoint.sqlite3',config)
    latest = journal.latest()
    pending = [r for r in rows if key(r) not in latest or
               (args.retry_failures and latest[key(r)]['status']=='failed')]
    skipped = len(rows)-len(pending)
    deferred = 0
    if args.available_only:
        availability = {path:path.is_file() for path in {dataset.archive_path(r) for r in pending}}
        ready = [r for r in pending if availability[dataset.archive_path(r)]]
        deferred = len(pending)-len(ready)
        pending = ready
    print('INPUT_READY '+json.dumps(dict(selected=len(rows),eligible=len(pending),completed_skipped=skipped,
                                         deferred_missing_archive=deferred)),flush=True)
    events = []
    processor = model = None
    model_load_sec = 0
    if pending:
        load_start = time.perf_counter()
        cache = BASE / 'models/huggingface'
        processor = WhisperProcessor.from_pretrained(config['model'], revision=revision, cache_dir=cache)
        model = WhisperForConditionalGeneration.from_pretrained(config['model'], revision=revision,
                    cache_dir=cache, torch_dtype=torch.float16, use_safetensors=True,
                    attn_implementation='sdpa').to('cuda').eval()
        model.generation_config.forced_decoder_ids = None
        torch.cuda.synchronize()
        model_load_sec = time.perf_counter()-load_start
        print('MODEL_READY ' + json.dumps(dict(revision=revision,model_load_sec=model_load_sec)),flush=True)
    opened = None
    archive = None
    interrupted = False
    transcript_cache = {}
    transcript_members = {}
    for row in pending:
        transcript_members.setdefault(dataset.archive_path(row), set()).add(row['transcript_member'])
    try:
        for row in pending:
            stage = 'loading'
            record = dict(row, utterance_id=key(row), reference=row['text'], prediction='', status='failed',
                          error='', stage=stage, loading_sec=0, inference_sec=0)
            t = time.perf_counter()
            try:
                path = dataset.archive_path(row)
                if opened != path:
                    if archive:
                        archive.close()
                    archive = None
                    archive = zipfile.ZipFile(path)
                    opened = path
                    # Read contiguous transcript entries once per ZIP, avoiding HDD
                    # seeks between distant WAV/TXT regions for every utterance.
                    transcript_cache = {member:archive.read(member).decode('utf-8-sig') for member in
                                        sorted(transcript_members[path],key=lambda m:archive.getinfo(m).header_offset)}
                payload = archive.read(row['audio_member'])
                reference = transcript_cache[row['transcript_member']]
                if reference.strip() != row['text'].strip():
                    raise ValueError('ZIP transcript differs from manifest')
                audio, rate = sf.read(io.BytesIO(payload), dtype='float32',always_2d=True)
                actual = len(audio)/rate
                if not 1 <= actual <= 30 or abs(actual-float(row['duration_sec'])) > 1e-5:
                    raise ValueError('Invalid/mismatched WAV duration')
                audio = audio.mean(axis=1)
                if rate != 16000:
                    divisor = math.gcd(rate,16000)
                    audio = resample_poly(audio,16000//divisor,rate//divisor).astype(np.float32)
                if not len(audio) or not np.isfinite(audio).all():
                    raise ValueError('Empty/non-finite audio')
                record.update(transcript_matches=True, source_sample_rate=rate,
                              wav_sha256=hashlib.sha256(payload).hexdigest())
                stage = 'features'
                inputs = processor(audio,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
                features = inputs.input_features.to('cuda',dtype=torch.float16)
                mask = inputs.attention_mask.to('cuda')
                torch.cuda.synchronize()
                record['loading_sec'] = time.perf_counter()-t
                stage = 'inference'
                torch.cuda.reset_peak_memory_stats()
                infer_start = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(features,attention_mask=mask,language='ko',task='transcribe',
                                               do_sample=False,num_beams=1,max_new_tokens=440,return_timestamps=False,
                                               return_dict_in_generate=True)
                torch.cuda.synchronize()
                record.update(inference_sec=time.perf_counter()-infer_start,
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                stage = 'decode'
                # Whisper's default tensor output strips EOS. The raw short-form
                # generation ModelOutput retains it, allowing real truncation checks.
                sequences = generated.sequences
                prediction = processor.batch_decode(sequences,skip_special_tokens=True)[0].strip()
                record['prediction'] = prediction
                if sequences[0,-1].item() != model.generation_config.eos_token_id:
                    raise ValueError('Generation token limit reached without EOS')
                stage = 'scoring'
                record.update(status='success',stage='complete',scores={'normalized':counts(row['text'],prediction),
                                                      'raw':counts(row['text'],prediction,False)})
                del features, mask, generated, sequences, inputs
            except Exception as exc:
                record.update(error=type(exc).__name__+': '+str(exc),stage=stage)
                if stage in ('loading','features'):
                    record['loading_sec'] = time.perf_counter()-t
                if isinstance(exc,torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            journal.save(record)
            events.append(dict(utterance_id=record['utterance_id'],status=record['status']))
            print(json.dumps(dict(done=len(events),utterance_id=record['utterance_id'],status=record['status'],
                                  inference_sec=record['inference_sec'],error=record['error']),ensure_ascii=False),flush=True)
            if len(events)%args.export_every == 0:
                journal.export(args.output,len(rows))
            if args.stop_after and len(events)>=args.stop_after:
                break
    except KeyboardInterrupt:
        interrupted = True
        print('Interrupted; committed utterances are preserved.',flush=True)
    finally:
        if archive:
            archive.close()
        scores = journal.export(args.output,len(rows))
        session = dict(started_utc=datetime.now(timezone.utc).isoformat(),processed=len(events),skipped=skipped,
                       deferred_missing_archive=deferred,
                       interrupted=interrupted,model_load_sec=model_load_sec,wall_sec=time.perf_counter()-started,
                       events=events,overall=scores['overall'],pending=scores['pending'])
        session_path = args.output / ('session_'+str(time.time_ns())+'.json')
        atomic_json(session_path,session)
        journal.db.close()
        print('SESSION '+json.dumps(session,ensure_ascii=False),flush=True)
    return 2 if scores['overall']['failed'] else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--revision',help='Immutable HF commit; first run resolves current commit if omitted')
    p.add_argument('--allow-full-test',action='store_true')
    p.add_argument('--available-only',action='store_true',help='Defer missing ZIPs without recording failures; keep the full immutable selection for future resume')
    p.add_argument('--retry-failures',action='store_true')
    p.add_argument('--stop-after',type=int)
    p.add_argument('--export-every',type=int,default=10)
    args = p.parse_args()
    if args.export_every < 1 or (args.stop_after is not None and args.stop_after<1):
        p.error('Counts must be positive')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
