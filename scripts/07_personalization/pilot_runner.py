"""Frozen E1/E3 pilot; no Test evaluation in this entry point."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import yaml
import torch
from transformers import set_seed
from transformers.trainer_utils import SaveStrategy
from src.personalization.trainer import build_model_and_processor, make_trainer
from src.personalization.dataset import load_speaker_splits, WhisperSpeechDataset
from src.personalization.e3_data import join_hypotheses, E3Dataset
from src.personalization.e3_trainer import make_e3_trainer, E3Audit

FROZEN = ROOT / 'reports/e1_e3_multispeaker_pilot_20261107'
GIB = 1024 ** 3


class PilotAudit(E3Audit):
    def on_log(self, args, state, control, logs=None, **kwargs):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(logs or {}, step=state.global_step), allow_nan=False)+'\n')

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads and not torch.isfinite(torch.stack(torch._foreach_norm(grads))).all():
            self.failure('Nonfinite gradient before optimizer step')
            raise FloatingPointError('Nonfinite gradient')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def frozen():
    fingerprint = json.loads((FROZEN / 'selection_fingerprint.json').read_text(encoding='utf-8'))
    for name, expected in fingerprint['hashes'].items():
        if digest(FROZEN / name) != expected:
            raise ValueError(f'Frozen selection changed: {name}')
    for name, expected in fingerprint['source_hashes'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Protected source changed: {name}')
    with (FROZEN / 'pilot_selection.csv').open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def config():
    return yaml.safe_load((ROOT / 'scripts/07_personalization/configs/e1.yaml').read_text(encoding='utf-8'))


def storage(out):
    model = Path(config()['paths']['general_model'])
    weight = sum(p.stat().st_size for p in model.glob('*.safetensors'))
    if not weight:
        raise ValueError('General safetensors missing')
    used = sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) if out.exists() else 0
    probe = out
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    # One canonical best per job; old + new best coexist during rotation.
    return dict(weight_gib=weight/GIB, final_20_models_gib=20*weight/GIB,
                conservative_peak_gib=(22*weight+GIB)/GIB,
                required_with_reserve_gib=(22*weight+6*GIB)/GIB,
                existing_new_root_gib=used/GIB, free_gib=free/GIB,
                budget_gib=30, reserve_gib=5)


def guard(out):
    plan = storage(out)
    completed = len(list(out.glob('*/*/attempt-*/summary.json')))
    additional = (max(0, 20-completed)+2)*plan['weight_gib']+1+5
    if plan['existing_new_root_gib'] + additional > 30 or plan['free_gib'] < additional:
        raise OSError(f'Insufficient storage; use a separate volume: {plan}, additional={additional}')
    return plan


def safe_root(out):
    out = out.resolve()
    # Dedicated new namespace only, including on an alternate volume.
    if out.name != 'pilot_e1_e3_v1' or out.is_symlink():
        raise ValueError('Output directory must be a dedicated pilot_e1_e3_v1 directory')
    forbidden = [ROOT/'checkpoints', ROOT/'data', ROOT/'outputs/personalization/E1',
                 ROOT/'outputs/personalization/E1_v2', ROOT/'outputs/personalization/E2', ROOT/'outputs/personalization/E3']
    if any(out == p.resolve() or p.resolve() in out.parents for p in forbidden):
        raise ValueError('Protected output path')
    marker = out / 'pilot_manifest.json'
    identity = dict(selection=digest(FROZEN/'pilot_selection.csv'),
                    config=digest(ROOT/'scripts/07_personalization/configs/e1.yaml'),
                    general=digest(Path(config()['paths']['general_model'])/'model.safetensors'),
                    runner=digest(__file__), lambda_correction=.1, margin=.1, mode='legacy_e1')
    if out.exists() and not marker.exists():
        raise ValueError('Refusing existing unmarked directory')
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != identity:
            raise ValueError('Run identity changed; choose a new versioned implementation/output')
    else:
        write(marker, identity)
    return out


def data(speaker, processor, cfg):
    rows = load_speaker_splits(cfg['paths']['manifest_csv'], speaker, cfg['columns'],
                              cfg['split_labels'], cfg['paths']['audio_root'], cfg['recording_id_column'])
    selected = next(r for r in frozen() if r['speaker_id'] == speaker)
    for split in rows:
        if len(rows[split]) != int(selected[split+'_count']):
            raise ValueError(f'Split count mismatch {speaker}/{split}')
    joined, report = join_hypotheses(rows['train'], ROOT/'outputs/personalization/error_profiles/predictions.csv',
        ROOT/f'outputs/personalization/error_profiles/profiles/{speaker}.json',
        cfg['paths']['general_model'], processor, cfg['training']['generation_max_length'])
    if report['counts'].get('eligible', 0) != int(selected['usable_correction_candidates']):
        raise ValueError('Candidate count changed')
    return rows, joined, report


def datasets(rows, joined, processor, cfg, experiment):
    kw = dict(processor=processor, text_col=cfg['columns']['text'],
              max_audio_seconds=30., max_label_length=160)
    return dict(train=(E3Dataset(joined, **kw) if experiment == 'E3' else WhisperSpeechDataset(rows['train'], **kw)),
                val=WhisperSpeechDataset(rows['val'], **kw))


def trainer(experiment, model, processor, ds, directory, cfg, audit):
    settings = cfg['training']
    if experiment == 'E1':
        t = make_trainer(model, processor, ds['train'], ds['val'], directory, settings)
        t.add_callback(audit)
        def observe(module, args, kwargs, output):
            if module.training and kwargs.get('labels') is not None:
                loss = float(output.loss.detach())
                if not torch.isfinite(output.loss):
                    raise FloatingPointError('Nonfinite ASR loss')
                audit.add(dict(asr_loss=loss, correction_loss=0., total_loss=loss,
                               eligible=0, used=0, active=0, reference_score=None, hypothesis_score=None),
                          kwargs['labels'].shape[0])
        model.register_forward_hook(observe, with_kwargs=True)
    else:
        t = make_e3_trainer(model, processor, ds, directory, settings, .1, .1, 'legacy_e1', audit=audit)
    if not t.model_accepts_loss_kwargs:
        raise ValueError('Historical E1 accumulation no longer matches installed Trainer')
    t.args.save_strategy = SaveStrategy.BEST
    t.args.save_only_model = True
    t.args.save_total_limit = 1
    return t


def run_job(out, speaker, experiment):
    cfg = config()
    job = out / speaker / experiment
    job.mkdir(parents=True, exist_ok=True)
    attempt = job / f'attempt-{len(list(job.glob("attempt-*")))+1:03d}'
    attempt.mkdir()
    status = job/'status.json'
    write(status, dict(status='running', attempt=str(attempt), pid=os.getpid()))
    audit = PilotAudit(attempt)
    try:
        guard(out)
        set_seed(42)
        model, processor = build_model_and_processor(cfg['paths']['general_model'])
        rows, joined, report = data(speaker, processor, cfg)
        write(attempt/'candidate_audit.json', report)
        t = trainer(experiment, model, processor, datasets(rows, joined, processor, cfg, experiment), attempt/'checkpoints', cfg, audit)
        write(attempt/'training_arguments.json', t.args.to_dict())
        write(attempt/'configuration.json', cfg)
        # Guards run before each new best is serialized, never remove old experiments.
        from transformers import TrainerCallback
        class DiskGuard(TrainerCallback):
            def on_evaluate(self, args, state, control, **kwargs):
                guard(out)
                for p in Path(args.output_dir).glob('checkpoint-*'):
                    if p.is_symlink() or (p.lstat().st_file_attributes & 1024) or p.resolve().parent != Path(args.output_dir).resolve():
                        raise ValueError('Unsafe checkpoint rotation path')
        t.add_callback(DiskGuard())
        t.train()
        if not t.state.best_model_checkpoint:
            raise RuntimeError('No best validation checkpoint')
        processor.save_pretrained(t.state.best_model_checkpoint)
        t.state.save_to_json(str(attempt/'trainer_state.json'))
        write(attempt/'summary.json', dict(speaker=speaker, experiment=experiment,
            best_model=t.state.best_model_checkpoint, best_validation_cer=t.state.best_metric,
            best_epoch=audit.report.get('best_epoch'), test_evaluated=False,
            candidates=report, audit=audit.report))
        write(status, dict(status='completed', attempt=str(attempt)))
    except BaseException as exc:
        write(status, dict(status='failed', attempt=str(attempt), error=repr(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['plan', 'preflight', 'smoke', 'train', 'worker'], default='plan')
    parser.add_argument('--output-root', type=Path, default=ROOT/'outputs/personalization/pilot_e1_e3_v1')
    parser.add_argument('--speakers', nargs='+')
    parser.add_argument('--experiments', nargs='+', choices=['E1', 'E3'], default=['E1', 'E3'])
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--skip-failed', action='store_true')
    a = parser.parse_args()
    os.chdir(ROOT)
    selected = frozen()
    speakers = a.speakers or [r['speaker_id'] for r in selected]
    if not set(speakers) <= {r['speaker_id'] for r in selected}:
        raise ValueError('Only frozen pilot speakers are allowed')
    if a.mode == 'plan':
        print(json.dumps(storage(a.output_root), indent=2)); return
    if a.mode in ('preflight', 'smoke'):
        from src.personalization.pilot_checks import check
        check(a.mode, speakers); return
    out = safe_root(a.output_root)
    if a.mode == 'worker':
        run_job(out, speakers[0], a.experiments[0]); return
    lock = out/'runner.lock'
    # An interrupted parent leaves the lock for explicit operator inspection.
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    try:
        guard(out)
        for speaker in speakers:
            for exp in a.experiments:
                status = out/speaker/exp/'status.json'
                old = json.loads(status.read_text(encoding='utf-8')) if status.exists() else {}
                if old.get('status') == 'completed':
                    continue
                if old and not a.retry_failed:
                    if a.skip_failed:
                        continue
                    raise RuntimeError(f'Incomplete job {speaker}/{exp}; use --retry-failed or --skip-failed')
                cmd = [sys.executable, __file__, '--mode', 'worker', '--output-root', str(out),
                       '--speakers', speaker, '--experiments', exp]
                logdir = out/speaker/exp
                logdir.mkdir(parents=True, exist_ok=True)
                with (logdir/f'console-{time.time_ns()}.log').open('w', encoding='utf-8') as f:
                    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
