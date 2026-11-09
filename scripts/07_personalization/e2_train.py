"""E2 entry point. Default is preflight only; training requires --mode train.

Each lambda run starts from E1's General checkpoint. Training evaluates only
validation. --mode finalize requires an explicit reviewed candidate list.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import torch
import yaml
from transformers import WhisperConfig, WhisperProcessor, set_seed
from transformers.models.whisper.modeling_whisper import WhisperEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.personalization.ctc_head import WhisperWithAuxiliaryCTC
from src.personalization.ctc_utils import (
    VOCAB_SIZE, ID_TO_TOKEN, TOKEN_TO_ID, E2SpeechDataset, E2Collator,
    encode_ctc, decode_ctc, minimum_ctc_frames, preflight_splits,
)
from src.personalization.dataset import load_speaker_splits
from src.personalization.e2_trainer import build_e2_model_and_processor, make_e2_trainer
from src.personalization.evaluator import compute_asr_metrics, decode_predictions, save_predictions


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    e2 = yaml.safe_load(path.read_text(encoding='utf-8'))
    e1_path = (path.parent / e2['e1_config']).resolve()
    config = yaml.safe_load(e1_path.read_text(encoding='utf-8'))
    if e2['unsupported_character_policy'] != 'error':
        raise ValueError('Only explicit unsupported-character policy "error" is implemented')
    config['e2'] = e2
    return config


def fingerprint(config, speaker):
    """Prevent lambda comparisons across different data/settings/checkpoints."""
    identity_config = json.loads(json.dumps(config))
    # Expanding a future search grid must not invalidate otherwise identical runs.
    identity_config['e2'].pop('lambda_ctc_values', None)
    payload = dict(config=identity_config, speaker=speaker,
                   manifest_sha256=hashlib.sha256(project_path(config['paths']['manifest_csv']).read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def lambda_value(value):
    try:
        weight = float(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('lambda must be a finite nonnegative number') from exc
    if not math.isfinite(weight) or weight < 0:
        raise argparse.ArgumentTypeError('lambda must be a finite nonnegative number')
    return 0.0 if weight == 0 else weight


def lambda_directory(weight):
    return f'lambda_{lambda_value(weight)}'


def general_identity(path):
    path = project_path(path)
    weights = sorted(path.glob('*.safetensors')) or sorted(path.glob('pytorch_model*.bin'))
    if not weights:
        raise FileNotFoundError(f'No General checkpoint weights: {path}')
    result = {}
    for file in [path / 'config.json', *weights]:
        with file.open('rb') as stream:
            result[file.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path.resolve()), sha256=result)


def load_data(config, speaker, processor, encoder):
    paths = config['paths']
    rows = load_speaker_splits(
        project_path(paths['manifest_csv']), speaker, config['columns'], config['split_labels'],
        project_path(paths['audio_root']), config.get('recording_id_column'))
    counts = {k: len(v) for k, v in rows.items()}
    if speaker == config['e2']['pilot_speaker'] and counts != config['e2']['expected_pilot_splits']:
        raise ValueError(f'Unexpected pilot split counts: {counts}')
    report = preflight_splits(rows, processor, encoder, config['columns']['text'], config['training'])
    settings = config['training']
    datasets = {split: E2SpeechDataset(
        subset, processor, config['columns']['text'], encoder=encoder,
        max_audio_seconds=float(settings.get('max_audio_seconds', 30)),
        max_label_length=int(settings['generation_max_length'])) for split, subset in rows.items()}
    return rows, datasets, report


def check_vocabulary():
    assert len(ID_TO_TOKEN) == len(TOKEN_TO_ID) == VOCAB_SIZE == 11183
    assert set(TOKEN_TO_ID.values()) == set(range(VOCAB_SIZE))
    assert TOKEN_TO_ID['가'] == 1 and TOKEN_TO_ID['힣'] == 11172
    assert TOKEN_TO_ID['0'] == 11173 and TOKEN_TO_ID['9'] == 11182
    assert decode_ctc(encode_ctc('700회 당첨자 수 알려 줘')) == '700회당첨자수알려줘'
    assert decode_ctc(encode_ctc('내일 4시까지')) == '내일4시까지'
    assert minimum_ctc_frames(encode_ctc('700')) == 4
    print('PASS vocabulary: 11183 unique IDs, Hangul/digit round trips, 700 needs 4 frames')


def smoke_test(model, processor, datasets, report, config, device):
    """One real batch, no optimizer step; separate loss gradients + generation."""
    model.to(device)
    model.train()
    # Non-reentrant checkpointing allows independent autograd.grad probes.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    digit_indices = [i for i, row in enumerate(datasets['train'].rows)
                     if any(c in '0123456789' for c in row[config['columns']['text']])]
    # Include 700's repeated zero in the actual model batch, not only unit tests.
    repeated = [i for i in digit_indices if '700' in datasets['train'].rows[i][config['columns']['text']]]
    indices = list(dict.fromkeys(repeated + digit_indices))[:2]
    if len(indices) < 2:
        indices = list(range(min(2, len(datasets['train']))))
    batch = E2Collator(processor)([datasets['train'][i] for i in indices])
    batch = {k: v.to(device) for k, v in batch.items()}
    output = model(**batch)
    encoder_probe = model.model.encoder.conv1.weight
    asr_grad = torch.autograd.grad(output.asr_loss, encoder_probe, retain_graph=True)[0]
    ctc_grad = torch.autograd.grad(output.ctc_loss, encoder_probe, retain_graph=True)[0]
    assert torch.isfinite(asr_grad).all() and asr_grad.abs().sum() > 0
    assert torch.isfinite(ctc_grad).all() and ctc_grad.abs().sum() > 0
    assert torch.allclose(output.loss, output.asr_loss + model.config.lambda_ctc * output.ctc_loss)
    results = dict(device=str(device), batch_size=len(indices),
                   batch_text=[datasets['train'].rows[i][config['columns']['text']] for i in indices],
                   input_features=list(batch['input_features'].shape),
                   ctc_input_lengths=batch['ctc_input_lengths'].tolist(),
                   ctc_target_lengths=batch['ctc_target_lengths'].tolist(),
                   asr_loss=float(output.asr_loss.detach()), ctc_loss=float(output.ctc_loss.detach()),
                   weighted_ctc_loss=float(output.weighted_ctc_loss.detach()), total_loss=float(output.loss.detach()),
                   asr_encoder_gradient_norm=float(asr_grad.norm()),
                   ctc_encoder_gradient_norm=float(ctc_grad.norm()),
                   digit_train_utterances=len(report['digit_train']))
    del asr_grad, ctc_grad
    output.loss.backward()
    for name, parameter in [('encoder', encoder_probe), ('ctc_head', model.ctc_head.weight),
                            ('decoder', model.model.decoder.layers[0].fc1.weight)]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        results[name + '_total_gradient_norm'] = float(parameter.grad.norm())
    model.zero_grad(set_to_none=True)
    del output
    model.gradient_checkpointing_disable()
    model.eval()
    # No test split access for smoke inference; same training batch only.
    with torch.no_grad():
        generated = model.generate(**batch, max_new_tokens=8)
    assert generated.ndim == 2 and generated.shape[0] == len(indices)
    results['decoder_smoke_text'] = processor.batch_decode(generated, skip_special_tokens=True)
    # Save the COMPLETE model through Trainer, then reload it as the E2 class.
    # Temporary artifacts are isolated from all experiment/checkpoint directories.
    with tempfile.TemporaryDirectory(prefix='ieum_e2_smoke_') as temporary:
        trainer = make_e2_trainer(model, processor, datasets, Path(temporary) / 'trainer',
                                  config['training'], use_cpu=str(device) == 'cpu')
        checkpoint = Path(temporary) / 'checkpoint'
        trainer.save_model(str(checkpoint))
        restored = WhisperWithAuxiliaryCTC.from_pretrained(checkpoint, local_files_only=True)
        assert torch.equal(restored.ctc_head.weight.cpu(), model.ctc_head.weight.detach().cpu())
        assert torch.equal(restored.ctc_head.bias.cpu(), model.ctc_head.bias.detach().cpu())
        assert restored.config.lambda_ctc == model.config.lambda_ctc
        results['ctc_checkpoint_round_trip'] = True
        restored.eval()
        with torch.no_grad():
            restored_ids = restored.generate(**{k: v.cpu() for k, v in batch.items()}, max_new_tokens=8)
        assert torch.equal(generated.cpu(), restored_ids)
        del restored
        # Exercise the actual Trainer generation+loss route, not only generate().
        trainer.model.eval()
        loss, prediction, labels = trainer.prediction_step(
            trainer.model, batch, False, max_new_tokens=8)
        assert torch.isfinite(loss) and prediction.ndim == 2 and labels.ndim == 2
        metrics = compute_asr_metrics(prediction.cpu().numpy(), labels.cpu().numpy(), processor.tokenizer)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
        results['trainer_prediction_step'] = True
        results['metrics_function'] = 'src.personalization.evaluator.compute_asr_metrics (unchanged E1)'
    print('SMOKE PASS (no optimizer step / no training run)\n' + json.dumps(results, ensure_ascii=False, indent=2))


def train_one(config, speaker, weight, model, processor, rows, datasets, report, run_dir):
    # Never overwrite or resume an existing lambda run implicitly.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    identity = general_identity(config['paths']['general_model'])
    (run_dir / 'initialization.json').write_text(json.dumps(dict(
        source=identity, seed=config['training']['seed'], lambda_ctc=weight,
        resume_from_checkpoint=False, ctc_head='newly initialized'), indent=2), encoding='utf-8')
    trainer = make_e2_trainer(model, processor, datasets, run_dir / 'checkpoints', config['training'], run_dir=run_dir)
    try:
        trainer.train()  # Only this explicit mode starts training; never resume.
    except BaseException as exc:
        trainer.run_recorder.finish(trainer.args, 'failed', exc)
        raise
    if trainer.state.best_model_checkpoint is None:
        raise RuntimeError('No validation-selected checkpoint; refusing final evaluation')
    best_dir = run_dir / 'best_model'
    trainer.save_model(str(best_dir))
    validation = trainer.evaluate(metric_key_prefix='eval')
    if not all(torch.isfinite(torch.tensor(validation[k])) for k in ('eval_cer', 'eval_wer')):
        raise FloatingPointError('Invalid validation metrics')
    summary = dict(status='validation_complete', speaker_id=speaker, lambda_ctc=weight,
                   comparison_fingerprint=fingerprint(config, speaker),
                   general_model=str(project_path(config['paths']['general_model'])),
                   splits={k: len(v) for k, v in rows.items()}, training=config['training'],
                   best_checkpoint=trainer.state.best_model_checkpoint,
                   best_model=str(best_dir.resolve()), best_validation_cer=trainer.state.best_metric,
                   validation_metrics=validation, test_evaluated=False,
                   best_epoch=trainer.run_recorder.report.get('best_epoch'),
                   initialization=identity,
                   training_seconds=trainer.run_recorder.report['training_seconds'],
                   gpu_peak_allocated_bytes=trainer.run_recorder.report['gpu_peak_allocated_bytes'],
                   gpu_peak_reserved_bytes=trainer.run_recorder.report['gpu_peak_reserved_bytes'],
                   nonfinite_observed=trainer.run_recorder.report['nonfinite_observed'])
    trainer.run_recorder.finish(trainer.args, 'validation_complete')
    trainer.save_state()
    (run_dir / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def finalize(config, speaker, root, weights):
    candidates = []
    for weight in weights:
        summary_path = root / lambda_directory(weight) / 'run_summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if (summary['status'] != 'validation_complete'
                or summary['comparison_fingerprint'] != fingerprint(config, speaker)
                or summary['lambda_ctc'] != weight):
            raise ValueError(f'Incompatible or incomplete run: {summary_path}')
        candidates.append(summary)
    if len({json.dumps(r['initialization'], sort_keys=True) for r in candidates}) != 1:
        raise ValueError('Candidate General checkpoint identities differ')
    selected = min(candidates, key=lambda r: (r['best_validation_cer'], r['lambda_ctc']))
    output = root / 'selected_test'
    if output.exists():
        raise FileExistsError(f'Test evaluation output already exists: {output}')
    model = WhisperWithAuxiliaryCTC.from_pretrained(selected['best_model'], local_files_only=True)
    processor = WhisperProcessor.from_pretrained(selected['best_model'], local_files_only=True)
    rows, datasets, _ = load_data(config, speaker, processor, model.model.encoder)
    # Persist the decision BEFORE reading test predictions. Test cannot select lambda.
    output.mkdir(parents=True, exist_ok=False)
    decision = dict(selection='minimum validation CER; tie: smaller lambda',
                    lambda_ctc=selected['lambda_ctc'], best_model=selected['best_model'],
                    candidates=[{k: r[k] for k in ('lambda_ctc', 'best_validation_cer')} for r in candidates])
    (output / 'selection.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    trainer = make_e2_trainer(model, processor, datasets, output / 'trainer', config['training'])
    prediction = trainer.predict(datasets['test'], metric_key_prefix='test')
    metrics = compute_asr_metrics(prediction.predictions, prediction.label_ids, processor.tokenizer)
    references, hypotheses = decode_predictions(prediction.predictions, prediction.label_ids, processor.tokenizer)
    save_predictions(output, rows['test'], references, hypotheses, metrics)
    print(json.dumps(dict(**decision, test_metrics=metrics), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/e2.yaml')
    parser.add_argument('--speaker', default='KSM_M_45')
    parser.add_argument('--lambda-ctc', type=lambda_value, default=0.1,
                        help='finite nonnegative weight; phase 1: 0.03, 0.1, 0.3; other values allowed')
    parser.add_argument('--selection-lambdas', nargs='+', type=lambda_value,
                        help='explicit reviewed candidate set, required ONLY for finalize')
    parser.add_argument('--mode', choices=('validate', 'smoke', 'train', 'finalize'), default='validate')
    parser.add_argument('--smoke-device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if args.mode == 'finalize' and not args.selection_lambdas:
        parser.error('finalize requires --selection-lambdas after reviewing validation results')
    if args.selection_lambdas and len(set(args.selection_lambdas)) != len(args.selection_lambdas):
        parser.error('--selection-lambdas must be distinct')
    torch.set_num_threads(args.cpu_threads)
    config = load_config(args.config)
    set_seed(int(config['training'].get('seed', 42)))
    check_vocabulary()
    root = project_path(config['e2']['output_root']) / args.speaker
    if args.mode == 'finalize':
        finalize(config, args.speaker, root, args.selection_lambdas)
        return
    if args.mode == 'validate':
        processor = WhisperProcessor.from_pretrained(
            'openai/whisper-small', language=config.get('language', 'korean'),
            task=config.get('task', 'transcribe'), local_files_only=True)
        with torch.device('meta'):
            encoder = WhisperEncoder(WhisperConfig.from_pretrained(
                project_path(config['paths']['general_model']), local_files_only=True))
        load_data(config, args.speaker, processor, encoder)
        return
    run_dir = root / lambda_directory(args.lambda_ctc)
    if args.mode == 'train' and run_dir.exists():
        raise FileExistsError(f'Refusing to overwrite/resume existing run: {run_dir}')
    model, processor = build_e2_model_and_processor(
        project_path(config['paths']['general_model']), args.lambda_ctc,
        config.get('language', 'korean'), config.get('task', 'transcribe'))
    rows, datasets, report = load_data(config, args.speaker, processor, model.model.encoder)
    if args.mode == 'smoke':
        smoke_test(model, processor, datasets, report, config, args.smoke_device)
    else:
        train_one(config, args.speaker, args.lambda_ctc, model, processor, rows, datasets, report, run_dir)


if __name__ == '__main__':
    main()
