"""Run E1 for one speaker from the repository root.

Example: python scripts/07_personalization/e1_train.py --config scripts/07_personalization/configs/e1.yaml --speaker AGS_M_28
"""
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.personalization.dataset import load_speaker_splits, WhisperSpeechDataset
from src.personalization.evaluator import compute_asr_metrics, decode_predictions, save_predictions
from src.personalization.trainer import build_model_and_processor, make_trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--speaker', required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    seed = int(config['training'].get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    paths = config['paths']
    output = Path(paths['output_root']).expanduser() / args.speaker
    if output.exists() and (output / 'test_metrics.json').exists():
        raise FileExistsError(f'Completed result already exists; refusing to overwrite: {output}')
    output.mkdir(parents=True, exist_ok=True)
    rows = load_speaker_splits(
        csv_path=paths['manifest_csv'], speaker_id=args.speaker,
        columns=config['columns'], labels=config['split_labels'],
        audio_root=paths['audio_root'], recording_col=config.get('recording_id_column'),
    )
    print('Speaker:', args.speaker, '| splits:', {k: len(v) for k,v in rows.items()})
    model, processor = build_model_and_processor(
        paths['general_model'], config.get('language', 'korean'), config.get('task', 'transcribe')
    )
    settings = config['training']
    datasets = {
        split: WhisperSpeechDataset(subset, processor, config['columns']['text'],
            max_audio_seconds=float(settings.get('max_audio_seconds', 30.0)),
            max_label_length=int(settings['generation_max_length']))
        for split, subset in rows.items()
    }
    trainer = make_trainer(model, processor, datasets['train'], datasets['val'], output / 'checkpoints', settings)
    trainer.train()
    # The loaded best checkpoint is chosen ONLY using validation CER.
    print('Best validation checkpoint:', trainer.state.best_model_checkpoint)
    best_dir = output / 'best_model'
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    pred = trainer.predict(datasets['test'], metric_key_prefix='test')
    metrics = compute_asr_metrics(pred.predictions, pred.label_ids, processor.tokenizer)
    references, hypotheses = decode_predictions(pred.predictions, pred.label_ids, processor.tokenizer)
    save_predictions(output, rows['test'], references, hypotheses, metrics)
    summary = {
        'speaker_id': args.speaker, 'splits': {k: len(v) for k,v in rows.items()},
        'best_checkpoint': trainer.state.best_model_checkpoint,
        'best_validation_cer': trainer.state.best_metric,
        'test_cer': metrics['cer'], 'test_wer': metrics['wer'],
        'full_finetuning': True, 'general_model': paths['general_model'],
        'seed': seed, 'training': settings,
    }
    (output / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()