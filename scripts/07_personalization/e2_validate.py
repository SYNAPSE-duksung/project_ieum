"""Read-only E2 syllable-CTC preflight using E1's manifest and normalization.

Run from any directory with the project's Python environment. No model weights,
alignment, training, downloads, or output files are needed. Exit: 0 clean,
1 utterance/data issues, 2 configuration/runtime failure.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import soundfile as sf
import torch
import yaml
from transformers import WhisperConfig, WhisperFeatureExtractor
from transformers.models.whisper.modeling_whisper import WhisperEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.personalization.evaluator import normalize_text


def conv_length(length, conv):
    """PyTorch Conv1d output length, including stride/padding/dilation."""
    return (length + 2 * conv.padding[0]
            - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1


def frame_lengths(samples, sample_rate, extractor, encoder):
    # librosa.resample's default fix=True uses ceil(target_sr * n / orig_sr).
    resampled = (samples * extractor.sampling_rate + sample_rate - 1) // sample_rate
    kept = min(resampled, extractor.n_samples)
    # E1 calls the extractor with default max_length padding and truncation.
    # Its sample attention mask is sliced at ::hop_length (not floor(n/hop)).
    # STFT drops its last frame; padded input is exactly n_samples long.
    padded_mel = extractor.n_samples // extractor.hop_length
    valid_mel = min((kept + extractor.hop_length - 1) // extractor.hop_length,
                    padded_mel)
    def encode(n):
        return conv_length(conv_length(n, encoder.conv1), encoder.conv2) if n else 0
    return resampled, valid_mel, encode(valid_mel), encode(padded_mel)


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate(args):
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    columns, labels, paths = config['columns'], config['split_labels'], config['paths']
    manifest = project_path(paths['manifest_csv'])
    audio_root = project_path(paths['audio_root'])
    # Same processor source as E1's build_model_and_processor; offline only.
    extractor = WhisperFeatureExtractor.from_pretrained(
        'openai/whisper-small', local_files_only=True)
    model_config = WhisperConfig.from_pretrained(
        project_path(paths['general_model']), local_files_only=True)
    # Inspect the installed implementation's actual convolutions, without weights.
    with torch.device('meta'):
        encoder = WhisperEncoder(model_config)
    if extractor.padding_side != 'right' or extractor.n_samples % extractor.hop_length:
        raise ValueError('Unsupported feature extractor padding/grid; review length calculation')
    if extractor.nb_max_frames != model_config.max_source_positions * encoder.conv1.stride[0] * encoder.conv2.stride[0]:
        raise ValueError('Feature extractor and encoder input lengths disagree')
    split_names = ('train', 'val', 'test')
    split_lookup = {str(labels[k]).lower(): k for k in split_names}
    if len(split_lookup) != 3:
        raise ValueError('Split labels must be distinct')
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = set(columns[k] for k in ('speaker', 'split', 'audio', 'text')) | {'batch'}
        recording_col = config.get('recording_id_column')
        if recording_col:
            required.add(recording_col)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}; actual={reader.fieldnames}')
        print('Manifest:', manifest)
        print('CSV columns:', ', '.join(reader.fieldnames))
        rows = [(line, row) for line, row in enumerate(reader, 2)
                if row[columns['speaker']].strip() == args.speaker]
    print('Audio path: audio_root / batch / audio_member; root:', audio_root)
    print('Normalization: E1 normalize_text (NFC, lowercase, punctuation -> space), then remove spaces')
    print(f'Input: {extractor.sampling_rate} Hz, hop={extractor.hop_length}, '
          f'padding=max_length, truncation=True, max_samples={extractor.n_samples}')
    print(f'Encoder conv1={encoder.conv1}; conv2={encoder.conv2}')
    print('CTC requirement: target length + adjacent equal pairs <= valid encoder frames')
    print('Valid frames use feature attention-mask length; padded frames are NOT CTC input length.')
    counts, issue_counts = Counter(), Counter()
    stats = {k: [] for k in split_names}
    seen = {}
    affected = 0
    max_samples = int(float(config['training'].get('max_audio_seconds', 30)) * extractor.sampling_rate)
    for line, row in rows:
        issues = []
        split = split_lookup.get(row[columns['split']].strip().lower(), 'unknown')
        counts[split] += 1
        if split == 'unknown':
            issues.append('unknown split: ' + repr(row[columns['split']]))
        raw = row[columns['text']]
        target = normalize_text(raw).replace(' ', '')
        invalid = Counter(c for c in target if not '\uac00' <= c <= '\ud7a3')
        if invalid:
            issues.append('non-Hangul target characters: ' + ', '.join(
                f'{c!r} U+{ord(c):04X} x{n}' for c, n in sorted(invalid.items())))
        if not target:
            issues.append('empty CTC target after normalization')
        repeats = sum(a == b for a, b in zip(target, target[1:]))
        required_frames = len(target) + repeats
        member = Path(row[columns['audio']]).expanduser()
        audio = (member if member.is_absolute() else audio_root / row['batch'] / member).resolve()
        if audio in seen:
            issues.append(f'duplicate audio; first CSV line={seen[audio]}')
        seen[audio] = line
        if recording_col and not row[recording_col].strip():
            issues.append('empty recording ID')
        detail = {}
        try:
            info = sf.info(str(audio))
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError('empty audio or invalid sample rate')
            n, mel, valid, padded = frame_lengths(info.frames, info.samplerate, extractor, encoder)
            detail = dict(seconds=round(info.frames / info.samplerate, 6),
                          sample_rate=info.samplerate, channels=info.channels,
                          resampled_samples=n, valid_mel=mel,
                          valid_encoder=valid, padded_encoder=padded)
            if split in stats:
                stats[split].append((detail['seconds'], len(target), repeats, valid))
            if n > max_samples:
                issues.append(f'E1 rejects audio exceeding max_audio_seconds ({n} > {max_samples} samples)')
            if n > extractor.n_samples:
                issues.append(f'feature extractor would truncate {n - extractor.n_samples} samples; full transcript retained')
            if required_frames > valid:
                issues.append(f'CTC unalignable: {len(target)} labels + {repeats} repeats = {required_frames} > {valid} valid frames')
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(f'WAV inspection failed: {exc}')
        if issues:
            affected += 1
            for issue in issues:
                issue_counts[issue.split(':', 1)[0]] += 1
            print('ISSUE ' + json.dumps(dict(
                csv_line=line, split=split, recording_id=row.get(recording_col),
                audio=str(audio), raw_text=raw, ctc_target=target,
                target_length=len(target), adjacent_repeats=repeats,
                required_frames=required_frames, **detail, reasons=issues), ensure_ascii=False))
    print('\nSummary:', args.speaker)
    for split in split_names:
        print(f'  {split} ({labels[split]}): {counts[split]} utterances')
        values = stats[split]
        if values:
            for index, name in enumerate(('WAV seconds', 'target syllables/characters', 'adjacent repeats', 'valid encoder frames')):
                numbers = [v[index] for v in values]
                print(f'    {name}: min={min(numbers)} max={max(numbers)}')
    if counts['unknown']:
        print('  unknown split:', counts['unknown'])
    empty = [s for s in split_names if not counts[s]]
    if empty:
        print('ERROR: empty/missing splits:', ', '.join(empty))
    print(f'Total={len(rows)}; affected utterances={affected}; issues={sum(issue_counts.values())}')
    for reason, count in sorted(issue_counts.items()):
        print(f'  {reason}: {count}')
    print('No rows excluded or modified. Alignment feasibility is a length check, not forced alignment.')
    return 1 if affected or empty else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/e1.yaml')
    parser.add_argument('--speaker', default='KSM_M_45')
    args = parser.parse_args()
    try:
        return validate(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
