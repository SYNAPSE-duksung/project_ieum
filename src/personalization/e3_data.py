"""Join fixed General hypotheses to Train only; preserve E1 audio/labels."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from .dataset import WhisperSpeechDataset, WhisperCollator
from .evaluator import normalize_text


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def join_hypotheses(rows, csv_path, profile_path, general_path, processor, max_length):
    """Validate provenance metadata/keys/references before accepting cached text."""
    profile = json.loads(Path(profile_path).read_text(encoding='utf-8'))
    info = profile['experiment_info']
    if Path(info['model_dir']).resolve() != Path(general_path).resolve():
        raise ValueError('Profile General model path does not match E1 General checkpoint')
    if info.get('profile_source') != 'whisper_predictions' or profile.get('split') != 'profile_train':
        raise ValueError('Expected General Whisper profile_train predictions')
    speaker = rows[0]['speaker_id'].strip()
    if profile['speaker_id'] != speaker:
        raise ValueError('Profile speaker mismatch')
    cache = {}
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'speaker_id', 'batch', 'audio_member', 'reference', 'prediction'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Unexpected General prediction CSV schema')
        for row in reader:
            if row['speaker_id'].strip() != speaker:
                continue
            key = (row['batch'].strip(), row['audio_member'].strip().replace('\\', '/'))
            if key in cache:
                raise ValueError(f'Duplicate cached hypothesis: {key}')
            cache[key] = row
    expected = {(r['batch'].strip(), r['audio_member'].strip().replace('\\', '/')) for r in rows}
    if set(cache) != expected or profile['utterance_count'] != len(rows):
        raise ValueError('Cache must match exactly the selected Train utterances; no val/test candidates allowed')
    joined, counts, excluded = [], Counter(), []
    for row in rows:
        if row['personal_split'].strip() != 'profile_train':
            raise ValueError('Correction hypotheses are Train-only')
        key = (row['batch'].strip(), row['audio_member'].strip().replace('\\', '/'))
        cached = cache[key]
        if cached['reference'] != row['text'].strip():
            raise ValueError(f'Cached reference mismatch: {key}')
        text = cached['prediction']
        ids = processor.tokenizer(text.strip(), add_special_tokens=True).input_ids
        reason = ('empty' if not normalize_text(text) else
                  'same' if normalize_text(text) == normalize_text(row['text']) else
                  'too_long' if len(ids) > max_length else 'eligible')
        counts[reason] += 1
        if reason in ('empty', 'too_long'):
            excluded.append(dict(audio=row['_resolved_audio'], prediction=text, reason=reason, tokens=len(ids)))
        joined.append(dict(row, _general_hypothesis=text, _correction_reason=reason))
    report = dict(speaker_id=speaker, train_utterances=len(rows), counts=dict(counts), exclusions=excluded,
                  candidate_csv=str(Path(csv_path).resolve()), candidate_csv_sha256=file_hash(csv_path),
                  profile_json_sha256=file_hash(profile_path),
                  source_model=str(Path(general_path).resolve()),
                  source_provenance='generator code + Profile metadata; historical CSV has no model checksum',
                  equality_policy='E1 normalize_text including whitespace; not CER space removal',
                  empty_or_long_policy='exclude only from correction, retain ASR; never truncate candidates')
    return joined, report


class E3Dataset(WhisperSpeechDataset):
    def __getitem__(self, index):
        result = super().__getitem__(index)
        row = self.rows[index]
        if '_correction_reason' in row:
            eligible = row['_correction_reason'] == 'eligible'
            result['correction_eligible'] = eligible
            result['correction_labels'] = self.processor.tokenizer(
                row['_general_hypothesis'].strip(), add_special_tokens=True).input_ids if eligible else []
        return result


class E3Collator(WhisperCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        if not any('correction_eligible' in f for f in features):
            return batch  # Validation/Test carry no hypotheses.
        eligible = torch.tensor([f.get('correction_eligible', False) for f in features], dtype=torch.bool)
        labels = [f.get('correction_labels', []) for f in features]
        width = max(1, max(map(len, labels)))
        padded = torch.full((len(features), width), -100, dtype=torch.long)
        for i, ids in enumerate(labels):
            if ids:
                padded[i, :len(ids)] = torch.tensor(ids)
        # Same rule as E1 WhisperCollator; no alternative shifting convention.
        bos = self.processor.tokenizer.bos_token_id
        if bos is not None and eligible.any() and (padded[eligible, 0] == bos).all():
            padded = padded[:, 1:]
        batch.update(correction_labels=padded, correction_eligible=eligible)
        return batch
