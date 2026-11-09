"""Shared speaker-level manifest loading and Whisper audio dataset."""
import csv
from pathlib import Path
import soundfile as sf
import torch
from torch.utils.data import Dataset
import librosa


def load_speaker_splits(csv_path, speaker_id, columns, labels, audio_root, recording_col=None):
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty manifest')
    required = [columns[k] for k in ('speaker', 'split', 'audio', 'text')]
    if recording_col:
        required.append(recording_col)
    missing = set(required) - set(rows[0])
    if missing:
        raise ValueError(f'Missing manifest columns: {sorted(missing)}; actual: {list(rows[0])}')
    speaker_rows = [r for r in rows if r[columns['speaker']].strip() == speaker_id]
    if not speaker_rows:
        raise ValueError(f'No rows for speaker {speaker_id}')
    result = {key: [] for key in ('train', 'val', 'test')}
    for row in speaker_rows:
        split = row[columns['split']].strip().lower()
        for key, value in labels.items():
            if split == str(value).lower():
                result[key].append(row)
                break
        else:
            raise ValueError(f'Unknown split for {speaker_id}: {split}')
    if any(len(v) == 0 for v in result.values()):
        raise ValueError(f'Empty split: { {k:len(v) for k,v in result.items()} }')
    root = Path(audio_root).expanduser()
    seen = {}
    for split, subset in result.items():
        for row in subset:
            path = Path(row[columns['audio']]).expanduser()
            full_path = (
                path if path.is_absolute()
                else root / row["batch"] / path
            ).resolve()
            if not full_path.is_file():
                raise FileNotFoundError(full_path)
            if full_path in seen:
                raise ValueError(f'Duplicate audio file across or within splits: {full_path} ({seen[full_path]}, {split})')
            seen[full_path] = split
            row['_resolved_audio'] = str(full_path)
            if not row[columns['text']].strip():
                raise ValueError(f'Empty transcript: {full_path}')
    if recording_col:
        for split, subset in result.items():
            for row in subset:
                rid = row[recording_col].strip()
                if not rid:
                    raise ValueError('Empty recording ID; cannot verify recording leakage')
    return result


class WhisperSpeechDataset(Dataset):
    def __init__(self, rows, processor, text_col, max_audio_seconds=30.0, max_label_length=448):
        self.rows = rows
        self.processor = processor
        self.text_col = text_col
        self.max_audio_seconds = max_audio_seconds
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = row['_resolved_audio']
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        # 기존 범용 모델과 동일하게 모노 음성으로 변환
        audio = audio.mean(axis=1)
        # 기존 범용 모델과 동일한 리샘플링 방식
        if sr != 16000:
            audio = librosa.resample(audio,orig_sr=sr,target_sr=16000)
            sr = 16000
        if len(audio) > int(self.max_audio_seconds * sr):
            raise ValueError(f'Audio longer than {self.max_audio_seconds}s: {path}')
        extracted = self.processor.feature_extractor(audio,sampling_rate=sr,return_tensors="pt",return_attention_mask=True)
        features = extracted.input_features[0]
        attention_mask = extracted.attention_mask[0]
        ids = self.processor.tokenizer(row[self.text_col].strip(), add_special_tokens=True).input_ids
        if len(ids) > self.max_label_length:
            raise ValueError(f'Transcript too long ({len(ids)} tokens): {path}')
        return {
            "input_features": features,
            "attention_mask": attention_mask,
            "labels": ids
        }

class WhisperCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        inputs = torch.stack([
            x["input_features"] for x in features
        ])

        attention_masks = torch.stack([
            x["attention_mask"] for x in features
        ])

        padded = self.processor.tokenizer.pad(
            [{"input_ids": x["labels"]} for x in features],
            return_tensors="pt"
        )

        labels = padded["input_ids"].masked_fill(
            padded["attention_mask"] == 0,
            -100
        )

        # Whisper decoder의 시작 토큰 중복 방지
        start = self.processor.tokenizer.bos_token_id
        if start is not None and (labels[:, 0] == start).all():
            labels = labels[:, 1:]

        return {
            "input_features": inputs,
            "attention_mask": attention_masks,
            "labels": labels
        }