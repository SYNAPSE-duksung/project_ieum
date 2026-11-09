"""Fixed E2 character vocabulary and read-only preflight; no verbalization/NFC."""
from collections import Counter
import json
import unicodedata

import soundfile as sf
import torch

from .dataset import WhisperSpeechDataset, WhisperCollator

BLANK_ID = 0
HANGUL_COUNT = 11172
VOCAB_SIZE = 11183
VOCAB_VERSION = 'hangul_digits_v1'
ID_TO_TOKEN = ('<blank>',) + tuple(chr(c) for c in range(0xAC00, 0xD7A4)) + tuple('0123456789')
TOKEN_TO_ID = {token: i for i, token in enumerate(ID_TO_TOKEN)}


def unsupported_characters(text):
    return Counter(c for c in str(text)
                   if c not in TOKEN_TO_ID and not c.isspace()
                   and not unicodedata.category(c).startswith('P'))


def encode_ctc(text):
    """Only whitespace/punctuation may be discarded; everything else must map."""
    bad = unsupported_characters(text)
    if bad:
        raise ValueError(f'Unsupported CTC characters: {dict(bad)!r}')
    return [TOKEN_TO_ID[c] for c in str(text) if c in TOKEN_TO_ID]


def decode_ctc(ids):
    """Decode target IDs, not greedy-collapse logits; repeated digits survive."""
    values = [int(i) for i in ids]
    if any(i <= 0 or i >= VOCAB_SIZE for i in values):
        raise ValueError('CTC target IDs must be 1..11182 (blank is not a target)')
    return ''.join(ID_TO_TOKEN[i] for i in values)


def minimum_ctc_frames(ids):
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def encoder_output_lengths(mel_lengths, encoder):
    """Actual Conv1d formula, usable on an integer or a tensor."""
    lengths = mel_lengths
    for conv in (encoder.conv1, encoder.conv2):
        lengths = ((lengths + 2 * conv.padding[0]
                    - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1)
                   // conv.stride[0] + 1)
    return lengths


def wav_lengths(frames, sample_rate, extractor, encoder):
    # E1/librosa resample(fix=True) rounds up the resampled sample count.
    samples = (frames * extractor.sampling_rate + sample_rate - 1) // sample_rate
    kept = min(samples, extractor.n_samples)
    # E1 uses default padding=max_length/truncation=True. The extractor slices
    # sample attention masks at ::hop_length, producing ceil(kept/hop), capped
    # by the padded STFT frame count (STFT's last frame is dropped).
    mel = min((kept + extractor.hop_length - 1) // extractor.hop_length,
              extractor.n_samples // extractor.hop_length)
    return samples, mel, encoder_output_lengths(mel, encoder) if mel else 0


def validate_extractor(extractor, encoder):
    if (extractor.sampling_rate != 16000 or extractor.padding_side != 'right'
            or extractor.n_samples % extractor.hop_length):
        raise ValueError('Unsupported extractor sample rate, padding side or frame grid')
    padded = encoder_output_lengths(extractor.nb_max_frames, encoder)
    if padded != encoder.config.max_source_positions:
        raise ValueError('Extractor and encoder padded lengths disagree')


def preflight_splits(splits, processor, encoder, text_col, settings):
    """Report every selected utterance problem, then fail; never exclude rows."""
    extractor = processor.feature_extractor
    validate_extractor(extractor, encoder)
    issues, bad_counts, ignored_counts = [], Counter(), Counter()
    summary, digit_train = {}, []
    max_samples = int(float(settings.get('max_audio_seconds', 30)) * extractor.sampling_rate)
    for split, rows in splits.items():
        frame_counts, target_lengths = [], []
        for row in rows:
            text, audio = row[text_col], row['_resolved_audio']
            bad = unsupported_characters(text)
            bad_counts.update(bad)
            ignored_counts.update(c for c in text if c.isspace() or unicodedata.category(c).startswith('P'))
            reasons = []
            if bad:
                reasons.append(f'unsupported characters: {dict(bad)!r}')
            ids = None if bad else encode_ctc(text)
            if ids == []:
                reasons.append('empty CTC target')
            if split == 'train' and any(c in '0123456789' for c in text):
                digit_train.append(dict(audio=audio, text=text, target_ids=ids,
                                        target=decode_ctc(ids) if ids else None))
            tokens = processor.tokenizer(text.strip(), add_special_tokens=True).input_ids
            if len(tokens) > int(settings['generation_max_length']):
                reasons.append(f'decoder labels too long: {len(tokens)}')
            try:
                info = sf.info(audio)
                if info.frames <= 0 or info.samplerate <= 0:
                    raise ValueError('empty/invalid WAV')
                samples, mel, valid = wav_lengths(info.frames, info.samplerate, extractor, encoder)
                frame_counts.append(valid)
                if samples > max_samples:
                    reasons.append(f'E1 maximum audio duration exceeded: {samples} samples')
                if samples > extractor.n_samples:
                    reasons.append('extractor would truncate audio but retain full transcript')
                if ids is not None:
                    target_lengths.append(len(ids))
                    required = minimum_ctc_frames(ids)
                    if required > valid:
                        reasons.append(f'CTC unalignable: needs {required}, available {valid} frames')
            except (OSError, RuntimeError, ValueError) as exc:
                reasons.append(f'WAV inspection: {exc}')
            if reasons:
                issues.append(dict(split=split, audio=audio, text=text, reasons=reasons))
        summary[split] = dict(utterances=len(rows),
                             valid_frame_range=[min(frame_counts), max(frame_counts)] if frame_counts else [],
                             target_length_range=[min(target_lengths), max(target_lengths)] if target_lengths else [])
    report = dict(vocabulary_version=VOCAB_VERSION, vocabulary_size=VOCAB_SIZE,
                  unsupported_policy='error; only whitespace and Unicode punctuation are removed',
                  unsupported=[dict(character=c, codepoint=f'U+{ord(c):04X}', count=n)
                               for c, n in sorted(bad_counts.items())],
                  removed_whitespace_punctuation=dict(ignored_counts), splits=summary,
                  digit_train=digit_train, issues=issues)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise ValueError(f'Preflight failed: {len(issues)} utterances; no rows excluded')
    return report


class E2SpeechDataset(WhisperSpeechDataset):
    """E1 audio/decoder preprocessing unchanged, plus unpadded CTC targets."""
    def __init__(self, *args, encoder, **kwargs):
        super().__init__(*args, **kwargs)
        # Keep only convolution geometry, not a reference to the full model.
        self.convs = [(c.kernel_size[0], c.stride[0], c.padding[0], c.dilation[0])
                      for c in (encoder.conv1, encoder.conv2)]

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        ids = encode_ctc(self.rows[idx][self.text_col])
        length = int(item['attention_mask'].sum())
        for kernel, stride, padding, dilation in self.convs:
            length = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
        if not ids or minimum_ctc_frames(ids) > length:
            raise ValueError(f'Invalid CTC target/alignment: {self.rows[idx]["_resolved_audio"]}')
        item.update(ctc_targets=ids, ctc_target_lengths=len(ids), ctc_input_lengths=length)
        return item


class E2Collator(WhisperCollator):
    def __call__(self, features):
        batch = super().__call__(features)
        # Concatenated targets avoid introducing padding tokens into CTCLoss.
        batch['ctc_targets'] = torch.tensor([v for f in features for v in f['ctc_targets']], dtype=torch.long)
        for key in ('ctc_target_lengths', 'ctc_input_lengths'):
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        return batch
