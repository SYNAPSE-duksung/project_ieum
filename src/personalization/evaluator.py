"""Shared Whisper CER/WER evaluation; no test-set model selection."""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    chars = [
        " " if unicodedata.category(c).startswith("P") else c
        for c in text
    ]
    return " ".join("".join(chars).split())


def decode_predictions(predictions, label_ids, tokenizer):
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    if predictions.ndim == 3:
        raise ValueError(
            'Expected generated token IDs, got logits. '
            'Set predict_with_generate=True.'
        )

    labels = np.where(
        label_ids == -100,
        tokenizer.pad_token_id,
        label_ids
    )

    hypotheses = [
        normalize_text(x)
        for x in tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )
    ]

    references = [
        normalize_text(x)
        for x in tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )
    ]

    return references, hypotheses

def edit_distance(a, b):
    prev = list(range(len(b) + 1))

    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(
                min(
                    curr[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (x != y)
                )
            )
        prev = curr

    return prev[-1]


def metric_counts(reference, prediction):
    ref = normalize_text(reference)
    hyp = normalize_text(prediction)

    rc = ref.replace(" ", "")
    hc = hyp.replace(" ", "")

    rw = ref.split()
    hw = hyp.split()

    return (
        edit_distance(rc, hc),
        len(rc),
        edit_distance(rw, hw),
        len(rw)
    )

def compute_asr_metrics(predictions, label_ids, tokenizer):
    references, hypotheses = decode_predictions(
        predictions, label_ids, tokenizer
    )

    total = [0, 0, 0, 0]

    for ref, hyp in zip(references, hypotheses):
        counts = metric_counts(ref, hyp)

        for i in range(4):
            total[i] += counts[i]

    ce, ct, we, wt = total

    return {
        "cer": ce / ct if ct else 0.0,
        "wer": we / wt if wt else 0.0
    }


def save_predictions(output_dir, rows, references, hypotheses, metrics):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not (len(rows) == len(references) == len(hypotheses)):
        raise ValueError(
            'Prediction and manifest lengths do not match'
        )

    with (out / 'test_predictions.jsonl').open(
        'w', encoding='utf-8'
    ) as f:
        for row, ref, hyp in zip(rows, references, hypotheses):
            f.write(
                json.dumps(
                    {
                        'audio_path': row['_resolved_audio'],
                        'reference': ref,
                        'hypothesis': hyp
                    },
                    ensure_ascii=False
                ) + '\n'
            )

    (out / 'test_metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
