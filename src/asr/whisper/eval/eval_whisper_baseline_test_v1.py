import csv
import json
import os
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__import__("os").environ.get("IEUM_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/evaluation_config_original.json"

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = json.load(f)

for key in ("manifest_csv", "audio_root", "output_root"):
    if key in CFG:
        path = Path(CFG[key])
        CFG[key] = str(path if path.is_absolute() else ROOT / path)

OUT = ROOT / "results/whisper_small_baseline_test_v1"
RESULT_PATH = OUT / "test_metrics.json"

if RESULT_PATH.exists():
    raise FileExistsError(
        f"Result already exists: {RESULT_PATH}"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16 is unavailable")

DEVICE = "cuda"
BATCH_SIZE = 4

def normalize(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    chars = [
        " " if unicodedata.category(c).startswith("P") else c
        for c in text
    ]
    return " ".join("".join(chars).split())

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(
                curr[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (x != y),
            ))
        prev = curr
    return prev[-1]

def metric_counts(reference, prediction):
    ref = normalize(reference)
    hyp = normalize(prediction)

    ref_chars = ref.replace(" ", "")
    hyp_chars = hyp.replace(" ", "")

    cer_errors = edit_distance(ref_chars, hyp_chars)
    cer_total = len(ref_chars)

    ref_words = ref.split()
    hyp_words = hyp.split()

    wer_errors = edit_distance(ref_words, hyp_words)
    wer_total = len(ref_words)

    return cer_errors, cer_total, wer_errors, wer_total

class ValidationDataset(Dataset):
    def __init__(self, manifest, processor):
        self.processor = processor
        self.rows = []

        with open(
            manifest,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            for row in csv.DictReader(f):
                if row["split"].strip() == "test":
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        member = row["audio_member"].replace("\\", "/")
        path = Path(CFG["audio_root"]).joinpath(
            row["batch"],
            *member.split("/"),
        )

        audio, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )
        audio = audio.mean(axis=1)

        if sr != 16000:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=16000,
            )

        extracted = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return {
            "features": extracted.input_features[0],
            "attention_mask": extracted.attention_mask[0],
            "text": row["text"],
            "group": row["disability_group"],
        }

def collate(items):
    return {
        "input_features": torch.stack(
            [x["features"] for x in items]
        ),
        "attention_mask": torch.stack(
            [x["attention_mask"] for x in items]
        ),
        "texts": [x["text"] for x in items],
        "groups": [x["group"] for x in items],
    }

def percentages(counts):
    ce, ct, we, wt = counts

    return {
        "cer": 100 * ce / ct if ct else None,
        "wer": 100 * we / wt if wt else None,
    }

print("========== BASELINE TEST EVALUATION ==========", flush=True)

start_time = time.perf_counter()

processor = WhisperProcessor.from_pretrained(
    CFG["model_id"],
    language="korean",
    task="transcribe",
)

dataset = ValidationDataset(
    CFG["manifest_csv"],
    processor,
)

if len(dataset) != 46834:
    raise RuntimeError(
        f"Validation count mismatch: {len(dataset)}"
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate,
)

model = WhisperForConditionalGeneration.from_pretrained(
    CFG["model_id"],
    attn_implementation="sdpa",
).to(DEVICE)

model.eval()

generation_config = model.generation_config
generation_config.language = "korean"
generation_config.task = "transcribe"
generation_config.max_length = 160
generation_config.do_sample = False
generation_config.num_beams = 1

overall = [0, 0, 0, 0]
by_group = defaultdict(lambda: [0, 0, 0, 0])

print("Model:", CFG["model_id"], flush=True)
print("Test samples:", len(dataset), flush=True)
print("Total batches:", len(loader), flush=True)

with torch.inference_mode():
    for batch_idx, batch in enumerate(loader, 1):
        inputs = batch["input_features"].to(DEVICE)
        masks = batch["attention_mask"].to(DEVICE)

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            predicted_ids = model.generate(
                input_features=inputs,
                attention_mask=masks,
                generation_config=generation_config,
            )

        predictions = processor.tokenizer.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )

        for ref, hyp, group in zip(
            batch["texts"],
            predictions,
            batch["groups"],
        ):
            counts = metric_counts(ref, hyp)

            for i in range(4):
                overall[i] += counts[i]
                by_group[group][i] += counts[i]

        if batch_idx % 500 == 0:
            elapsed = (
                time.perf_counter() - start_time
            ) / 60

            print(
                f"Test {batch_idx}/{len(loader)} "
                f"| Elapsed {elapsed:.1f} min",
                flush=True,
            )

group_metrics = {
    group: percentages(counts)
    for group, counts in sorted(by_group.items())
}

macro_cer_values = [
    m["cer"]
    for m in group_metrics.values()
    if m["cer"] is not None
]

result = {
    "experiment": "whisper_small_baseline_test_v1",
    "model": CFG["model_id"],
    "training": False,
    "split": "test",
    "samples": len(dataset),
    "test": {
        **percentages(overall),
        "macro_group_cer": (
            sum(macro_cer_values)
            / len(macro_cer_values)
        ),
        "group_metrics": group_metrics,
    },
    "elapsed_minutes": (
        time.perf_counter() - start_time
    ) / 60,
}

OUT.mkdir(parents=True, exist_ok=True)

with RESULT_PATH.open(
    "x",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2,
    )

print()
print(json.dumps(
    result,
    ensure_ascii=False,
    indent=2,
))
print("Saved:", RESULT_PATH)
print("RESULT: PASS")
print("========== COMPLETE ==========")
