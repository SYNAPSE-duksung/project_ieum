import copy
import csv
import json
import math
import os
import random
import shutil
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

ROOT = Path(__import__("os").environ.get("IEUM_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/training_config.json"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = json.load(f)

for key in ("manifest_csv", "audio_root", "output_root"):
    if key in CFG:
        path = Path(CFG[key])
        CFG[key] = str(path if path.is_absolute() else ROOT / path)
OUT = Path(CFG["output_root"])
CHECKPOINT_ROOT = OUT / "checkpoints"
MODEL_ID = CFG["model_id"]
SEED = int(CFG["seed"])
BATCH_SIZE = int(CFG["micro_batch_size"])
ACCUM = int(CFG["gradient_accumulation_steps"])
EPOCHS = int(CFG["max_epochs"])
STEPS_PER_EPOCH = int(CFG["steps_per_epoch"])
TOTAL_STEPS = int(CFG["max_optimizer_steps"])
SAVE_EVERY = int(CFG["checkpoint_every_steps"])
WARMUP = max(1, round(TOTAL_STEPS * CFG["warmup_ratio"]))
PATIENCE = int(CFG["early_stopping"]["patience"])
MIN_DELTA = float(CFG["early_stopping"]["min_delta_percentage_points"])
DEVICE = "cuda"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    chars = [" " if unicodedata.category(c).startswith("P") else c for c in text]
    return " ".join("".join(chars).split())

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        curr = [i]
        for j, y in enumerate(b, 1):
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = curr
    return prev[-1]

def metric_counts(reference, prediction):
    ref = normalize(reference)
    hyp = normalize(prediction)
    rc, hc = ref.replace(" ", ""), hyp.replace(" ", "")
    rw, hw = ref.split(), hyp.split()
    return edit_distance(rc, hc), len(rc), edit_distance(rw, hw), len(rw)

class SpeechDataset(Dataset):
    def __init__(self, manifest, split, processor):
        self.processor = processor
        self.rows = []
        with open(manifest, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["split"].strip() == split:
                    self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        member = row["audio_member"].replace("\\", "/")
        path = Path(CFG["audio_root"]).joinpath(row["batch"], *member.split("/"))
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        extracted = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        labels = self.processor.tokenizer(row["text"], add_special_tokens=True).input_ids
        return {"features": extracted.input_features[0], "attention_mask": extracted.attention_mask[0], "labels": labels, "text": row["text"], "group": row["disability_group"]}

def collate_train(items):
    inputs = torch.stack([x["features"] for x in items])
    padded = processor.tokenizer.pad([{"input_ids": x["labels"]} for x in items], return_tensors="pt")
    labels = padded["input_ids"].masked_fill(padded["attention_mask"].ne(1), -100)
    if (labels[:, 0] == processor.tokenizer.bos_token_id).all().item():
        labels = labels[:, 1:]
    return {"input_features": inputs, "labels": labels}

def collate_eval(items):
    return {"input_features": torch.stack([x["features"] for x in items]), "attention_mask": torch.stack([x["attention_mask"] for x in items]), "texts": [x["text"] for x in items], "groups": [x["group"] for x in items]}

def get_epoch_indices(epoch, n):
    return np.random.default_rng(SEED + epoch).permutation(n).tolist()

def save_json_atomic(path, data):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)

def save_checkpoint(step, next_batch, epoch):
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    target = CHECKPOINT_ROOT / f"step_{step:06d}"
    temporary = CHECKPOINT_ROOT / f".step_{step:06d}_writing"
    if target.exists() or temporary.exists():
        raise FileExistsError(f"Checkpoint path already exists: {target}")
    temporary.mkdir()
    model.save_pretrained(temporary / "model", safe_serialization=True)
    state = {"step": step, "epoch": epoch, "next_batch": next_batch, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
    torch.save(state, temporary / "training_state.pt")
    processor.save_pretrained(temporary / "processor")
    (temporary / "complete.marker").write_text("complete", encoding="utf-8")
    temporary.rename(target)
    save_json_atomic(OUT / "latest_checkpoint.json", {"path": str(target), "step": step, "epoch": epoch, "next_batch": next_batch})
    print(f"Checkpoint saved: {target}", flush=True)
    # Keep last two periodic checkpoints and every epoch-end checkpoint.
    completed = sorted(p for p in CHECKPOINT_ROOT.glob("step_*") if p.is_dir() and (p / "complete.marker").exists())
    protected_steps = {STEPS_PER_EPOCH * e for e in range(1, EPOCHS + 1)}
    for old in completed[:-2]:
        old_step = int(old.name.split("_")[1])
        if old_step not in protected_steps:
            shutil.rmtree(old)
    return target

@torch.inference_mode()
def evaluate():
    model.eval()
    overall = [0, 0, 0, 0]
    by_group = defaultdict(lambda: [0, 0, 0, 0])
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.language = "korean"
    generation_config.task = "transcribe"
    generation_config.max_length = 160
    generation_config.do_sample = False
    generation_config.num_beams = 1
    for batch_idx, batch in enumerate(val_loader, 1):
        inputs = batch["input_features"].to(DEVICE)
        masks = batch["attention_mask"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted_ids = model.generate(input_features=inputs, attention_mask=masks, generation_config=generation_config)
        predictions = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        for ref, hyp, group in zip(batch["texts"], predictions, batch["groups"]):
            counts = metric_counts(ref, hyp)
            for i in range(4):
                overall[i] += counts[i]
                by_group[group][i] += counts[i]
        if batch_idx % 500 == 0:
            print(f"Validation {batch_idx}/{len(val_loader)}", flush=True)
    def percentages(counts):
        ce, ct, we, wt = counts
        return {"cer": 100 * ce / ct if ct else None, "wer": 100 * we / wt if wt else None}
    group_metrics = {group: percentages(counts) for group, counts in sorted(by_group.items())}
    macro_cer = [m["cer"] for m in group_metrics.values() if m["cer"] is not None]
    result = {**percentages(overall), "macro_group_cer": sum(macro_cer) / len(macro_cer) if macro_cer else None, "group_metrics": group_metrics}
    model.train()
    return result

def validation_path(epoch):
    return OUT / f"epoch_{epoch}_validation.json"

def read_history():
    history = []
    for epoch in range(1, EPOCHS + 1):
        path = validation_path(epoch)
        if not path.exists():
            break
        with path.open("r", encoding="utf-8") as f:
            history.append(json.load(f))
    return history

def earlystop_status(history):
    # Keep separate best (strict minimum) and patience anchor (min_delta).
    best_cer = float("inf")
    best_epoch = None
    anchor_cer = float("inf")
    bad_epochs = 0
    for entry in history:
        cer = float(entry["validation"]["cer"])
        if cer < best_cer:
            best_cer, best_epoch = cer, int(entry["epoch"])
        if cer < anchor_cer - MIN_DELTA or math.isinf(anchor_cer):
            anchor_cer = cer
            bad_epochs = 0
        else:
            bad_epochs += 1
    return best_cer, best_epoch, bad_epochs

def finish(history, reason, step, elapsed_minutes):
    best_cer, best_epoch, bad_epochs = earlystop_status(history)
    best_step = best_epoch * STEPS_PER_EPOCH
    best_checkpoint = CHECKPOINT_ROOT / f"step_{best_step:06d}"
    if not (best_checkpoint / "complete.marker").exists():
        raise RuntimeError("Best checkpoint is missing")
    result = {"experiment": CFG["experiment"], "stop_reason": reason, "completed_epochs": len(history), "completed_steps": step, "best_epoch": best_epoch, "best_validation_cer": best_cer, "best_checkpoint": str(best_checkpoint), "validation_history": history, "elapsed_minutes_this_run": elapsed_minutes, "early_stopping_bad_epochs": bad_epochs}
    save_json_atomic(OUT / "best_checkpoint.json", {"path": str(best_checkpoint), "epoch": best_epoch, "cer": best_cer})
    save_json_atomic(OUT / "training_complete.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print("RESULT: PASS", flush=True)
    print("========== COMPLETE ==========", flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16 is unavailable")
if CFG["scheduler"] != "linear" or CFG["initialization"] != "pretrained_base":
    raise RuntimeError("Unexpected scheduler or initialization")
if EPOCHS != 5 or ACCUM != 4 or BATCH_SIZE != 4 or PATIENCE != 2:
    raise RuntimeError("Unexpected experiment settings")
if STEPS_PER_EPOCH * EPOCHS != TOTAL_STEPS:
    raise RuntimeError("Step count mismatch")
set_seed(SEED)
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="korean", task="transcribe")
train_dataset = SpeechDataset(CFG["manifest_csv"], "train", processor)
val_dataset = SpeechDataset(CFG["manifest_csv"], "validation", processor)
if len(train_dataset) != CFG["train_samples"] or len(val_dataset) != CFG["validation_samples"]:
    raise RuntimeError("Dataset count mismatch")
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_eval)
micro_batches = math.ceil(len(train_dataset) / BATCH_SIZE)
if math.ceil(micro_batches / ACCUM) != STEPS_PER_EPOCH:
    raise RuntimeError("Optimizer step count mismatch")
OUT.mkdir(parents=True, exist_ok=True)
latest_path = OUT / "latest_checkpoint.json"
complete_path = OUT / "training_complete.json"
if complete_path.exists():
    raise RuntimeError("Training already completed")
resume_dir = None
if latest_path.exists():
    with latest_path.open("r", encoding="utf-8") as f:
        latest = json.load(f)
    resume_dir = Path(latest["path"])
    if not (resume_dir / "complete.marker").exists():
        raise RuntimeError("Incomplete checkpoint")
    model_source = resume_dir / "model"
    print(f"Resuming from {resume_dir}", flush=True)
else:
    model_source = MODEL_ID
model = WhisperForConditionalGeneration.from_pretrained(model_source, attn_implementation="sdpa").to(DEVICE)
model.config.use_cache = False
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"], fused=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP, num_training_steps=TOTAL_STEPS)
global_step = 0
resume_epoch = 1
start_batch = 0
if resume_dir is not None:
    # Load only checkpoint states created locally by this script.
    state = torch.load(resume_dir / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    global_step = int(state["step"])
    resume_epoch = int(state["epoch"])
    start_batch = int(state["next_batch"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    if not (1 <= resume_epoch <= EPOCHS) or not (0 <= start_batch <= micro_batches):
        raise RuntimeError("Invalid resume position")
    expected_step = (resume_epoch - 1) * STEPS_PER_EPOCH + math.ceil(start_batch / ACCUM)
    if global_step != expected_step:
        raise RuntimeError("Resume step/batch mismatch")
start_time = time.perf_counter()
print("========== EARLY STOPPING TRAINING ==========", flush=True)
print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)}", flush=True)
print(f"Epochs: {EPOCHS} | Patience: {PATIENCE} | Min delta: {MIN_DELTA}", flush=True)
print(f"Steps per epoch: {STEPS_PER_EPOCH} | Maximum steps: {TOTAL_STEPS}", flush=True)
print(f"Warmup steps: {WARMUP} | Starting step: {global_step}", flush=True)
for epoch in range(resume_epoch, EPOCHS + 1):
    history = read_history()
    if len(history) > epoch:
        raise RuntimeError("Validation history is ahead of checkpoint")
    if len(history) == epoch:
        if start_batch != micro_batches or global_step != epoch * STEPS_PER_EPOCH:
            raise RuntimeError("Validated epoch does not match checkpoint")
        _, _, bad = earlystop_status(history)
        if bad >= PATIENCE or epoch == EPOCHS:
            finish(history, "early_stopping" if bad >= PATIENCE else "max_epochs", global_step, (time.perf_counter() - start_time) / 60)
            break
        start_batch = 0
        continue
    if len(history) == epoch - 1 and len(history) > 0:
        _, _, bad = earlystop_status(history)
        if bad >= PATIENCE:
            finish(history, "early_stopping", global_step, (time.perf_counter() - start_time) / 60)
            break
    batch_offset = start_batch if epoch == resume_epoch else 0
    if batch_offset < micro_batches:
        indices = get_epoch_indices(epoch, len(train_dataset))
        remaining_indices = indices[batch_offset * BATCH_SIZE:]
        train_loader = DataLoader(Subset(train_dataset, remaining_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_train)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        print(f"========== EPOCH {epoch}/{EPOCHS} TRAIN ==========" , flush=True)
        for local_batch_idx, batch in enumerate(train_loader):
            batch_idx = batch_offset + local_batch_idx + 1
            inputs = batch["input_features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_features=inputs, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {batch_idx}")
            remaining_in_epoch = micro_batches - ((batch_idx - 1) // ACCUM) * ACCUM
            divisor = min(ACCUM, remaining_in_epoch)
            (loss / divisor).backward()
            should_step = batch_idx % ACCUM == 0 or batch_idx == micro_batches
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite gradient at epoch {epoch}, batch {batch_idx}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 100 == 0:
                    elapsed = (time.perf_counter() - start_time) / 60
                    print(f"Epoch {epoch} | Step {global_step}/{TOTAL_STEPS} | Loss {loss.item():.4f} | LR {scheduler.get_last_lr()[0]:.8f} | Elapsed {elapsed:.1f} min", flush=True)
                if global_step % SAVE_EVERY == 0:
                    save_checkpoint(global_step, batch_idx, epoch)
    if global_step != epoch * STEPS_PER_EPOCH:
        raise RuntimeError(f"Epoch {epoch} step mismatch: {global_step}")
    end_checkpoint = CHECKPOINT_ROOT / f"step_{global_step:06d}"
    if not (end_checkpoint / "complete.marker").exists():
        save_checkpoint(global_step, micro_batches, epoch)
    if not validation_path(epoch).exists():
        print(f"========== EPOCH {epoch} VALIDATION ==========" , flush=True)
        metrics = evaluate()
        save_json_atomic(validation_path(epoch), {"epoch": epoch, "step": global_step, "validation": metrics})
        print(f"Epoch {epoch} CER: {metrics['cer']:.6f}% | WER: {metrics['wer']:.6f}% | Macro CER: {metrics['macro_group_cer']:.6f}%", flush=True)
    history = read_history()
    best_cer, best_epoch, bad_epochs = earlystop_status(history)
    print(f"Best epoch: {best_epoch} | Best CER: {best_cer:.6f}% | Patience: {bad_epochs}/{PATIENCE}", flush=True)
    if bad_epochs >= PATIENCE or epoch == EPOCHS:
        reason = "early_stopping" if bad_epochs >= PATIENCE else "max_epochs"
        finish(history, reason, global_step, (time.perf_counter() - start_time) / 60)
        break
    start_batch = 0
