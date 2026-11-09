"""Persistent epoch metrics and numerical/resource audit for explicit E2 runs."""
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import torch
from transformers import TrainerCallback


class E2RunRecorder(TrainerCallback):
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.started = None
        self.train_components = defaultdict(float)
        self.samples = 0
        self.report = dict(status='prepared', epochs=[], nonfinite_events=[],
                           nonfinite_observed=False, training_seconds=None,
                           gpu_peak_allocated_bytes=0, gpu_peak_reserved_bytes=0)

    def persist(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.run_dir / 'training_audit.json.tmp'
        temporary.write_text(json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(self.run_dir / 'training_audit.json')

    def nonfinite(self, stage, state, message):
        self.report['nonfinite_observed'] = True
        self.report['nonfinite_events'].append(dict(stage=stage, step=state.global_step,
                                                    epoch=state.epoch, message=str(message)))
        self.persist()

    def record_train(self, outputs, count, state):
        for name, value in (('asr_loss', outputs.asr_loss), ('ctc_loss', outputs.ctc_loss),
                            ('weighted_ctc_loss', outputs.weighted_ctc_loss), ('total_loss', outputs.loss)):
            number = float(value.detach())
            if not math.isfinite(number):
                self.nonfinite(name, state, number)
                raise FloatingPointError(f'Non-finite {name}')
            self.train_components[name] += number * count
        self.samples += count

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()
        self.report.update(status='training', precision='bf16' if args.bf16 else 'fp16' if args.fp16 else 'fp32',
                           device=str(args.device), gradient_check='loss components and gradients each microbatch')
        if args.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(args.device)
            self.report['gpu_name'] = torch.cuda.get_device_name(args.device)
        self.persist()

    def on_epoch_end(self, args, state, control, **kwargs):
        entry = dict(epoch=state.epoch, global_step=state.global_step, train_samples=self.samples,
                     train={k: v / self.samples for k, v in self.train_components.items()} if self.samples else {})
        self.report['epochs'].append(entry)
        self.train_components.clear()
        self.samples = 0
        self.persist()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if 'eval_cer' not in metrics:
            return
        for key in ('eval_cer', 'eval_wer', 'eval_loss'):
            if key in metrics and not math.isfinite(float(metrics[key])):
                self.nonfinite(key, state, str(metrics[key]))
                raise FloatingPointError(f'Non-finite {key}')
        epoch = next((r for r in self.report['epochs'] if r['global_step'] == state.global_step), None)
        if epoch is None:
            return
        if 'validation' in epoch:
            return  # Post-training reevaluation of the loaded best model is not a new epoch.
        epoch['validation'] = {k: float(v) for k, v in metrics.items() if k in ('eval_cer', 'eval_wer', 'eval_loss')}
        best = min((r for r in self.report['epochs'] if 'validation' in r), key=lambda r: r['validation']['eval_cer'])
        self.report.update(best_epoch=best['epoch'], best_validation_cer=best['validation']['eval_cer'])
        self.persist()

    def on_save(self, args, state, control, **kwargs):
        self.report['best_checkpoint'] = state.best_model_checkpoint
        self.persist()

    def finish(self, args, status, error=None):
        self.report['status'] = status
        if self.started is not None and (self.report['training_seconds'] is None or status == 'failed'):
            self.report['training_seconds'] = time.perf_counter() - self.started
        if args.device.type == 'cuda':
            self.report.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                               gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        if error is not None:
            self.report['error'] = str(error)
        self.persist()

    def on_train_end(self, args, state, control, **kwargs):
        self.finish(args, 'trained')
