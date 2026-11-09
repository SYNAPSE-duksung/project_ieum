"""Whisper E3 ranking with explicit E1-compatible or mean accumulation."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.modeling_outputs import BaseModelOutput

from .e3_data import E3Collator
from .e3_loss import token_mean_log_probability, correction_margin
from .evaluator import compute_asr_metrics


class E3Audit(TrainerCallback):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.report = dict(epochs=[], nonfinite_events=[], status='prepared')
        self.sums = defaultdict(float)
        self.counts = Counter()
        self.started = None

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self.directory / 'training_audit.json.tmp'
        temp.write_text(json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temp.replace(self.directory / 'training_audit.json')

    def add(self, stats, batch_size):
        for name in ('asr_loss', 'correction_loss', 'total_loss'):
            self.sums[name] += stats[name] * batch_size
        self.counts.update(samples=batch_size, eligible=stats['eligible'], used=stats['used'],
                           active=stats['active'], excluded=batch_size-stats['eligible'])
        if stats['used']:
            self.sums['reference_score_sum'] += stats['reference_score']*stats['used']
            self.sums['hypothesis_score_sum'] += stats['hypothesis_score']*stats['used']

    def failure(self, message):
        self.report['status'] = 'failed'
        self.report['nonfinite_events'].append(str(message))
        self.save()

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()
        self.report['status'] = 'training'
        if args.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(args.device)
        self.save()

    def on_epoch_end(self, args, state, control, **kwargs):
        n = self.counts['samples']
        used=self.counts['used']
        self.report['epochs'].append(dict(epoch=state.epoch, step=state.global_step,
            train={k: self.sums[k]/n for k in ('asr_loss','correction_loss','total_loss')},
            reference_score_mean=self.sums['reference_score_sum']/used if used else None,
            hypothesis_score_mean=self.sums['hypothesis_score_sum']/used if used else None,
            margin_active_fraction=self.counts['active']/used if used else 0.,candidate_usage=dict(self.counts)))
        self.sums.clear(); self.counts.clear(); self.save()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if 'eval_cer' not in metrics:
            return
        item = next((r for r in self.report['epochs'] if r['step'] == state.global_step), None)
        if item is None or 'validation' in item:
            return
        item['validation'] = {k: v for k, v in metrics.items() if k in ('eval_cer', 'eval_wer', 'eval_loss')}
        best = min((r for r in self.report['epochs'] if 'validation' in r), key=lambda r: r['validation']['eval_cer'])
        self.report.update(best_epoch=best['epoch'], best_validation_cer=best['validation']['eval_cer'])
        self.save()

    def on_save(self, args, state, control, **kwargs):
        self.report['best_checkpoint'] = state.best_model_checkpoint
        self.save()

    def on_train_end(self, args, state, control, **kwargs):
        self.report.update(status='trained', training_seconds=time.perf_counter()-self.started)
        if args.device.type == 'cuda':
            self.report.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                               gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        self.save()


class E3Trainer(Seq2SeqTrainer):
    def __init__(self, *args, correction_weight=.1, margin=.1, accumulation_mode='legacy_e1', audit=None, **kwargs):
        super().__init__(*args, **kwargs)
        if accumulation_mode not in ('legacy_e1', 'mean'):
            raise ValueError('Unknown accumulation mode')
        self.correction_weight, self.margin = correction_weight, margin
        self.accumulation_mode, self.audit = accumulation_mode, audit
        # Explicit compatibility with the historical E1 path. In mean mode,
        # Trainer divides microbatch-mean loss by actual accumulation count.
        self.model_accepts_loss_kwargs = accumulation_mode == 'legacy_e1'
        self.last_components = {}
        self._log_sums = defaultdict(float)
        self._log_batches = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        candidate_labels = inputs.pop('correction_labels', None)
        eligible = inputs.pop('correction_eligible', None)
        output = model(**inputs, return_dict=True)
        asr = output.loss
        correction = asr.new_zeros(())
        indices = eligible.nonzero(as_tuple=True)[0] if eligible is not None else None
        used, active, ref_mean, hyp_mean = 0, 0, None, None
        if model.training and self.correction_weight > 0 and indices is not None and indices.numel():
            tokenizer = self.processing_class.tokenizer
            refs = token_mean_log_probability(output.logits.index_select(0, indices),
                inputs['labels'].index_select(0, indices), tokenizer.all_special_ids, tokenizer.eos_token_id)
            # Encoder states retain their graph; no detach or no_grad here.
            hypotheses = model(encoder_outputs=BaseModelOutput(
                last_hidden_state=output.encoder_last_hidden_state.index_select(0, indices)),
                attention_mask=inputs['attention_mask'].index_select(0, indices),
                labels=candidate_labels.index_select(0, indices), use_cache=False, return_dict=True)
            hyps = token_mean_log_probability(hypotheses.logits,
                candidate_labels.index_select(0, indices), tokenizer.all_special_ids, tokenizer.eos_token_id)
            correction, active_mask = correction_margin(refs, hyps, self.margin)
            used, active = indices.numel(), int(active_mask.sum())
            ref_mean, hyp_mean = float(refs.detach().mean()), float(hyps.detach().mean())
        total = asr + self.correction_weight * correction
        if not torch.isfinite(total):
            if self.audit:
                self.audit.failure('Non-finite loss')
            raise FloatingPointError('Non-finite E3 loss')
        self.last_components = dict(asr_loss=float(asr.detach()), correction_loss=float(correction.detach()),
            total_loss=float(total.detach()), eligible=int(indices.numel()) if indices is not None else 0,
            used=used, active=active, active_fraction=active/used if used else 0.,
            reference_score=ref_mean, hypothesis_score=hyp_mean)
        # Exposed only for diagnostic autograd probes, not retained across batches.
        output.loss = total
        if model.training:
            if self.audit:
                self.audit.add(self.last_components, inputs['labels'].shape[0])
            for key in ('asr_loss', 'correction_loss', 'total_loss'):
                self._log_sums[key] += self.last_components[key]
            self._log_batches += 1
        if return_outputs:
            return total, dict(output, asr_loss=asr, correction_loss=correction)
        return total

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        # No correction on Validation/Test, even if a caller supplies extra fields.
        clean = {k: v for k, v in inputs.items() if not k.startswith('correction_')}
        return super().prediction_step(model, clean, prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        if gradients and not torch.isfinite(torch.stack(torch._foreach_norm(gradients))).all():
            if self.audit:
                self.audit.failure('Non-finite gradient')
            raise FloatingPointError('Non-finite E3 gradient')
        return loss

    def log(self, logs, start_time=None):
        logs = dict(logs)
        if 'loss' in logs and self._log_batches:
            logs.update({k: v/self._log_batches for k, v in self._log_sums.items()})
            self._log_sums.clear(); self._log_batches = 0
        super().log(logs, start_time=start_time)
        if self.audit:
            with (self.audit.directory / 'training_logs.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(logs, step=self.state.global_step), allow_nan=False) + '\n')


def make_e3_trainer(model, processor, datasets, directory, settings, correction_weight=.1,
                    margin=.1, accumulation_mode='legacy_e1', use_cpu=False, audit=None):
    cuda = torch.cuda.is_available() and not use_cpu
    args = Seq2SeqTrainingArguments(output_dir=str(directory), use_cpu=use_cpu,
        per_device_train_batch_size=int(settings['batch_size']), per_device_eval_batch_size=int(settings['eval_batch_size']),
        gradient_accumulation_steps=int(settings['gradient_accumulation']), learning_rate=float(settings['learning_rate']),
        num_train_epochs=float(settings['epochs']), warmup_steps=0, optim='adamw_torch',
        adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8, weight_decay=0., max_grad_norm=1., lr_scheduler_type='linear',
        predict_with_generate=True, generation_max_length=int(settings['generation_max_length']), generation_num_beams=1,
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='cer', greater_is_better=False, save_total_limit=5,
        logging_steps=int(settings.get('logging_steps', 20)),
        fp16=cuda and not torch.cuda.is_bf16_supported(), bf16=cuda and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=bool(settings.get('gradient_checkpointing', True)),
        report_to='none', remove_unused_columns=False, seed=int(settings['seed']), data_seed=None,
        label_names=['labels'])
    return E3Trainer(model=model, args=args, train_dataset=datasets['train'], eval_dataset=datasets['val'],
        data_collator=E3Collator(processor), processing_class=processor, correction_weight=correction_weight,
        margin=margin, accumulation_mode=accumulation_mode, audit=audit, callbacks=[audit] if audit else None,
        compute_metrics=lambda p: compute_asr_metrics(p.predictions, p.label_ids, processor.tokenizer))
