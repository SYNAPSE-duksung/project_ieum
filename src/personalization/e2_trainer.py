"""E2 Trainer integration: separate generation inputs and supervised loss."""
from collections import defaultdict
import json
import math

import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperConfig, WhisperProcessor

from .ctc_head import WhisperWithAuxiliaryCTC, CTC_KEYS
from .ctc_utils import E2Collator
from .evaluator import compute_asr_metrics
from .e2_monitor import E2RunRecorder


def build_e2_model_and_processor(general_model, lambda_ctc, language='korean', task='transcribe'):
    config = WhisperConfig.from_pretrained(general_model, local_files_only=True)
    if hasattr(config, 'ctc_vocab_version'):
        raise ValueError('Training must start from the General checkpoint, not an E2 checkpoint')
    config.lambda_ctc = float(lambda_ctc)
    model, info = WhisperWithAuxiliaryCTC.from_pretrained(
        general_model, config=config, local_files_only=True, output_loading_info=True)
    if set(info['missing_keys']) != {'ctc_head.weight', 'ctc_head.bias'} or info['unexpected_keys'] or info.get('mismatched_keys'):
        raise ValueError(f'Unexpected General checkpoint loading report: {info}')
    model.requires_grad_(True)
    processor = WhisperProcessor.from_pretrained(
        'openai/whisper-small', language=language, task=task, local_files_only=True)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = 160
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    return model, processor


class E2Trainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.run_recorder = kwargs.pop('run_recorder', None)
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self._components = defaultdict(lambda: [0., 0])

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            outputs = model(**inputs)
        except FloatingPointError as exc:
            if self.run_recorder:
                self.run_recorder.nonfinite('forward', self.state, exc)
            raise
        prefix = '' if model.training else 'eval_'
        count = inputs['labels'].shape[0]
        if model.training and self.run_recorder:
            self.run_recorder.record_train(outputs, count, self.state)
        for name, value in (('asr_loss', outputs.asr_loss), ('ctc_loss', outputs.ctc_loss),
                            ('weighted_ctc_loss', outputs.weighted_ctc_loss), ('total_loss', outputs.loss)):
            if value is not None:
                item = self._components[prefix + name]
                item[0] += float(value.detach()) * count
                item[1] += count
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.run_recorder:
            gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            finite = torch.isfinite(loss).all()
            if gradients:
                finite = finite & torch.isfinite(torch.stack(torch._foreach_norm(gradients))).all()
            if not bool(finite):
                self.run_recorder.nonfinite('backward', self.state, 'Non-finite loss/gradient norm')
                raise FloatingPointError('Non-finite loss/gradient norm')
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        # Parent Seq2SeqTrainer calls model(**inputs) instead of compute_loss in
        # generate mode. Compute E2 loss explicitly, then delegate generation
        # with NO reference tensors. Only decoder IDs are returned for metrics.
        prepared = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, prepared).detach().mean() if 'labels' in prepared else None
        if prediction_loss_only:
            return loss, None, None
        if not self.args.predict_with_generate:
            raise ValueError('E2 evaluation requires predict_with_generate=True')
        generation_inputs = {k: v for k, v in prepared.items()
                             if k not in (*CTC_KEYS, 'labels', 'decoder_input_ids', 'decoder_attention_mask')}
        _, generated, _ = super().prediction_step(
            model, generation_inputs, False, ignore_keys=ignore_keys, **gen_kwargs)
        labels = prepared.get('labels')
        if labels is not None and labels.shape[-1] < generated.shape[-1]:
            labels = torch.nn.functional.pad(labels, (0, generated.shape[-1] - labels.shape[-1]), value=-100)
        return loss, generated, labels

    def log(self, logs, start_time=None):
        logs = dict(logs)
        if self.run_recorder:
            for key, value in logs.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    self.run_recorder.nonfinite('log', self.state, f'{key}={value}')
                    raise FloatingPointError(f'Non-finite logged metric: {key}')
        prefix = 'eval_' if any(k.startswith('eval_') for k in logs) else ''
        if 'loss' in logs or 'eval_loss' in logs:
            for name in ('asr_loss', 'ctc_loss', 'weighted_ctc_loss', 'total_loss'):
                key = prefix + name
                total, count = self._components.pop(key, (0., 0))
                if count:
                    logs[key] = total / count
        super().log(logs, start_time=start_time)
        if self.run_recorder and self.is_world_process_zero():
            with (self.run_recorder.run_dir / 'training_logs.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(logs, step=self.state.global_step), default=float, allow_nan=False) + '\n')


def make_e2_trainer(model, processor, datasets, output_dir, settings, use_cpu=False, run_dir=None):
    cuda = torch.cuda.is_available() and not use_cpu
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir), use_cpu=use_cpu,
        per_device_train_batch_size=int(settings['batch_size']),
        per_device_eval_batch_size=int(settings['eval_batch_size']),
        gradient_accumulation_steps=int(settings['gradient_accumulation']),
        learning_rate=float(settings['learning_rate']), num_train_epochs=float(settings['epochs']),
        warmup_steps=0, predict_with_generate=True,
        generation_max_length=int(settings['generation_max_length']), generation_num_beams=1,
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='cer', greater_is_better=False, save_total_limit=2,
        logging_steps=int(settings.get('logging_steps', 20)),
        fp16=cuda and not torch.cuda.is_bf16_supported(),
        bf16=cuda and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=bool(settings.get('gradient_checkpointing', True)),
        report_to='none', remove_unused_columns=False, seed=int(settings.get('seed', 42)),
        label_names=['labels'],
    )
    recorder = E2RunRecorder(run_dir) if run_dir is not None else None
    return E2Trainer(model=model, args=args, train_dataset=datasets['train'],
                     eval_dataset=datasets['val'], data_collator=E2Collator(processor),
                     processing_class=processor, run_recorder=recorder,
                     callbacks=[recorder] if recorder else None,
                     compute_metrics=lambda p: compute_asr_metrics(p.predictions, p.label_ids, processor.tokenizer))
