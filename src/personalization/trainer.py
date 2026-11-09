"""E1 ordinary full fine-tuning of Whisper encoder and decoder."""
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from .dataset import WhisperCollator
from .evaluator import compute_asr_metrics


def build_model_and_processor(model_path, language='korean', task='transcribe'):
    processor = WhisperProcessor.from_pretrained("openai/whisper-small",language=language,task=task)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    # FULL FT: encoder + decoder are both trainable.
    model.requires_grad_(True)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None
    # 기존 범용 모델과 생성 설정 통일
    model.generation_config.max_length = 160
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    print('Trainable encoder:', sum(p.numel() for p in model.model.encoder.parameters() if p.requires_grad))
    print('Trainable decoder:', sum(p.numel() for p in model.model.decoder.parameters() if p.requires_grad))
    return model, processor


def make_trainer(model, processor, train_dataset, val_dataset, output_dir, settings):
    fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(settings['batch_size']),
        per_device_eval_batch_size=int(settings['eval_batch_size']),
        gradient_accumulation_steps=int(settings['gradient_accumulation']),
        learning_rate=float(settings['learning_rate']),
        num_train_epochs=float(settings['epochs']),
        warmup_steps=0,
        predict_with_generate=True,
        generation_max_length=int(settings['generation_max_length']),
        generation_num_beams=1,
        eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='cer', greater_is_better=False,
        save_total_limit=2, logging_steps=int(settings.get('logging_steps', 20)),
        fp16=fp16, bf16=bf16, gradient_checkpointing=bool(settings.get('gradient_checkpointing', True)),
        report_to='none', remove_unused_columns=False, seed=int(settings.get('seed', 42)),
    )
    return Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=WhisperCollator(processor), processing_class=processor,
        compute_metrics=lambda p: compute_asr_metrics(p.predictions, p.label_ids, processor.tokenizer),
    )
