"""CPU regression checks for E2; no training or original-file writes.

python scripts/07_personalization/test_e2.py
Full General-checkpoint batch check: e2_train.py --mode smoke
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import librosa
import numpy as np
import torch
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.personalization.ctc_head import WhisperWithAuxiliaryCTC
from src.personalization.ctc_utils import (
    ID_TO_TOKEN, TOKEN_TO_ID, VOCAB_SIZE, encode_ctc, decode_ctc,
    minimum_ctc_frames, unsupported_characters, wav_lengths, encoder_output_lengths,
    E2Collator, preflight_splits,
)
from src.personalization.e2_trainer import make_e2_trainer
from src.personalization.evaluator import normalize_text, compute_asr_metrics
from src.personalization.dataset import WhisperCollator
from e2_train import load_config, load_data, check_vocabulary, lambda_value, lambda_directory, fingerprint


def compact_model():
    # Preserve the real 768 -> 11183 head, shrink only Transformer depth/time.
    config = WhisperConfig(d_model=768, encoder_layers=1, decoder_layers=1,
                           encoder_attention_heads=12, decoder_attention_heads=12,
                           encoder_ffn_dim=32, decoder_ffn_dim=32, vocab_size=32,
                           max_source_positions=8, max_target_positions=16,
                           pad_token_id=0, bos_token_id=1, eos_token_id=2,
                           decoder_start_token_id=1, suppress_tokens=[], begin_suppress_tokens=[])
    config.lambda_ctc = .3
    return WhisperWithAuxiliaryCTC(config)


class E2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.processor = WhisperProcessor.from_pretrained(
            'openai/whisper-small', language='korean', task='transcribe', local_files_only=True)
        cls.model = compact_model()

    def batch(self):
        return dict(input_features=torch.randn(2, 80, 16),
                    attention_mask=torch.tensor([[1] * 7 + [0] * 9, [1] * 16]),
                    labels=torch.tensor([[4, 5, 2], [5, 6, 2]]),
                    ctc_targets=torch.tensor(encode_ctc('700가')),
                    ctc_target_lengths=torch.tensor([3, 1]),
                    ctc_input_lengths=torch.tensor([4, 8]))

    def test_fixed_vocabulary_and_round_trips(self):
        check_vocabulary()
        self.assertEqual(len(set(ID_TO_TOKEN)), VOCAB_SIZE)
        text = ''.join(ID_TO_TOKEN[1:])
        self.assertEqual(decode_ctc(encode_ctc(text)), text)
        self.assertEqual(encode_ctc('0 9'), [11173, 11182])
        with self.assertRaises(ValueError):
            decode_ctc([0])

    def test_explicit_unsupported_policy(self):
        self.assertEqual(decode_ctc(encode_ctc('가, 3!\t')), '가3')
        self.assertEqual(dict(unsupported_characters('AA+가🙂')), {'A': 2, '+': 1, 'ᄀ': 1, 'ᅡ': 1, '🙂': 1})
        for text in ('가A', '１２', '가', '가+나'):
            with self.assertRaises(ValueError):
                encode_ctc(text)

    def test_ctc_repeated_numbers(self):
        self.assertEqual(minimum_ctc_frames(encode_ctc('700')), 4)
        self.assertEqual(minimum_ctc_frames(encode_ctc('000')), 5)
        batch = self.batch()
        batch['attention_mask'][0] = torch.tensor([1] * 6 + [0] * 10)
        batch['ctc_input_lengths'][0] = 3
        with self.assertRaisesRegex(ValueError, 'alignment impossible'):
            self.model(**batch)

    def test_actual_extractor_padding_truncation_boundaries(self):
        f = self.processor.feature_extractor
        for n in (1, 159, 160, 161, 319, 320, 321, 479999, 480000, 480001):
            with self.subTest(samples=n):
                out = f(np.zeros(n, np.float32), sampling_rate=16000,
                        return_attention_mask=True, return_tensors='pt')
                _, mel, valid = wav_lengths(n, 16000, f, self.model.model.encoder)
                self.assertEqual(mel, int(out.attention_mask.sum()))
                self.assertEqual(valid, (mel + 1) // 2)
                self.assertEqual(out.input_features.shape[-1], 3000)
        for n in (1, 441, 442, 44101):
            converted = librosa.resample(np.zeros(n, np.float32), orig_sr=44100, target_sr=16000)
            self.assertEqual(wav_lengths(n, 44100, f, self.model.model.encoder)[0], len(converted))

    def test_both_losses_backpropagate_and_padding_lengths(self):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        output = self.model(**self.batch())
        self.assertTrue(torch.isfinite(output.ctc_loss))
        self.assertTrue(torch.allclose(output.loss, output.asr_loss + .3 * output.ctc_loss))
        probe = self.model.model.encoder.conv1.weight
        for loss in (output.asr_loss, output.ctc_loss):
            gradient = torch.autograd.grad(loss, probe, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)
        output.loss.backward()
        for parameter in (self.model.ctc_head.weight, self.model.model.decoder.layers[0].fc1.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.model.zero_grad(set_to_none=True)
        batch = self.batch()
        batch['ctc_input_lengths'][0] = 8  # Incorrectly using padded time must fail.
        with self.assertRaisesRegex(ValueError, 'attention mask'):
            self.model(**batch)

    def test_checkpoint_roundtrip_and_decoder_generation(self):
        self.model.eval()
        batch = self.batch()
        # No CTC computation may occur during generation.
        def forbid_ctc(*args):
            raise AssertionError('CTC head called during decoder generation')
        handle = self.model.ctc_head.register_forward_hook(forbid_ctc)
        try:
            with torch.no_grad():
                generated = self.model.generate(**batch, max_new_tokens=3, language=None, task=None)
        finally:
            handle.remove()
        with tempfile.TemporaryDirectory(prefix='ieum_e2_unit_') as temporary:
            self.model.save_pretrained(temporary)
            restored = WhisperWithAuxiliaryCTC.from_pretrained(temporary, local_files_only=True)
            self.assertTrue(torch.equal(restored.ctc_head.weight, self.model.ctc_head.weight))
            self.assertEqual(restored.config.lambda_ctc, .3)
            restored.eval()
            with torch.no_grad():
                again = restored.generate(**batch, max_new_tokens=3, language=None, task=None)
            self.assertTrue(torch.equal(generated, again))

    def test_trainer_generation_loss_logging_and_accumulation_contract(self):
        from transformers import Seq2SeqTrainingArguments
        from src.personalization.e2_trainer import E2Trainer
        with tempfile.TemporaryDirectory(prefix='ieum_e2_trainer_') as temporary:
            args = Seq2SeqTrainingArguments(output_dir=temporary, use_cpu=True, report_to='none',
                                           predict_with_generate=True, label_names=['labels'],
                                           generation_max_length=8, gradient_accumulation_steps=4)
            # Real evaluation loop with per-example features/collation.
            batch = self.batch()
            examples = []
            for i, text in enumerate(('700', '가')):
                examples.append(dict(input_features=batch['input_features'][i],
                                     attention_mask=batch['attention_mask'][i],
                                     labels=batch['labels'][i].tolist(),
                                     ctc_targets=encode_ctc(text), ctc_target_lengths=len(text),
                                     ctc_input_lengths=int(batch['ctc_input_lengths'][i])))
            trainer = E2Trainer(model=self.model, args=args, eval_dataset=examples,
                                data_collator=E2Collator(self.processor),
                                compute_metrics=lambda p: compute_asr_metrics(p.predictions, p.label_ids,
                                                                              self.processor.tokenizer))
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            self.model.train()
            trainer.compute_loss(self.model, self.batch())
            trainer.log({'loss': 1.})
            for key in ('asr_loss', 'ctc_loss', 'weighted_ctc_loss', 'total_loss'):
                self.assertIn(key, trainer.state.log_history[-1])
            self.model.eval()
            loss, ids, labels = trainer.prediction_step(self.model, self.batch(), False, max_new_tokens=3)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(ids.ndim, 2)
            self.assertEqual(labels.shape[0], 2)
            trainer.log({'eval_loss': float(loss), 'eval_cer': 0.})
            self.assertIn('eval_ctc_loss', trainer.state.log_history[-1])
            loss_only, ids, labels = trainer.prediction_step(self.model, self.batch(), True)
            self.assertTrue(torch.isfinite(loss_only))
            self.assertIsNone(ids)
            metrics = trainer.evaluate(max_new_tokens=3)
            self.assertIn('eval_cer', metrics)
            self.assertIn('eval_wer', metrics)
            checkpoint = Path(temporary) / 'checkpoint-1'
            trainer.save_model(str(checkpoint))
            before = self.model.ctc_head.weight.detach().clone()
            with torch.no_grad():
                self.model.ctc_head.weight.zero_()
            trainer.state.best_model_checkpoint = str(checkpoint)
            trainer._load_best_model()
            self.assertTrue(torch.equal(before, self.model.ctc_head.weight))

    def test_pilot_all_splits_and_six_digit_train_utterances(self):
        config = load_config(ROOT / 'scripts/07_personalization/configs/e2.yaml')
        with torch.device('meta'):
            from transformers.models.whisper.modeling_whisper import WhisperEncoder
            encoder = WhisperEncoder(WhisperConfig.from_pretrained(config['paths']['general_model'], local_files_only=True))
        with contextlib.redirect_stdout(io.StringIO()):
            rows, datasets, report = load_data(config, 'KSM_M_45', self.processor, encoder)
        self.assertEqual({k: len(v) for k, v in rows.items()}, {'train': 168, 'val': 65, 'test': 67})
        self.assertEqual(len(report['digit_train']), 6)
        self.assertEqual(report['unsupported'], [])
        self.assertEqual(report['issues'], [])
        collator = E2Collator(self.processor)
        for item in report['digit_train']:
            text = item['text']
            self.assertEqual([c for c in decode_ctc(item['target_ids']) if c in '0123456789'],
                             [c for c in text if c in '0123456789'])
            index = next(i for i, row in enumerate(datasets['train'].rows) if row['text'] == text)
            example = datasets['train'][index]
            self.assertEqual(example['ctc_targets'], item['target_ids'])
            self.assertGreaterEqual(example['ctc_input_lengths'], minimum_ctc_frames(example['ctc_targets']))
        features = [datasets['train'][i] for i in (0, 1)]
        e1 = WhisperCollator(self.processor)(features)
        e2 = collator(features)
        for key in e1:
            self.assertTrue(torch.equal(e1[key], e2[key]))
        self.assertEqual(e2['ctc_targets'].numel(), int(e2['ctc_target_lengths'].sum()))

    def test_all_unsupported_characters_reported_before_failure(self):
        config = load_config(ROOT / 'scripts/07_personalization/configs/e2.yaml')
        with torch.device('meta'):
            from transformers.models.whisper.modeling_whisper import WhisperEncoder
            encoder = WhisperEncoder(WhisperConfig.from_pretrained(config['paths']['general_model'], local_files_only=True))
        rows = {'train': [{'text': '가AA', '_resolved_audio': 'missing.wav'}],
                'val': [{'text': '나A+', '_resolved_audio': 'missing.wav'}], 'test': []}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaisesRegex(ValueError, '2 utterances'):
            preflight_splits(rows, self.processor, encoder, 'text', config['training'])
        import json
        report = json.loads(buffer.getvalue())
        counts = {r['character']: r['count'] for r in report['unsupported']}
        self.assertEqual(counts, {'+': 1, 'A': 3})
        self.assertEqual(len(report['issues']), 2)

    def test_e1_evaluation_normalization_reused(self):
        import src.personalization.e2_trainer as e2
        self.assertIs(e2.compute_asr_metrics, compute_asr_metrics)
        self.assertEqual(normalize_text('가, A 700!'), '가 a 700')

    def test_flexible_lambda_and_comparison_identity(self):
        import argparse
        import copy
        config = load_config(ROOT / 'scripts/07_personalization/configs/e2.yaml')
        self.assertEqual(config['e2']['lambda_ctc_values'], [.03, .1, .3])
        for value in ('0.03', '0.05', '0.1', '0.2', '0.3', '0'):
            self.assertGreaterEqual(lambda_value(value), 0)
        for value in ('nan', 'inf', '-0.1', 'bad'):
            with self.assertRaises(argparse.ArgumentTypeError):
                lambda_value(value)
        self.assertEqual(lambda_directory(.03), 'lambda_0.03')
        self.assertEqual(lambda_directory(float('0.030')), 'lambda_0.03')
        self.assertNotEqual(lambda_directory(.100000001), lambda_directory(.1))
        expanded = copy.deepcopy(config)
        expanded['e2']['lambda_ctc_values'].extend([.05, .2])
        self.assertEqual(fingerprint(config, 'KSM_M_45'), fingerprint(expanded, 'KSM_M_45'))

    def test_epoch_audit_and_nonfinite_persistence(self):
        import json
        from types import SimpleNamespace
        from transformers import TrainerState, TrainerControl
        from src.personalization.e2_monitor import E2RunRecorder
        args = SimpleNamespace(device=torch.device('cpu'), bf16=False, fp16=False)
        state, control = TrainerState(), TrainerControl()
        with tempfile.TemporaryDirectory(prefix='ieum_e2_audit_') as temporary:
            recorder = E2RunRecorder(temporary)
            recorder.on_train_begin(args, state, control)
            for epoch, cer in ((1, .2), (2, .3)):
                state.epoch, state.global_step = epoch, epoch * 21
                outputs = SimpleNamespace(asr_loss=torch.tensor(2.), ctc_loss=torch.tensor(10.),
                                          weighted_ctc_loss=torch.tensor(1.), loss=torch.tensor(3.))
                recorder.record_train(outputs, 2, state)
                recorder.on_epoch_end(args, state, control)
                recorder.on_evaluate(args, state, control, dict(eval_cer=cer, eval_wer=cer*2, eval_loss=3.))
            # Final evaluation of best checkpoint must not rewrite epoch 2.
            recorder.on_evaluate(args, state, control, dict(eval_cer=.2, eval_wer=.4, eval_loss=3.))
            recorder.on_train_end(args, state, control)
            report = json.loads((Path(temporary) / 'training_audit.json').read_text())
            self.assertEqual(report['best_epoch'], 1)
            self.assertEqual(report['epochs'][1]['validation']['eval_cer'], .3)
            self.assertEqual(report['epochs'][0]['train'], dict(asr_loss=2., ctc_loss=10., weighted_ctc_loss=1., total_loss=3.))
            self.assertFalse(report['nonfinite_observed'])
            self.assertGreater(report['training_seconds'], 0)
            bad = SimpleNamespace(asr_loss=torch.tensor(float('nan')), ctc_loss=torch.tensor(10.),
                                  weighted_ctc_loss=torch.tensor(1.), loss=torch.tensor(float('nan')))
            with self.assertRaises(FloatingPointError):
                recorder.record_train(bad, 1, state)
            report = json.loads((Path(temporary) / 'training_audit.json').read_text())
            self.assertTrue(report['nonfinite_observed'])
            self.assertEqual(report['nonfinite_events'][0]['stage'], 'asr_loss')

    def test_trainer_microbatch_audit_detects_nonfinite_gradients(self):
        import json
        from transformers import Seq2SeqTrainingArguments
        from src.personalization.e2_monitor import E2RunRecorder
        from src.personalization.e2_trainer import E2Trainer
        with tempfile.TemporaryDirectory(prefix='ieum_e2_backward_audit_') as temporary:
            args = Seq2SeqTrainingArguments(output_dir=temporary, use_cpu=True, report_to='none')
            recorder = E2RunRecorder(temporary)
            trainer = E2Trainer(model=self.model, args=args, run_recorder=recorder, callbacks=[recorder])
            trainer.current_gradient_accumulation_steps = 1
            recorder.on_train_begin(args, trainer.state, trainer.control)
            self.model.zero_grad(set_to_none=True)
            trainer.training_step(self.model, self.batch())  # Backward only, no optimizer step.
            self.assertEqual(recorder.samples, 2)
            trainer.log({'loss': 1.})
            self.assertTrue((Path(temporary) / 'training_logs.jsonl').is_file())
            self.model.zero_grad(set_to_none=True)
            handle = self.model.ctc_head.weight.register_hook(lambda g: torch.full_like(g, float('inf')))
            try:
                with self.assertRaisesRegex(FloatingPointError, 'gradient'):
                    trainer.training_step(self.model, self.batch())
            finally:
                handle.remove()
                self.model.zero_grad(set_to_none=True)
            report = json.loads((Path(temporary) / 'training_audit.json').read_text())
            self.assertTrue(report['nonfinite_observed'])
            self.assertEqual(report['nonfinite_events'][-1]['stage'], 'backward')


if __name__ == '__main__':
    unittest.main(verbosity=2)
