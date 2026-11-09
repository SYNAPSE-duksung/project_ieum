"""CPU regression for the new model-only checkpoint lifecycle."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_utils import SaveStrategy
import pilot_runner as p


class PilotTests(unittest.TestCase):
    def test_frozen_selection(self):
        selected = p.frozen()
        self.assertEqual(len(selected),10)
        self.assertNotIn('KSM_M_45', {r['speaker_id'] for r in selected})
        self.assertEqual(next(r for r in selected if r['speaker_id']=='YHS_F_26')['usable_correction_candidates'],'0')

    def test_space(self):
        plan = p.storage(p.ROOT/'outputs/personalization/pilot_e1_e3_v1')
        self.assertLess(plan['required_with_reserve_gib'],30)

    def test_model_only_rotation_and_reload(self):
        cfg = WhisperConfig(vocab_size=32,d_model=8,encoder_layers=1,decoder_layers=1,
            encoder_attention_heads=1,decoder_attention_heads=1,encoder_ffn_dim=16,decoder_ffn_dim=16,
            num_mel_bins=4,max_source_positions=4,max_target_positions=8,
            pad_token_id=0,bos_token_id=1,eos_token_id=2,decoder_start_token_id=1)
        report = p.ROOT/'reports/pilot_runner_setup_20261107'
        report.mkdir(parents=True,exist_ok=True)
        # Only disposable, newly created test files are rotated/cleaned here.
        with tempfile.TemporaryDirectory(prefix='checkpoint_probe_',dir=report) as directory:
            model = WhisperForConditionalGeneration(cfg)
            args = Seq2SeqTrainingArguments(output_dir=directory,use_cpu=True,report_to='none',
                save_strategy='best',eval_strategy='epoch',save_only_model=True,save_total_limit=1,
                load_best_model_at_end=True,metric_for_best_model='cer',greater_is_better=False)
            t = Seq2SeqTrainer(model=model,args=args,eval_dataset=[{'labels':[1,2]}])
            for step in (1,2):
                t.state.global_step=step
                t.state.best_global_step=step
                t.state.best_metric=.3/step
                t._save_checkpoint(model,None)
            checkpoints=list(Path(directory).glob('checkpoint-*'))
            self.assertEqual([v.name for v in checkpoints],['checkpoint-2'])
            files={v.name for v in checkpoints[0].iterdir()}
            self.assertNotIn('optimizer.pt',files)
            self.assertNotIn('scheduler.pt',files)
            self.assertFalse(any(n.startswith('rng_state') for n in files))
            restored=WhisperForConditionalGeneration.from_pretrained(checkpoints[0])
            self.assertTrue(all(torch.equal(v,restored.state_dict()[n]) for n,v in model.state_dict().items()))
            p.write(report/'checkpoint_probe.json',dict(retained_checkpoint='checkpoint-2',
                retained_count=1,optimizer_saved=False,scheduler_saved=False,weights_reload_exact=True,
                note='Synthetic tiny Whisper checkpoint lifecycle; not Personal training or Test evaluation'))

    def test_short_training_selects_validation_best(self):
        cfg = WhisperConfig(vocab_size=32,d_model=8,encoder_layers=1,decoder_layers=1,
            encoder_attention_heads=1,decoder_attention_heads=1,encoder_ffn_dim=16,decoder_ffn_dim=16,
            num_mel_bins=4,max_source_positions=4,max_target_positions=8,apply_spec_augment=False,
            pad_token_id=0,bos_token_id=1,eos_token_id=2,decoder_start_token_id=1,use_cache=False)
        report = p.ROOT/'reports/pilot_runner_setup_20261107'
        with tempfile.TemporaryDirectory(prefix='short_train_',dir=report) as directory:
            model=WhisperForConditionalGeneration(cfg)
            args=Seq2SeqTrainingArguments(output_dir=directory,use_cpu=True,report_to='none',
                per_device_train_batch_size=2,gradient_accumulation_steps=4,num_train_epochs=2,
                save_strategy='best',eval_strategy='epoch',save_only_model=True,save_total_limit=1,
                load_best_model_at_end=True,metric_for_best_model='cer',greater_is_better=False)
            samples=[dict(input_features=torch.zeros(4,8),labels=torch.tensor([3,2])) for _ in range(8)]
            calls=[]
            def metrics(prediction):
                calls.append(1)
                return dict(cer=.1 if len(calls)==1 else .2,wer=.2)
            t=Seq2SeqTrainer(model=model,args=args,train_dataset=samples,eval_dataset=samples[:2],compute_metrics=metrics)
            t.train()
            self.assertEqual(t.state.global_step,2)
            self.assertEqual(t.state.best_metric,.1)
            self.assertEqual(Path(t.state.best_model_checkpoint).name,'checkpoint-1')
            self.assertEqual(len(list(Path(directory).glob('checkpoint-*'))),1)
            restored=WhisperForConditionalGeneration.from_pretrained(t.state.best_model_checkpoint)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[n]) for n,v in model.state_dict().items()))
            self.assertEqual(model.proj_out.weight.data_ptr(),model.model.decoder.embed_tokens.weight.data_ptr())
            p.write(report/'short_training_probe.json',dict(optimizer_steps=2,synthetic_epochs=2,
                best_checkpoint='checkpoint-1',best_cer=.1,validation_best_selected=True,
                best_reload_weights_exact=True,tied_proj_out_verified=True,
                note='Tiny CPU synthetic model only; not a pilot speaker full training'))


if __name__ == '__main__':
    unittest.main()
