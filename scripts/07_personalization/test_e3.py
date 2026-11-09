"""E3 CPU tests; no optimizer steps or existing-file writes."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.personalization.e3_data import E3Collator, join_hypotheses
from src.personalization.e3_loss import token_mean_log_probability, correction_margin
from src.personalization.e3_trainer import make_e3_trainer
from src.personalization.dataset import WhisperCollator
from src.personalization.evaluator import compute_asr_metrics
from e3_train import load_config, load_data


class E3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.processor=WhisperProcessor.from_pretrained('openai/whisper-small',language='korean',task='transcribe',local_files_only=True)
        cfg=WhisperConfig(d_model=16,encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,
            decoder_attention_heads=2,encoder_ffn_dim=32,decoder_ffn_dim=32,max_source_positions=8,
            max_target_positions=160,vocab_size=51865,pad_token_id=50257,bos_token_id=50257,
            eos_token_id=50257,decoder_start_token_id=50258,suppress_tokens=[],begin_suppress_tokens=[])
        cls.model=WhisperForConditionalGeneration(cfg).requires_grad_(True)
        cls.config=load_config(ROOT/'scripts/07_personalization/configs/e3.yaml')
        cls.settings=dict(cls.config['training'],gradient_checkpointing=False)

    def features(self,eligible=True):
        return [dict(input_features=torch.randn(80,16),attention_mask=torch.ones(16,dtype=torch.long),
            labels=self.processor.tokenizer(text,add_special_tokens=True).input_ids,
            correction_eligible=eligible,correction_labels=self.processor.tokenizer(hyp,add_special_tokens=True).input_ids if eligible else [])
            for text,hyp in [('오늘 뉴스 알려 줘','오늘 뉴스 알려 주세요'),('가방이 있어요','가방이 없어요')]]

    def trainer(self,directory,weight=.1,margin=.1,mode='legacy_e1'):
        return make_e3_trainer(self.model,self.processor,dict(train=[],val=[]),directory,
                               self.settings,weight,margin,mode,use_cpu=True)

    def test_score_mask_padding_prefix_and_eos(self):
        logits=torch.randn(1,6,10,requires_grad=True)
        labels=torch.tensor([[1,2,5,9,-100,-100]])
        score=token_mean_log_probability(logits,labels,[1,2,9],9)
        expected=logits.log_softmax(-1)[0,[2,3],[5,9]].mean()
        self.assertTrue(torch.allclose(score,expected))
        changed=logits.detach().clone();changed[0,:2]=100;changed[0,4:]=-100
        self.assertTrue(torch.allclose(score,token_mean_log_probability(changed,labels,[1,2,9],9)))
        changed[0,3,9]+=2
        self.assertFalse(torch.allclose(score,token_mean_log_probability(changed,labels,[1,2,9],9)))

    def test_margin_formula(self):
        loss,active=correction_margin(torch.tensor([-2.,-1.]),torch.tensor([-1.,-3.]),.1)
        self.assertAlmostEqual(float(loss),.55,places=6)
        self.assertEqual(active.tolist(),[True,False])

    def test_collator_preserves_e1_labels(self):
        features=self.features()
        base=WhisperCollator(self.processor)(features);e3=E3Collator(self.processor)(features)
        for key in base:self.assertTrue(torch.equal(base[key],e3[key]))

    def test_zero_lambda_loss_and_gradients_match_e1_both_modes(self):
        for mode in ('legacy_e1','mean'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory(prefix='ieum_e3_zero_') as temp:
                trainer=self.trainer(temp,weight=0,mode=mode)
                base=Seq2SeqTrainer(model=self.model,args=copy.deepcopy(trainer.args),processing_class=self.processor,eval_dataset=[])
                if mode=='mean':base.model_accepts_loss_kwargs=False
                trainer.current_gradient_accumulation_steps=base.current_gradient_accumulation_steps=4
                batch=E3Collator(self.processor)(self.features());clean={k:v for k,v in batch.items() if not k.startswith('correction_')}
                self.model.zero_grad(set_to_none=True)
                a=trainer.training_step(self.model,batch,num_items_in_batch=torch.tensor(20))
                gradient=self.model.model.encoder.conv1.weight.grad.clone()
                self.model.zero_grad(set_to_none=True)
                b=base.training_step(self.model,clean,num_items_in_batch=torch.tensor(20))
                self.assertTrue(torch.allclose(a,b))
                self.assertTrue(torch.allclose(gradient,self.model.model.encoder.conv1.weight.grad))
                self.assertEqual(trainer.last_components['used'],0)
                self.model.zero_grad(set_to_none=True)

    def test_correction_gradient_reaches_encoder_and_decoder(self):
        with tempfile.TemporaryDirectory(prefix='ieum_e3_grad_') as temp:
            trainer=self.trainer(temp,margin=100)
            self.model.train();batch=E3Collator(self.processor)(self.features())
            total,output=trainer.compute_loss(self.model,batch,return_outputs=True)
            self.assertEqual(trainer.last_components['active'],2)
            gradients=torch.autograd.grad(output['correction_loss'],
                (self.model.model.encoder.conv1.weight,self.model.model.decoder.layers[0].fc1.weight))
            for g in gradients:
                self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.abs().sum()),0)

    def test_all_same_zero_and_eval_no_correction(self):
        with tempfile.TemporaryDirectory(prefix='ieum_e3_same_') as temp:
            trainer=self.trainer(temp)
            self.model.train();batch=E3Collator(self.processor)(self.features(False))
            trainer.compute_loss(self.model,batch)
            self.assertEqual(trainer.last_components['correction_loss'],0)
            self.model.eval();trainer.compute_loss(self.model,E3Collator(self.processor)(self.features()))
            self.assertEqual(trainer.last_components['used'],0)
            loss,ids,labels=trainer.prediction_step(self.model,batch,False,max_new_tokens=3)
            self.assertTrue(torch.isfinite(loss));self.assertEqual(ids.ndim,2)

    def test_real_train_cache_and_no_val_test_candidates(self):
        rows,datasets,report=load_data(self.config,'KSM_M_45',self.processor)
        self.assertEqual(report['counts'],dict(same=134,eligible=34))
        self.assertEqual(len(rows['train']),168)
        self.assertTrue(all('_general_hypothesis' not in r for r in rows['val']+rows['test']))

    def test_empty_long_and_missing_candidate_policies(self):
        with tempfile.TemporaryDirectory(prefix='ieum_e3_cache_') as temp:
            p=Path(temp);rows=[dict(batch='B',audio_member=f'{i}.wav',speaker_id='S',personal_split='profile_train',text='가',_resolved_audio=f'{i}.wav') for i in range(3)]
            cached=[dict(speaker_id='S',batch='B',audio_member=f'{i}.wav',reference='가',prediction=h) for i,h in enumerate(['가','','나 '*30])]
            with (p/'pred.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(cached[0]));writer.writeheader();writer.writerows(cached)
            profile=dict(speaker_id='S',split='profile_train',utterance_count=3,
                experiment_info=dict(model_dir=str(p),profile_source='whisper_predictions'))
            (p/'profile.json').write_text(json.dumps(profile),encoding='utf-8')
            joined,report=join_hypotheses(rows,p/'pred.csv',p/'profile.json',p,self.processor,10)
            self.assertEqual(report['counts'],dict(same=1,empty=1,too_long=1))
            self.assertEqual(len(joined),3)
            with self.assertRaisesRegex(ValueError,'exactly'):
                join_hypotheses(rows[:2],p/'pred.csv',p/'profile.json',p,self.processor,10)

    def test_stock_model_save_reload_and_no_added_parameters(self):
        with tempfile.TemporaryDirectory(prefix='ieum_e3_save_') as temp:
            self.model.save_pretrained(temp)
            restored=WhisperForConditionalGeneration.from_pretrained(temp,local_files_only=True)
            self.assertFalse(any('ctc' in key for key in restored.state_dict()))
            self.assertTrue(torch.equal(restored.model.encoder.conv1.weight,self.model.model.encoder.conv1.weight))

    def test_e1_metric_function_identity(self):
        import src.personalization.e3_trainer as module
        self.assertIs(module.compute_asr_metrics,compute_asr_metrics)

    def test_epoch_recording_keeps_best_and_original_epoch_validation(self):
        from types import SimpleNamespace
        from src.personalization.e3_trainer import E3Audit
        with tempfile.TemporaryDirectory(prefix='ieum_e3_audit_') as temp:
            audit=E3Audit(temp)
            for epoch,cer in ((1,.1),(2,.2)):
                stats=dict(asr_loss=2.,correction_loss=.3,total_loss=2.03,eligible=1,used=1,active=1,
                           reference_score=-1.,hypothesis_score=-.8)
                audit.add(stats,2)
                state=SimpleNamespace(epoch=epoch,global_step=epoch*21)
                audit.on_epoch_end(None,state,None)
                audit.on_evaluate(None,state,None,dict(eval_cer=cer,eval_wer=cer,eval_loss=2.))
            audit.on_evaluate(None,state,None,dict(eval_cer=.1,eval_wer=.1,eval_loss=2.))
            stored=json.loads((Path(temp)/'training_audit.json').read_text())
            self.assertEqual(stored['best_epoch'],1)
            self.assertEqual(stored['epochs'][1]['validation']['eval_cer'],.2)
            self.assertEqual(stored['epochs'][0]['candidate_usage']['used'],1)
            self.assertEqual(stored['epochs'][0]['margin_active_fraction'],1.)


if __name__=='__main__':unittest.main(verbosity=2)
