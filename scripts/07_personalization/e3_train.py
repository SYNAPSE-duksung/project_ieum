"""E3 fixed-hypothesis ranking. Default validate; full train/test are explicit."""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor, set_seed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.personalization.dataset import load_speaker_splits
from src.personalization.e3_data import E3Dataset, E3Collator, join_hypotheses, file_hash
from src.personalization.e3_trainer import make_e3_trainer, E3Audit
from src.personalization.evaluator import compute_asr_metrics, decode_predictions, save_predictions
from src.personalization.trainer import make_trainer


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_config(filename):
    extra = yaml.safe_load(filename.read_text(encoding='utf-8'))
    config = yaml.safe_load((filename.parent / extra['e1_config']).read_text(encoding='utf-8'))
    config['e3'] = extra
    return config


def build(config):
    processor = WhisperProcessor.from_pretrained('openai/whisper-small', language=config.get('language', 'korean'),
                                                 task=config.get('task', 'transcribe'), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(path(config['paths']['general_model']), local_files_only=True)
    model.requires_grad_(True)
    model.generation_config.language = config.get('language', 'korean')
    model.generation_config.task = config.get('task', 'transcribe')
    model.generation_config.forced_decoder_ids = None
    model.generation_config.max_length = 160
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    return model, processor


def load_data(config, speaker, processor):
    p, e3 = config['paths'], config['e3']
    rows = load_speaker_splits(path(p['manifest_csv']), speaker, config['columns'], config['split_labels'],
                               path(p['audio_root']), config.get('recording_id_column'))
    if {k: len(v) for k, v in rows.items()} != e3['expected_splits']:
        raise ValueError('Unexpected speaker split counts')
    rows['train'], report = join_hypotheses(rows['train'], path(e3['general_predictions']),
        path(e3['general_profile']), path(p['general_model']), processor, int(config['training']['generation_max_length']))
    import soundfile as sf
    problems = []
    for split, subset in rows.items():
        for row in subset:
            n = len(processor.tokenizer(row['text'].strip(), add_special_tokens=True).input_ids)
            if n > int(config['training']['generation_max_length']):
                problems.append(dict(split=split, audio=row['_resolved_audio'], reason=f'reference too long: {n}'))
            info = sf.info(row['_resolved_audio'])
            if info.frames <= 0 or info.duration > float(config['training']['max_audio_seconds']):
                problems.append(dict(split=split, audio=row['_resolved_audio'], reason='empty or long WAV'))
    if problems:
        raise ValueError(json.dumps(problems, ensure_ascii=False))
    datasets = {k: E3Dataset(v, processor, config['columns']['text'],
        max_audio_seconds=float(config['training']['max_audio_seconds']),
        max_label_length=int(config['training']['generation_max_length'])) for k, v in rows.items()}
    print('PREFLIGHT ' + json.dumps(report, ensure_ascii=False), flush=True)
    return rows, datasets, report


def smoke(model, processor, datasets, config, weight, device, mode):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    model.to(device).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    eligible = [i for i, r in enumerate(datasets['train'].rows) if r['_correction_reason']=='eligible']
    same = [i for i, r in enumerate(datasets['train'].rows) if r['_correction_reason']=='same']
    results = []
    with tempfile.TemporaryDirectory(prefix='ieum_e3_smoke_') as temp:
        trainer = make_e3_trainer(model, processor, datasets, temp, config['training'], weight,
                                 config['e3']['margin'], mode, use_cpu=device=='cpu')
        # Same real eligible batch, ASR-only, to quantify added Decoder memory.
        baseline_batch=E3Collator(processor)([datasets['train'][i] for i in eligible[:2]])
        baseline_batch={k:v.to(device) for k,v in baseline_batch.items()}
        trainer.correction_weight=0
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda' and trainer.args.bf16):
            baseline_loss=trainer.compute_loss(model,baseline_batch)
        baseline_loss.backward()
        asr_only_peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
        model.zero_grad(set_to_none=True)
        del baseline_loss,baseline_batch
        if device=='cuda':
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        trainer.correction_weight=weight
        for indices in (eligible[:2], same[:2]):
            batch = E3Collator(processor)([datasets['train'][i] for i in indices])
            batch = {k:v.to(device) for k,v in batch.items()}
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=device=='cuda' and trainer.args.bf16):
                total, outputs = trainer.compute_loss(model, batch, return_outputs=True)
            stats = dict(trainer.last_components, indices=indices,
                         texts=[datasets['train'].rows[i]['text'] for i in indices])
            if stats['active']:
                probes = (model.model.encoder.conv1.weight, model.model.decoder.layers[0].fc1.weight)
                gradients = torch.autograd.grad(outputs['correction_loss'], probes, retain_graph=True)
                for name, gradient in zip(('correction_encoder_gradient_l2', 'correction_decoder_gradient_l2'), gradients):
                    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
                    stats[name] = float(gradient.norm())
            total.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if not stats['eligible']:
                assert stats['correction_loss']==0
            results.append(stats)
            del total, outputs
        model.zero_grad(set_to_none=True)
        # Compare lambda=0's loss and accumulated gradients against actual E1.
        batch = E3Collator(processor)([datasets['train'][i] for i in eligible[:2]])
        batch = {k:v.to(device) for k,v in batch.items()}
        trainer.correction_weight=0
        trainer.current_gradient_accumulation_steps=4
        if device=='cpu':
            import copy
            from transformers import Seq2SeqTrainer
            e1=Seq2SeqTrainer(model=model,args=copy.deepcopy(trainer.args),processing_class=processor,eval_dataset=[])
        else:
            e1 = make_trainer(model, processor, datasets['train'], datasets['val'], Path(temp)/'e1', config['training'])
        e1.current_gradient_accumulation_steps=4
        if mode=='mean':
            e1.model_accepts_loss_kwargs=False
        nitems=(batch['labels']!=-100).sum()*4
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda' and trainer.args.bf16):
            zero_loss=trainer.training_step(model,batch,num_items_in_batch=nitems)
        g0=model.model.encoder.conv1.weight.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        clean={k:v for k,v in batch.items() if not k.startswith('correction_')}
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda' and trainer.args.bf16):
            base_loss=e1.training_step(model,clean,num_items_in_batch=nitems)
        g1=model.model.encoder.conv1.weight.grad.detach().clone()
        assert torch.allclose(zero_loss,base_loss,atol=1e-5,rtol=1e-5)
        relative_error=float((g0-g1).norm()/g1.norm().clamp_min(1e-12))
        print('ZERO_COMPARISON '+json.dumps(dict(e3_loss=float(zero_loss),e1_loss=float(base_loss),
            e3_gradient_norm=float(g0.norm()),e1_gradient_norm=float(g1.norm()),relative_l2_error=relative_error)),flush=True)
        assert relative_error < (.02 if device=='cuda' and trainer.args.bf16 else 1e-5)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda' and trainer.args.bf16):
            repeat_loss=e1.training_step(model,clean,num_items_in_batch=nitems)
        repeat_gradient=model.model.encoder.conv1.weight.grad.detach()
        repeat_error=float((repeat_gradient-g1).norm()/g1.norm().clamp_min(1e-12))
        print('E1_REPEAT '+json.dumps(dict(loss=float(repeat_loss),gradient_relative_l2_error=repeat_error)),flush=True)
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing_disable();model.eval()
        loss,ids,labels=trainer.prediction_step(model,batch,False,max_new_tokens=8)
        assert torch.isfinite(loss) and ids.ndim==2
        result=dict(device=device,accumulation_mode=mode,batches=results,lambda_zero_e1_loss=float(base_loss),
                    lambda_zero_e3_loss=float(zero_loss),lambda_zero_gradient_match=True,
                    lambda_zero_gradient_relative_l2_error=relative_error,
                    e1_repeat_gradient_relative_l2_error=repeat_error,
                    decoder_generation=True,validation_asr_only=True,optimizer_steps=0)
        if device=='cuda':
            result.update(gpu_name=torch.cuda.get_device_name(),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),bf16=trainer.args.bf16,
                          asr_only_peak_allocated_bytes=asr_only_peak,
                          extra_peak_allocated_bytes=torch.cuda.max_memory_allocated()-asr_only_peak)
        print('SMOKE ' + json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def train(model,processor,rows,datasets,report,config,weight,mode,output):
    output.mkdir(parents=True,exist_ok=False)
    (output/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    identity=dict(general_model=str(path(config['paths']['general_model'])),
                  general_weights_sha256=file_hash(path(config['paths']['general_model'])/'model.safetensors'),
                  manifest_sha256=file_hash(path(config['paths']['manifest_csv'])),seed=config['training']['seed'],
                  lambda_correction=weight,margin=config['e3']['margin'],accumulation_mode=mode,
                  score_policy='mean content+EOS log probability; all other special tokens and padding excluded')
    (output/'initialization.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    audit=E3Audit(output)
    trainer=make_e3_trainer(model,processor,datasets,output/'checkpoints',config['training'],weight,
                            config['e3']['margin'],mode,audit=audit)
    try:
        trainer.train()
        if trainer.state.best_model_checkpoint is None:
            raise RuntimeError('No validation-selected checkpoint')
        best=output/'best_model';trainer.save_model(str(best))
        validation=trainer.evaluate()
        trainer.save_state()
        baseline_state=ROOT/'outputs/personalization/E1_v2/KSM_M_45/checkpoints/checkpoint-105/trainer_state.json'
        baseline_history=json.loads(baseline_state.read_text())['log_history']
        baseline_best=min((r for r in baseline_history if 'eval_cer' in r),key=lambda r:r['eval_cer'])
        summary=dict(status='validation_complete',**identity,best_model=str(best.resolve()),
            best_checkpoint=trainer.state.best_model_checkpoint,best_epoch=audit.report.get('best_epoch'),
            best_validation_cer=trainer.state.best_metric,validation_metrics=validation,
            candidate_counts=report['counts'],test_evaluated=False,
            validation_e1_comparison=dict(e1_cer=baseline_best['eval_cer'],e1_wer=baseline_best['eval_wer'],
                delta_cer=validation['eval_cer']-baseline_best['eval_cer'],
                delta_wer=validation['eval_wer']-baseline_best['eval_wer']))
        (output/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        audit.report['status']='validation_complete';audit.save()
        print(json.dumps(summary,indent=2))
    except BaseException as exc:
        audit.report['status']='failed';audit.report['error']=str(exc);audit.save();raise


def test_selected(config,output):
    summary=json.loads((output/'run_summary.json').read_text())
    destination=output/'final_test'
    if destination.exists():
        raise FileExistsError(destination)
    model=WhisperForConditionalGeneration.from_pretrained(summary['best_model'],local_files_only=True)
    processor=WhisperProcessor.from_pretrained(summary['best_model'],local_files_only=True)
    rows,datasets,_=load_data(config,config['e3']['speaker'],processor)
    destination.mkdir(parents=True,exist_ok=False)
    trainer=make_e3_trainer(model,processor,datasets,destination/'trainer',config['training'],0.,
                            config['e3']['margin'],summary['accumulation_mode'])
    prediction=trainer.predict(datasets['test'],metric_key_prefix='test')
    metrics=compute_asr_metrics(prediction.predictions,prediction.label_ids,processor.tokenizer)
    references,hypotheses=decode_predictions(prediction.predictions,prediction.label_ids,processor.tokenizer)
    save_predictions(destination,rows['test'],references,hypotheses,metrics)
    baseline=json.loads((ROOT/'outputs/personalization/E1_v2/KSM_M_45/test_metrics.json').read_text())
    comparison=dict(e3=metrics,e1_historical=baseline,
                    delta_cer=metrics['cer']-baseline['cer'],delta_wer=metrics['wer']-baseline['wer'])
    (destination/'e1_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    print(json.dumps(comparison,indent=2))


def nonnegative(value):
    value=float(value)
    if not math.isfinite(value) or value<0:
        raise argparse.ArgumentTypeError('Expected finite nonnegative value')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/e3.yaml')
    parser.add_argument('--mode',choices=('validate','smoke','train','test'),default='validate')
    parser.add_argument('--lambda-correction',type=nonnegative)
    parser.add_argument('--accumulation-mode',choices=('legacy_e1','mean'))
    parser.add_argument('--smoke-device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args();config=load_config(args.config)
    torch.set_num_threads(4);set_seed(int(config['training']['seed']))
    weight=args.lambda_correction if args.lambda_correction is not None else config['e3']['lambda_correction']
    weight=nonnegative(weight)
    config['e3']['margin']=nonnegative(config['e3']['margin'])
    mode=args.accumulation_mode or config['e3']['accumulation_mode']
    output=path(config['e3']['output_root'])/config['e3']['speaker']/f'{mode}_lambda_{weight}'
    if args.mode=='test':
        test_selected(config,output);return
    if args.mode=='train' and output.exists():
        raise FileExistsError(output)
    if args.mode=='validate':
        processor=WhisperProcessor.from_pretrained('openai/whisper-small',language='korean',task='transcribe',local_files_only=True)
        load_data(config,config['e3']['speaker'],processor);return
    model,processor=build(config)
    rows,datasets,report=load_data(config,config['e3']['speaker'],processor)
    if args.mode=='smoke':
        smoke(model,processor,datasets,config,weight,args.smoke_device,mode)
    else:
        train(model,processor,rows,datasets,report,config,weight,mode,output)


if __name__=='__main__':
    main()
