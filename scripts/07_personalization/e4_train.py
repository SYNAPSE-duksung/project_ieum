"""E4-A/B KSM validation experiments; no Personal Test evaluation."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import yaml
from transformers import (set_seed, WhisperConfig, WhisperForConditionalGeneration,
    WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from transformers.trainer_utils import SaveStrategy
from src.personalization.trainer import build_model_and_processor,make_trainer
from src.personalization.dataset import load_speaker_splits,WhisperSpeechDataset
from src.personalization.e3_data import join_hypotheses
from src.personalization.e4_weighted import build_weights,WeightedDataset,WeightedCollator,WeightedTrainer,WeightedAudit

OUT=ROOT/'outputs/personalization/E4/KSM_M_45'
REPORT=ROOT/'reports/e4_experiment_20261107'


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    os.chdir(ROOT)
    cfg=yaml.safe_load((ROOT/'scripts/07_personalization/configs/e1.yaml').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'reports/e1_e3_multispeaker_pilot_20261107/selection_fingerprint.json').read_text(encoding='utf-8'))
    for name,h in frozen['source_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError(f'Protected legacy source changed: {name}')
    if sha(Path(cfg['paths']['general_model'])/'model.safetensors')!='709258bb503654029241494d3dc26cdd1fcdce611c8f01a0369f79cbd306e705':
        raise ValueError('General checkpoint changed')
    processor=WhisperProcessor.from_pretrained('openai/whisper-small',language='korean',task='transcribe',local_files_only=True)
    rows=load_speaker_splits(cfg['paths']['manifest_csv'],'KSM_M_45',cfg['columns'],cfg['split_labels'],cfg['paths']['audio_root'],cfg['recording_id_column'])
    assert {k:len(v) for k,v in rows.items()}==dict(train=168,val=65,test=67)
    joined,candidates=join_hypotheses(rows['train'],ROOT/'outputs/personalization/error_profiles/predictions.csv',
        ROOT/'outputs/personalization/error_profiles/profiles/KSM_M_45.json',cfg['paths']['general_model'],processor,160)
    profile=json.loads((ROOT/'outputs/personalization/error_profiles_v2/A_original/profiles/KSM_M_45.json').read_text(encoding='utf-8'))
    weighted,info=build_weights(rows['train'],joined,processor,profile)
    info.update(candidate_audit=candidates,config=cfg,model_sha256=sha(Path(cfg['paths']['general_model'])/'model.safetensors'))
    return cfg,processor,rows,weighted,info


def dataset(rows,processor):
    cls=WeightedDataset if '_token_weights' in rows[0] else WhisperSpeechDataset
    return cls(rows,processor,'text',max_audio_seconds=30,max_label_length=160)


def make(model,processor,train,val,directory,cfg,audit=None):
    template=make_trainer(model,processor,train,val,directory,cfg['training'])
    if not template.model_accepts_loss_kwargs:raise ValueError('E1 legacy scaling changed')
    args=template.args
    args.save_strategy=SaveStrategy.BEST;args.save_total_limit=1;args.save_only_model=True
    return WeightedTrainer(model=model,args=args,train_dataset=train,eval_dataset=val,
        data_collator=WeightedCollator(processor),processing_class=processor,
        compute_metrics=template.compute_metrics,audit=audit,callbacks=[audit] if audit else None)


def verification():
    write(REPORT/'verification.json',dict(passed=False,status='verification_running'))
    cfg,processor,rows,weighted,info=prepare()
    torch.set_num_threads(1);set_seed(42)
    ds=dataset(weighted['B'],processor)
    # Representative actual labels/weights and audio features, cropped only for
    # the tiny deterministic regression model. Full model smoke is separate.
    batch=WeightedCollator(processor)([ds[0],ds[1]])
    batch['input_features']=batch['input_features'][:,:,:16]
    batch['attention_mask']=batch['attention_mask'][:,:16]
    batch['token_weights']=torch.ones_like(batch['token_weights'])
    tiny=WhisperConfig(d_model=16,encoder_layers=1,decoder_layers=1,encoder_attention_heads=2,decoder_attention_heads=2,
        encoder_ffn_dim=32,decoder_ffn_dim=32,max_source_positions=8,max_target_positions=160,vocab_size=51865,
        pad_token_id=50257,bos_token_id=50257,eos_token_id=50257,decoder_start_token_id=50258,apply_spec_augment=False)
    initial=WhisperForConditionalGeneration(tiny)
    results=[];grads=None;updated=None
    for variant in ('E1','E4'):
        model=copy.deepcopy(initial)
        args=Seq2SeqTrainingArguments(output_dir=str(REPORT/'verification_unused'),use_cpu=True,report_to='none',
            learning_rate=1e-5,gradient_accumulation_steps=4,optim='adamw_torch',lr_scheduler_type='linear',
            weight_decay=0,max_grad_norm=1,seed=42,data_seed=None)
        cls=Seq2SeqTrainer if variant=='E1' else WeightedTrainer
        t=cls(model=model,args=args,processing_class=processor)
        assert t.model_accepts_loss_kwargs
        t.current_gradient_accumulation_steps=4
        t.create_optimizer_and_scheduler(1);model.train();losses=[]
        inp={k:v for k,v in batch.items() if variant!='E1' or k!='token_weights'}
        for _ in range(4):losses.append(float(t.training_step(model,inp)))
        current={n:v.grad.clone() for n,v in model.named_parameters() if v.grad is not None}
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        t.optimizer.step();t.lr_scheduler.step()
        params={n:v.detach().clone() for n,v in model.named_parameters()}
        result=dict(variant=variant,losses=losses,preclip_gradient_l2=norm,lr_after_step=t.lr_scheduler.get_last_lr(),
            model_accepts_loss_kwargs=t.model_accepts_loss_kwargs,accelerate_accumulation=t.accelerator.gradient_accumulation_steps)
        if grads is None:grads=current;updated=params
        else:
            result.update(max_gradient_difference=max(float((current[n]-v).abs().max()) for n,v in grads.items()),
                max_parameter_difference=max(float((params[n]-v).abs().max()) for n,v in updated.items()))
            assert max(abs(a-b) for a,b in zip(losses,results[0]['losses']))<1e-6
            assert result['max_gradient_difference']<1e-6 and result['max_parameter_difference']<1e-7
        results.append(result)
    del initial,model,t
    # Actual General model, full 30s feature padding, configured GPU BF16.
    model,processor=build_model_and_processor(cfg['paths']['general_model']);model.to('cuda').train()
    model.gradient_checkpointing_enable()
    probes=[]
    for variant in ('A','B'):
        source=weighted[variant]
        indices=[i for i,r in enumerate(source) if max(r['_token_weights'])>1][:2]
        ds=dataset(source,processor);collator=WeightedCollator(processor)
        full={k:v.to('cuda') for k,v in collator([ds[i] for i in indices]).items()}
        t=make(model,processor,ds,dataset(rows['val'],processor),REPORT/'verification_unused',cfg)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=t.compute_loss(model,full)
        loss.backward()
        norm=float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None]))))
        assert torch.isfinite(loss) and torch.isfinite(torch.tensor(norm))
        probes.append(dict(variant=variant,loss=float(loss.detach()),gradient_l2=norm,batch_indices=indices))
        model.zero_grad(set_to_none=True)
    import gc
    del model,t,loss;gc.collect();torch.cuda.empty_cache()
    # Confirm the actual General model as well, including the optimizer update.
    # FP32 isolates implementation equality from BF16 kernel repeat variation.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    baseline_gradients=None;baseline_parameters=None;general_equivalence=[]
    for variant in ('E1','E4'):
        set_seed(42)
        model,processor=build_model_and_processor(cfg['paths']['general_model'])
        ds=dataset(weighted['B'],processor);val=dataset(rows['val'],processor)
        if variant=='E1':
            t=make_trainer(model,processor,ds,val,REPORT/'verification_unused',cfg['training'])
        else:t=make(model,processor,ds,val,REPORT/'verification_unused',cfg)
        t.current_gradient_accumulation_steps=4
        t.create_optimizer_and_scheduler(105)
        model.gradient_checkpointing_enable();model.train()
        full=WeightedCollator(processor)([ds[0],ds[61]])
        full['token_weights']=torch.ones_like(full['token_weights'])
        if variant=='E1':full.pop('token_weights')
        inp=t._prepare_inputs(full)
        loss=t.training_step(model,inp)
        norm=float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None]))))
        result=dict(variant=variant,loss=float(loss),gradient_l2=norm,precision='FP32',batch_indices=[0,61])
        if baseline_gradients is None:
            baseline_gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
        else:
            max_grad=max(float((p.grad.detach().cpu()-baseline_gradients[n]).abs().max()) for n,p in model.named_parameters() if p.grad is not None)
            result['max_gradient_difference']=max_grad
            print('General equivalence partial:',json.dumps(result),flush=True)
            write(REPORT/'general_equivalence_partial.json',dict(results=general_equivalence+[result]))
            assert max_grad<1e-6
            assert abs(result['loss']-general_equivalence[0]['loss'])<1e-6
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        t.optimizer.step();t.lr_scheduler.step()
        if baseline_parameters is None:
            baseline_parameters={n:p.detach().cpu() for n,p in model.named_parameters()}
        else:
            max_update=max(float((p.detach().cpu()-baseline_parameters[n]).abs().max()) for n,p in model.named_parameters())
            result['max_parameter_difference']=max_update
            assert max_update<1e-7
        general_equivalence.append(result)
        del model,t,inp,full,loss;gc.collect();torch.cuda.empty_cache()
    del baseline_parameters,baseline_gradients;gc.collect()
    fingerprint=dict(script=sha(__file__),module=sha(ROOT/'src/personalization/e4_weighted.py'),
        e1_config=sha(ROOT/'scripts/07_personalization/configs/e1.yaml'))
    write(REPORT/'verification.json',dict(passed=True,regression=results,full_general_bf16_probes=probes,general_equivalence=general_equivalence,
        fingerprint=fingerprint,weight_info=info,note='Tiny deterministic 4-microbatch/1-update equality plus full General BF16 backward; no Test'))
    print(json.dumps(dict(passed=True,regression=results,full_general_bf16_probes=probes,general_equivalence=general_equivalence),indent=2))


def train(variant):
    check=json.loads((REPORT/'verification.json').read_text(encoding='utf-8'))
    assert check['passed']
    assert check['fingerprint']==dict(script=sha(__file__),module=sha(ROOT/'src/personalization/e4_weighted.py'),
        e1_config=sha(ROOT/'scripts/07_personalization/configs/e1.yaml'))
    cfg,processor,rows,weighted,info=prepare()
    directory=OUT/f'E4-{variant}'
    if directory.exists():raise FileExistsError(f'Refusing overwrite: {directory}')
    if not torch.cuda.is_available() or shutil.disk_usage(ROOT).free<10*1024**3:
        raise RuntimeError('CUDA or minimum 10GiB free space unavailable')
    directory.mkdir(parents=True)
    write(directory/'status.json',dict(status='starting',test_evaluated=False))
    audit=WeightedAudit(directory)
    try:
        set_seed(42)
        model,processor=build_model_and_processor(cfg['paths']['general_model'])
        t=make(model,processor,dataset(weighted[variant],processor),dataset(rows['val'],processor),directory/'checkpoints',cfg,audit)
        write(directory/'provenance.json',info);write(directory/'training_arguments.json',t.args.to_dict())
        t.train()
        processor.save_pretrained(t.state.best_model_checkpoint)
        t.state.save_to_json(str(directory/'trainer_state.json'))
        best=min((r for r in audit.report['epochs'] if 'validation' in r),key=lambda r:r['validation']['eval_cer'])
        summary=dict(experiment=f'E4-{variant}',best_epoch=best['epoch'],best_validation_cer=best['validation']['eval_cer'],
            best_validation_wer=best['validation']['eval_wer'],best_checkpoint=t.state.best_model_checkpoint,
            training_seconds=audit.report['training_seconds'],weight_info=info['counts'],
            weighted_syllable_types=info['weighted_syllable_types'] if variant=='B' else None,test_evaluated=False)
        write(directory/'summary.json',summary);write(directory/'status.json',dict(status='completed',test_evaluated=False))
        print(json.dumps(summary,indent=2))
    except BaseException as exc:
        write(directory/'status.json',dict(status='failed',error=repr(exc),test_evaluated=False));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['verify','train'],required=True)
    parser.add_argument('--variant',choices=['A','B'])
    a=parser.parse_args()
    if a.mode=='verify':verification()
    else:
        if not a.variant:parser.error('--variant required for train')
        train(a.variant)


if __name__=='__main__':main()
