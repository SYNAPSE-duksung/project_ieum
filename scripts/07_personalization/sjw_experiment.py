"""SJW_M_29 E1/E4-A/E4-B; Train provenance, independent fits, Validation only."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import soundfile as sf
import torch
import yaml
from transformers import WhisperProcessor,set_seed
from transformers.trainer_utils import SaveStrategy
from src.personalization.dataset import load_speaker_splits
from src.personalization.trainer import build_model_and_processor,make_trainer
from src.personalization.e3_data import join_hypotheses
from src.personalization.profile_builder_v2 import calculate_statistics
from src.personalization.speaker_weighted import build_speaker_weights
from src.personalization.e4_weighted import build_weights,WeightedAudit
from e4_train import dataset,make,write,sha

OUT=ROOT/'outputs/personalization/second_speaker/SJW_M_29'
REPORT=ROOT/'reports/sjw_experiment_20261107'


def prepare():
    os.chdir(ROOT)
    cfg=yaml.safe_load((ROOT/'scripts/07_personalization/configs/e1.yaml').read_text(encoding='utf-8'))
    if sha(Path(cfg['paths']['general_model'])/'model.safetensors')!='709258bb503654029241494d3dc26cdd1fcdce611c8f01a0369f79cbd306e705':raise ValueError('General changed')
    processor=WhisperProcessor.from_pretrained('openai/whisper-small',language='korean',task='transcribe',local_files_only=True)
    rows=load_speaker_splits(cfg['paths']['manifest_csv'],'SJW_M_29',cfg['columns'],cfg['split_labels'],cfg['paths']['audio_root'],cfg['recording_id_column'])
    joined,candidates=join_hypotheses(rows['train'],ROOT/'outputs/personalization/error_profiles/predictions.csv',
        ROOT/'outputs/personalization/error_profiles/profiles/SJW_M_29.json',cfg['paths']['general_model'],processor,160)
    # Reconstruct a NEW profile from this exact Personal Train key set only.
    overall,syllables,insertions=calculate_statistics([dict(reference=r['text'].strip(),prediction=h['_general_hypothesis']) for r,h in zip(rows['train'],joined)])
    profile=dict(speaker_id='SJW_M_29',split='profile_train',utterance_count=len(joined),overall_statistics=overall,
        syllable_profiles=syllables,insertion_profile=insertions)
    weights,info=build_speaker_weights(rows['train'],joined,processor,profile)
    return cfg,processor,rows,weights,dict(info,candidate_audit=candidates,profile=profile)


def preflight():
    cfg,processor,rows,weights,info=prepare()
    problems=[];durations={}
    for split,subset in rows.items():
        seconds=[]
        for row in subset:
            wav=sf.info(row['_resolved_audio']);seconds.append(wav.duration)
            if wav.duration>30 or len(processor.tokenizer(row['text'].strip()).input_ids)>160:problems.append(row['audio_member'])
        durations[split]=dict(count=len(subset),seconds=sum(seconds),max_seconds=max(seconds))
    assert not problems,problems
    if info['weighted_syllable_types']==0:
        write(REPORT/'preflight.json',dict(passed=False,reason='No E4-B target syllables',info=info));raise RuntimeError('No E4-B targets; stop before training')
    # The new adapter removes only speaker-specific assumptions. Prove KSM
    # numeric identity rather than changing any completed experiment module.
    k=load_speaker_splits(cfg['paths']['manifest_csv'],'KSM_M_45',cfg['columns'],cfg['split_labels'],cfg['paths']['audio_root'],cfg['recording_id_column'])
    j,_=join_hypotheses(k['train'],ROOT/'outputs/personalization/error_profiles/predictions.csv',ROOT/'outputs/personalization/error_profiles/profiles/KSM_M_45.json',cfg['paths']['general_model'],processor,160)
    p=json.loads((ROOT/'outputs/personalization/error_profiles_v2/A_original/profiles/KSM_M_45.json').read_text(encoding='utf-8'))
    old,oi=build_weights(k['train'],j,processor,p);new,ni=build_speaker_weights(k['train'],j,processor,p)
    assert old==new and oi['syllable_weights']==ni['syllable_weights']
    actual_error=sum(h['_correction_reason']!='same' for h in info['candidate_audit'].get('unused',[])) if False else info['counts']['a_weighted_utterances']
    assert actual_error==sum(max(r['_token_weights'])>1 for r in weights['A'])
    # Reuse the already validated, unmodified weighted Trainer and verify hashes.
    previous=json.loads((ROOT/'reports/e4_experiment_20261107/verification.json').read_text(encoding='utf-8'))
    assert previous['passed'] and previous['fingerprint']['module']==sha(ROOT/'src/personalization/e4_weighted.py')
    write(REPORT/'train_profile.json',info['profile'])
    targets=[dict(syllable=s,weight=w,**info['profile']['syllable_profiles'][s]) for s,w in info['syllable_weights'].items() if w>1]
    write(REPORT/'weighted_syllables.json',targets)
    result=dict(passed=True,splits=durations,weight_info=info,ksm_exact_weight_parity=True,
        prior_weighted_trainer_verification_reused=True,script_sha256=sha(__file__),adapter_sha256=sha(ROOT/'src/personalization/speaker_weighted.py'),
        free_gib=shutil.disk_usage(ROOT).free/1024**3,strong_filter_applied_to_training=False)
    write(REPORT/'preflight.json',result)
    print(json.dumps(dict(splits=durations,weight_counts=info['counts'],weighted_syllable_types=info['weighted_syllable_types'],passed=True),indent=2))


def train(experiment):
    check=json.loads((REPORT/'preflight.json').read_text(encoding='utf-8'))
    assert check['passed'] and check['script_sha256']==sha(__file__) and check['adapter_sha256']==sha(ROOT/'src/personalization/speaker_weighted.py')
    cfg,processor,rows,weights,info=prepare()
    if shutil.disk_usage(ROOT).free<10*1024**3 or not torch.cuda.is_available():raise RuntimeError('Insufficient disk/CUDA')
    directory=OUT/experiment
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True)
    audit=WeightedAudit(directory)
    write(directory/'status.json',dict(status='training',test_evaluated=False))
    try:
        set_seed(42);model,processor=build_model_and_processor(cfg['paths']['general_model'])
        train_data=dataset(rows['train'] if experiment=='E1' else weights[experiment[-1]],processor)
        val=dataset(rows['val'],processor)
        if experiment=='E1':
            t=make_trainer(model,processor,train_data,val,directory/'checkpoints',cfg['training'])
            assert t.model_accepts_loss_kwargs
            t.args.save_strategy=SaveStrategy.BEST;t.args.save_total_limit=1;t.args.save_only_model=True
            t.add_callback(audit)
            def observe(module,args,kwargs,output):
                if module.training and kwargs.get('labels') is not None:
                    if not torch.isfinite(output.loss):raise FloatingPointError('Nonfinite ASR loss')
                    value=float(output.loss.detach());audit.add(value,value,kwargs['labels'].shape[0])
            model.register_forward_hook(observe,with_kwargs=True)
        else:t=make(model,processor,train_data,val,directory/'checkpoints',cfg,audit)
        write(directory/'provenance.json',info);write(directory/'training_arguments.json',t.args.to_dict())
        t.train();processor.save_pretrained(t.state.best_model_checkpoint)
        t.state.save_to_json(str(directory/'trainer_state.json'))
        best=min((r for r in audit.report['epochs'] if 'validation' in r),key=lambda r:r['validation']['eval_cer'])
        result=dict(speaker='SJW_M_29',experiment=experiment,best_epoch=best['epoch'],best_validation_cer=best['validation']['eval_cer'],
            best_validation_wer=best['validation']['eval_wer'],best_checkpoint=t.state.best_model_checkpoint,
            training_seconds=audit.report['training_seconds'],test_evaluated=False)
        write(directory/'summary.json',result);write(directory/'status.json',dict(status='completed',test_evaluated=False));print(json.dumps(result,indent=2))
    except BaseException as exc:
        write(directory/'status.json',dict(status='failed',error=repr(exc),test_evaluated=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['preflight','train'],required=True);p.add_argument('--experiment',choices=['E1','E4-A','E4-B'])
    args=p.parse_args()
    if args.mode=='preflight':preflight()
    elif args.experiment:train(args.experiment)
    else:p.error('--experiment required')
