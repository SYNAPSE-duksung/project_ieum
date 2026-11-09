"""Verify and run independent E2 controls; no Personal Test model/data use."""
import argparse,gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
if 'verify' in sys.argv:os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import shutil
import torch,yaml,transformers,accelerate
from safetensors import safe_open
from torch.utils.data import DataLoader
from transformers import set_seed
from transformers.trainer_utils import IntervalStrategy,SaveStrategy
from src.personalization.dataset import load_speaker_splits,WhisperSpeechDataset
from src.personalization.trainer import build_model_and_processor,make_trainer
from src.personalization.e2_trainer import build_e2_model_and_processor
from src.personalization.ctc_utils import preflight_splits,encode_ctc,decode_ctc,minimum_ctc_frames,ID_TO_TOKEN,TOKEN_TO_ID,VOCAB_SIZE
from src.personalization.e2_controlled import (IdentifiedDataset,IdentifiedCollator,ControlAudit,
    make_control_trainer,ctc_diagnostics,summarize_ctc,gradient_norm)

CONFIG=Path(__file__).parent/'configs/e2_controlled.yaml'
CONTROL=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
CFG=yaml.safe_load((CONFIG.parent/CONTROL['e1_config']).read_text(encoding='utf-8'))
OUT=ROOT/CONTROL['output_root'];VERIFY=OUT/CONTROL.get('verification_dir','verification')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix('.tmp');t.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(path)
def digest_tensors(items):
    h=hashlib.sha256()
    for name,v in items:h.update(name.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def code_identity():
    paths=[Path(__file__),CONFIG,ROOT/'src/personalization/e2_controlled.py',ROOT/'src/personalization/ctc_head.py',ROOT/'src/personalization/e2_trainer.py',ROOT/'src/personalization/ctc_utils.py',CONFIG.parent/CONTROL['e1_config']]
    return {str(p):sha(p) for p in paths}
def environment():
    free,total=torch.cuda.mem_get_info() if torch.cuda.is_available() else (0,0)
    return dict(cuda=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        vram_free_bytes=free,vram_total_bytes=total,bf16=torch.cuda.is_bf16_supported(),disk_free_gib=shutil.disk_usage(ROOT).free/1024**3,
        versions=dict(torch=torch.__version__,transformers=transformers.__version__,accelerate=accelerate.__version__),
        git_status=subprocess.run(['git','status','--short'],capture_output=True,text=True,encoding='utf-8').stdout,
        git_head=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip())
def rows():
    all_rows=load_speaker_splits(CFG['paths']['manifest_csv'],CONTROL['speaker'],CFG['columns'],CFG['split_labels'],CFG['paths']['audio_root'],CFG['recording_id_column'])
    # Test is not encoded, loaded as audio, inferred, or used for selection.
    selected={k:all_rows[k] for k in ('train','val')}
    assert {k:len(v) for k,v in selected.items()}==CONTROL['expected_splits']
    return selected
def datasets(selected,processor,model):
    return {k:IdentifiedDataset(v,processor,CFG['columns']['text'],encoder=model.model.encoder,max_audio_seconds=30,max_label_length=160) for k,v in selected.items()}
def build(weight):return build_e2_model_and_processor(CFG['paths']['general_model'],weight)

def diagnose(model,processor,ds):
    model.eval();device=next(model.parameters()).device;result=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for batch in DataLoader(ds,batch_size=2,collate_fn=IdentifiedCollator(processor)):
            ids=batch.pop('_sample_ids');batch={k:v.to(device) for k,v in batch.items()}
            hidden=model.model.encoder(input_features=batch['input_features'],attention_mask=batch['attention_mask'],return_dict=True).last_hidden_state
            logits=model.ctc_head(hidden[:,:int(batch['ctc_input_lengths'].max())])
            result.extend(ctc_diagnostics(logits,batch['ctc_input_lengths'],ids))
    return result

def verify():
    if (VERIFY/'verification.json').exists():raise FileExistsError('Verification already exists')
    env=environment();print(json.dumps(env,indent=2),flush=True)
    if not env['cuda'] or not env['bf16'] or env['disk_free_gib']<10:raise RuntimeError('CUDA BF16 and >=10GiB free required')
    VERIFY.mkdir(parents=True,exist_ok=True);write(VERIFY/'environment.json',env)
    set_seed(42);model,processor=build(.03);selected=rows()
    preflight=preflight_splits(selected,processor,model.model.encoder,'text',CFG['training'])
    assert VOCAB_SIZE==11183==len(ID_TO_TOKEN)==len(TOKEN_TO_ID)
    assert len(set(TOKEN_TO_ID.values()))==11183 and TOKEN_TO_ID['0']==11173 and TOKEN_TO_ID['9']==11182
    assert decode_ctc(encode_ctc('700'))=='700' and minimum_ctc_frames(encode_ctc('700'))==4
    general=Path(CFG['paths']['general_model'])/'model.safetensors'
    assert sha(general)=='709258bb503654029241494d3dc26cdd1fcdce611c8f01a0369f79cbd306e705'
    with safe_open(general,framework='pt',device='cpu') as source:
        state=model.state_dict();assert all(torch.equal(source.get_tensor(k),state[k]) for k in source.keys())
    ds=datasets(selected,processor,model)
    # Greedy collapse order: merge repeated frame IDs, then remove blank.
    a=TOKEN_TO_ID['7'];b=TOKEN_TO_ID['0']
    fake=torch.full((1,5,11183),-100.);fake[0,range(5),[a,b,0,b,b]]=100.
    assert ctc_diagnostics(fake,torch.tensor([5]),['synthetic'])[0]['greedy']=='700'
    del fake
    model.to('cuda').train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    batch=IdentifiedCollator(processor)([ds['train'][140],ds['train'][143]]);batch.pop('_sample_ids');batch={k:v.to('cuda') for k,v in batch.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):o=model(**batch)
    enc=model.model.encoder.conv1.weight;head=model.ctc_head.weight;decoder=model.model.decoder.layers[0].fc1.weight
    g_asr=torch.autograd.grad(o.asr_loss,enc,retain_graph=True)[0]
    g_ctc=torch.autograd.grad(o.ctc_loss,[enc,head,decoder],retain_graph=True,allow_unused=True)
    assert g_ctc[2] is None and all(torch.isfinite(x).all() and x.norm()>0 for x in [g_asr,*g_ctc[:2]])
    probe=dict(asr_loss=float(o.asr_loss.detach()),ctc_loss=float(o.ctc_loss.detach()),asr_encoder_norm=float(g_asr.norm()),
        ctc_encoder_norm=float(g_ctc[0].norm()),ctc_head_norm=float(g_ctc[1].norm()),ctc_decoder_gradient_none=True)
    del o,g_asr,g_ctc,batch,model,enc,head,decoder,ds;gc.collect();torch.cuda.empty_cache()
    # Actual full-General E1 vs R2 one-optimizer-step training path. Deterministic
    # Configured BF16 with deterministic controls; not a full run.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    baseline=None;smokes=[];probe_root=VERIFY/f'smoke-{time.time_ns()}'
    for label in ('E1','R2'):
        set_seed(42);directory=probe_root/label;directory.mkdir(parents=True)
        if label=='E1':
            m,p=build_model_and_processor(CFG['paths']['general_model'])
            dataset=WhisperSpeechDataset(selected['train'][:8],p,'text',max_label_length=160)
            trainer=make_trainer(m,p,dataset,dataset,directory,CFG['training'])
        else:
            m,p=build(0.);d=datasets(dict(train=selected['train'][:8],val=selected['val'][:2]),p,m)
            audit=ControlAudit(directory);trainer=make_control_trainer(m,p,d,directory,CFG['training'],audit,False)
        trainer.args.max_steps=1
        trainer.args.eval_strategy=IntervalStrategy.NO;trainer.args.save_strategy=SaveStrategy.NO;trainer.args.load_best_model_at_end=False
        trace=[];raw_losses=[];backward=[]
        def enc_hook(module,args,kwargs):
            x=args[0] if args else kwargs['input_features'];trace.append(hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest())
        handle=m.model.encoder.register_forward_pre_hook(enc_hook,with_kwargs=True)
        original=trainer.accelerator.backward
        def backward_hook(loss,**kwargs):backward.append(float(loss.detach()));return original(loss,**kwargs)
        trainer.accelerator.backward=backward_hook
        # Capture true forward ASR scalar even when R2 bypasses the CTC wrapper.
        trainer.train();handle.remove()
        current={k:v.detach().cpu().clone() for k,v in m.model.state_dict().items()}
        smoke=dict(label=label,steps=trainer.state.global_step,backward=backward,feature_order=trace,model_accepts_loss_kwargs=trainer.model_accepts_loss_kwargs)
        assert smoke['steps']==1 and len(backward)==4
        if baseline is None:baseline=current
        else:
            diff=max(float((v-baseline[k]).abs().max()) for k,v in current.items());smoke['max_backbone_parameter_difference']=diff
            write(VERIFY/'smoke_partial.json',dict(smokes=smokes+[smoke]))
            assert diff<1e-7 and trace==smokes[0]['feature_order']
            assert max(abs(x-y) for x,y in zip(backward,smokes[0]['backward']))<1e-6
        smokes.append(smoke)
        del trainer,m,current,original;gc.collect();torch.cuda.empty_cache()
    del baseline
    old=Path('outputs/personalization/E2/KSM_M_45/lambda_0.03/checkpoints/checkpoint-21')
    from src.personalization.ctc_head import WhisperWithAuxiliaryCTC
    old_model=WhisperWithAuxiliaryCTC.from_pretrained(old,local_files_only=True).to('cuda')
    old_ds=datasets(selected,processor,old_model)['val'];old_rows=diagnose(old_model,processor,old_ds)
    write(VERIFY/'original_e2_ctc_validation.json',dict(summary=summarize_ctc(old_rows),utterances=old_rows))
    del old_model,old_ds;gc.collect();torch.cuda.empty_cache()
    write(VERIFY/'verification.json',dict(passed=True,identity=code_identity(),preflight=preflight,gradient_probe=probe,smokes=smokes,
        ctc_implementation_error_detected=False,scope='Train168/Validation65 only; actual full-model forward/backward and one-step E1/R2 parity; no full control training yet'))
    print('VERIFICATION PASSED',json.dumps(dict(probe=probe,smokes=smokes,original_ctc=summarize_ctc(old_rows)),indent=2),flush=True)

def train(name):
    check=json.loads((VERIFY/'verification.json').read_text(encoding='utf-8'));assert check['passed'] and check['identity']==code_identity()
    config=CONTROL['experiments'][name];directory=OUT/name
    if directory.exists():raise FileExistsError(directory)
    env=environment()
    if not env['cuda'] or env['disk_free_gib']<10:raise RuntimeError('GPU/storage not ready')
    directory.mkdir(parents=True);audit=ControlAudit(directory)
    write(directory/'status.json',dict(status='starting',test_evaluated=False));write(directory/'environment.json',env)
    try:
        set_seed(42);model,processor=build(config['lambda_ctc']);selected=rows();ds=datasets(selected,processor,model)
        identity=dict(backbone=digest_tensors(model.model.state_dict().items()),head=digest_tensors(model.ctc_head.state_dict().items()),
            general_sha256=sha(Path(CFG['paths']['general_model'])/'model.safetensors'),code_hashes=code_identity(),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),seed=42,resume=False)
        write(directory/'initialization.json',identity);write(directory/'configuration.json',dict(e1=CFG,controlled=CONTROL,experiment=config))
        trainer=make_control_trainer(model,processor,ds,directory/'checkpoints',CFG['training'],audit,config['ctc_enabled'])
        write(directory/'training_arguments.json',trainer.args.to_dict())
        if config['ctc_enabled']:
            initial=diagnose(model,processor,ds['val']);write(directory/'initial_ctc.json',dict(summary=summarize_ctc(initial),utterances=initial))
        set_seed(42);trainer.train()
        audit.report['scheduler_last_epoch']=trainer.lr_scheduler.last_epoch;audit.save()
        assert trainer.state.global_step==audit.optimizer_events==audit.report['scheduler_steps']==105
        best=min((r for r in audit.report['epochs'] if 'validation' in r),key=lambda r:r['validation']['eval_cer'])
        processor.save_pretrained(trainer.state.best_model_checkpoint);trainer.state.save_to_json(str(directory/'trainer_state.json'))
        write(directory/'summary.json',dict(experiment=name,lambda_ctc=config['lambda_ctc'],best_epoch=best['epoch'],best_checkpoint=trainer.state.best_model_checkpoint,
            best_validation_cer=best['validation']['eval_cer'],best_validation_wer=best['validation']['eval_wer'],best_ctc=best['ctc'],
            training_seconds=audit.report['training_seconds'],test_evaluated=False))
        write(directory/'status.json',dict(status='completed',test_evaluated=False))
    except BaseException as exc:
        write(directory/'status.json',dict(status='failed',error=repr(exc),test_evaluated=False));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['verify','train'],required=True);p.add_argument('--experiment',choices=['R1','R2'])
    args=p.parse_args()
    if args.mode=='verify':verify()
    elif args.experiment:train(args.experiment)
    else:p.error('--experiment required')
