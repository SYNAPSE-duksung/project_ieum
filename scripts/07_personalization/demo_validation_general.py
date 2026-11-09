"""Inference only: compare fixed KSM Validation with saved E1-equivalent outputs."""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import sys, json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
import soundfile as sf
from transformers import set_seed
from src.personalization.dataset import load_speaker_splits, WhisperSpeechDataset
from src.personalization.trainer import build_model_and_processor, make_trainer
from src.personalization.evaluator import decode_predictions, metric_counts, normalize_text

def main():
    dest = ROOT / 'outputs/personalization/evaluation/KSM_M_45/demo_validation_20261109'
    if dest.exists():
        raise FileExistsError(dest)
    cfg = yaml.safe_load((ROOT/'scripts/07_personalization/configs/e1.yaml').read_text(encoding='utf-8'))
    splits = load_speaker_splits(cfg['paths']['manifest_csv'], 'KSM_M_45', cfg['columns'], cfg['split_labels'], cfg['paths']['audio_root'], cfg['recording_id_column'])
    assert len(splits['val']) == 65
    assert not {r['_resolved_audio'] for r in splits['val']} & {r['_resolved_audio'] for r in splits['train']}
    saved = ROOT/'outputs/personalization/E2_controlled/KSM_M_45/R2/best_validation_evaluation/predictions.jsonl'
    e1 = {r['audio_member']:r for r in map(json.loads,saved.read_text(encoding='utf-8').splitlines())}
    dest.mkdir(parents=True)
    set_seed(42)
    model, processor = build_model_and_processor(cfg['paths']['general_model'], cfg['language'], cfg['task'])
    ds = WhisperSpeechDataset(splits['val'], processor, cfg['columns']['text'], max_label_length=160)
    trainer = make_trainer(model, processor, ds, ds, dest/'runtime', cfg['training'])
    pred = trainer.predict(ds)
    refs, hyps = decode_predictions(pred.predictions, pred.label_ids, processor.tokenizer)
    out=[]
    for row,ref,hyp in zip(splits['val'],refs,hyps):
        old=e1[row['audio_member']]
        assert old['reference']==ref==normalize_text(row['text'])
        ep=old['prediction']; g=metric_counts(ref,hyp); e=metric_counts(ref,ep)
        txt=Path(cfg['paths']['audio_root'])/row['batch']/'transcripts'/ (Path(row['audio_member']).stem+'.txt')
        transcript=txt.read_text(encoding='utf-8-sig').strip() if txt.exists() else None
        out.append(dict(id=Path(row['audio_member']).stem,split='personal_val',wav_path=row['_resolved_audio'],reference=ref,general_prediction=hyp,e1_prediction=ep,general_cer=g[0]/g[1],e1_cer=e[0]/e[1],error_reduction=g[0]-e[0],seconds=sf.info(row['_resolved_audio']).duration,processed_transcript=transcript,processed_transcript_matches=normalize_text(transcript)==ref if transcript is not None else None,e1_source=str(saved),listening_verified=False))
    with (dest/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    (dest/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    (dest/'provenance.json').write_text(json.dumps(dict(general_checkpoint=cfg['paths']['general_model'],manifest=cfg['paths']['manifest_csv'],generation=trainer.args.to_dict(),e1_source=str(saved),training_executed=False,original_audio_listening_verified=False),ensure_ascii=False,indent=2),encoding='utf-8')
    for r in out:
        if r['error_reduction']>0 or r['id'].endswith(('003238','003037','003048','003076','002976')):
            print(json.dumps(r,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
