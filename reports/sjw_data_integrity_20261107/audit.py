"""Read-only provenance audit of exactly 190 Train + 63 Validation utterances."""
import ast,csv,hashlib,json,re,sys,zipfile
from pathlib import Path
from collections import Counter,defaultdict
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.personalization.evaluator import normalize_text
DATA=ROOT.parent/'ieum_datasets'/'BATCH_026'
RUN=Path('D:/ieum_local_pilot/batch_runs/BATCH_026')
RAW_EXPORT=ROOT.parent/'dataset-restoration/selected_metadata_split.csv'
BAD='ID-03-34-N-SJW-04-M-29-SU'

def csvread(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    manifest=ROOT/'data/splits/personalization_split.csv'
    rows=[r for r in csvread(manifest) if r['speaker_id']=='SJW_M_29' and r['personal_split'] in ('profile_train','personal_val')]
    assert Counter(r['personal_split'] for r in rows)==Counter(profile_train=190,personal_val=63)
    rids={r['recording_id'] for r in rows}
    sources=defaultdict(list)
    for r in csvread(RAW_EXPORT):
        if r['filename'] in rids:sources[r['filename']].append(r)
    assert all(len(sources[r])==1 for r in rids)
    metadata=csvread(DATA/'metadata.csv');byname=defaultdict(list)
    for r in metadata:byname[Path(r['wav_path']).name].append(r)
    core=ROOT/'scripts/03_preprocessing_pilot/05_run_preprocessing_pilot15.py'
    nodes={n.name:n for n in ast.parse(core.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    ns={'re':re}
    for name in ('split_transcript','clean_transcript'):
        exec(compile(ast.Module(body=[nodes[name]],type_ignores=[]),str(core),'exec'),ns)
    binding=load(RUN/'batch_binding.json')
    assert all(hashlib.sha256(ast.dump(nodes[n],include_attributes=False).encode()).hexdigest()==binding['core_function_ast_sha256'][n]
               for n in ('split_transcript','clean_transcript','save_pass','vad_segment_by_energy','process_recording'))
    inputs={str(manifest):sha(manifest),str(DATA/'metadata.csv'):sha(DATA/'metadata.csv'),str(RAW_EXPORT):sha(RAW_EXPORT),str(core):sha(core)}
    archive_path=Path('D:/ieum_local_pilot/archives/BATCH_026.zip');archive=zipfile.ZipFile(archive_path)
    inputs[str(archive_path)]=sha(archive_path)
    group_info={};segmaps={};texts={};all_segments={};result_files={}
    for rid in sorted(rids):
        source=sources[rid][0];raw=ns['split_transcript'](source['Transcript']);texts[rid]=[ns['clean_transcript'](t) for t in raw]
        results=list((RUN/'records'/rid).glob('attempt_*/result.json'))
        good=[p for p in results if load(p)['result']['final_status']=='SUCCESS']
        assert len(good)==1
        result=load(good[0]);result_files[rid]=result
        folder=Path(result['core_output'])/rid/('vad'+str(result['result']['final_pass']))
        segments=csvread(folder/'segments.csv');all_segments[rid]=segments
        segmaps[rid]={r['wav_name']:r for r in segments};assert len(segmaps[rid])==len(segments)
        indices=[int(r['segment_index']) for r in segments]
        gaps=sorted(set(range(len(segments)))-set(indices));overlaps=sum(float(a['end_ms'])>float(b['start_ms']) for a,b in zip(segments,segments[1:]))
        # Recorded raw paths are historical; exported JSON fields remain available.
        group_info[rid]=dict(recording_id=rid,selected_train=sum(r['personal_split']=='profile_train' and r['recording_id']==rid for r in rows),
            selected_val=sum(r['personal_split']=='personal_val' and r['recording_id']==rid for r in rows),
            source_file_id=source['File_id'],source_json_path=source['json_path'],source_wav_path=source['wav_path'],
            source_evidence='exported original JSON fields; raw WAV/JSON files not available at supplied local root',
            original_duration_sec=float(source['duration_sec']),json_start_pos=source.get('Meta_info.StartPos',''),json_end_pos=source.get('Meta_info.EndPos',''),
            json_sentence_count=len(raw),vad_segment_count=len(segments),segment_index_gaps=gaps,overlapping_segments=overlaps,
            pairing_status=sorted({r['pairing_status'] for r in segments}),raw_json_has_per_utterance_times=False,
            result_json=str(good[0]),segments_csv=str(folder/'segments.csv'))
        inputs[str(good[0])]=sha(good[0]);inputs[str(folder/'segments.csv')]=sha(folder/'segments.csv')
    trainpred={r['audio_member']:r for r in csvread(ROOT/'outputs/personalization/error_profiles/predictions.csv') if r['speaker_id']=='SJW_M_29'}
    valpred={r['audio_member']:r for r in [json.loads(t) for t in (ROOT/'reports/sjw_experiment_20261107/E1_validation_predictions.jsonl').read_text(encoding='utf-8').splitlines()]}
    audit=[];hashes=defaultdict(list);indices_by_group=defaultdict(list)
    selected_indices=defaultdict(set)
    for r in rows:
        m=byname[Path(r['audio_member']).name][0]
        selected_indices[r['recording_id']].add(int(segmaps[r['recording_id']][Path(m['source_wav_path']).name]['segment_index']))
    for row in rows:
        rid=row['recording_id'];wav=DATA/row['audio_member'];txt=DATA/row['transcript_member']
        choices=byname[wav.name];assert len(choices)==1;meta=choices[0]
        srcwav=Path(meta['source_wav_path']);srctxt=Path(meta['source_transcript_path'])
        finalwav=Path(meta['wav_path']);finaltxt=Path(meta['transcript_path'])
        seg=segmaps[rid][srcwav.name];idx=int(seg['segment_index']);indices_by_group[rid].append(idx)
        nominal=texts[rid][idx];current=row['text'];flags=[]
        for label,p in [('current_wav',wav),('current_txt',txt),('core_wav',srcwav),('core_txt',srctxt)]:
            if not p.is_file():flags.append('missing_'+label)
        audio_hash=sha(wav);core_hash=sha(srcwav)
        copy_hash=sha(finalwav) if finalwav.is_file() else hashlib.sha256(archive.read(row['audio_member'])).hexdigest()
        hashes[audio_hash].append(row['audio_member'])
        if not audio_hash==core_hash==copy_hash:flags.append('wav_copy_hash_mismatch')
        text_hash=sha(txt)
        copy_txt_hash=sha(finaltxt) if finaltxt.is_file() else hashlib.sha256(archive.read(row['transcript_member'])).hexdigest()
        if not text_hash==sha(srctxt)==copy_txt_hash==meta['transcript_sha256']:flags.append('txt_copy_hash_mismatch')
        if not current.strip()==txt.read_text(encoding='utf-8').strip()==srctxt.read_text(encoding='utf-8').strip()==nominal.strip():flags.append('text_index_mismatch')
        if Path(row['source_wav_path']).name!=srcwav.name:flags.append('manifest_source_basename_mismatch')
        if rid!=meta['recording_id'] or row['speaker_id']!=meta['speaker_id']:flags.append('identity_mismatch')
        info=sf.info(wav);duration=(float(seg['end_ms'])-float(seg['start_ms']))/1000
        if abs(info.duration-duration)>1/info.samplerate:flags.append('duration_interval_mismatch')
        # Text recurrence is recorded, never interpreted as duplicate audio.
        duplicate_text_indices=[j for j,t in enumerate(texts[rid]) if t.strip()==nominal.strip()]
        pred=trainpred.get(row['audio_member'],valpred.get(row['audio_member']))
        hypothesis=pred.get('prediction',pred.get('hypothesis','')) if pred else ''
        compact=lambda text:normalize_text(text).replace(' ','')
        alternatives=[dict(original_sentence_index=j,text=t,index_delta=j-idx,
            status='UNCONFIRMED_MODEL_MATCH_TO_ORIGINAL_EXPORT_NOT_CORRECTED_GOLD')
            for j,t in enumerate(texts[rid]) if j!=idx and j in selected_indices[rid] and hypothesis and compact(t)==compact(hypothesis)]
        priority=rid==BAD or bool(flags)
        reason=';'.join(flags) if flags else ('same_recording_as_confirmed_mapping_problem;COUNT_MATCH_ORDER_ONLY_not_semantic_alignment' if rid==BAD else 'no_structural_anomaly;order_only_pairing_not_semantically_verified')
        audit.append(dict(speaker_id=row['speaker_id'],recording_id=rid,personal_split=row['personal_split'],wav_path=str(wav),current_text=current,
            original_text_candidate=nominal,candidate_basis='original JSON Transcript export sentence at same preprocessing index; NOT verified spoken text',
            suspicion_reason=reason,verification_status='PRIORITY_LISTENING_REQUIRED_CAUSE_UNRESOLVED' if priority else 'CHAIN_MATCHED_SEMANTICS_NOT_VERIFIED',
            auto_recoverable=False,manual_listening_required_for_semantic_confirmation=True,priority_suspect=priority,
            cause_unknown_for_suspect=priority,segment_index=idx,start_ms=seg['start_ms'],end_ms=seg['end_ms'],duration_sec=info.duration,
            pairing_status=seg['pairing_status'],source_json_path=sources[rid][0]['json_path'],source_original_wav_path=sources[rid][0]['wav_path'],
            core_wav_path=str(srcwav),core_txt_path=str(srctxt),segments_csv=group_info[rid]['segments_csv'],
            processed_wav_sha256=audio_hash,core_wav_sha256=core_hash,copy_hash_match=audio_hash==core_hash==copy_hash,
            final_copy_verification='direct D copy' if finalwav.is_file() else str(archive_path)+'::'+row['audio_member'],
            original_sentence_same_as_current=nominal.strip()==current.strip(),structural_flags=';'.join(flags),
            same_text_sentence_indices=json.dumps(duplicate_text_indices),model_hypothesis_reference_only=hypothesis,
            alternate_original_candidates_unverified=json.dumps(alternatives,ensure_ascii=False),
            raw_original_source_access='unavailable; exported metadata used',source_json_individual_timecode_available=False))
        for p in (wav,txt,srcwav,srctxt,finalwav,finaltxt):
            if p.is_file():inputs[str(p)]=sha(p)
    duplicates=[v for v in hashes.values() if len(v)>1]
    for rid in group_info:
        rs=[r for r in audit if r['recording_id']==rid]
        group_info[rid].update(selected_utterances=len(rs),selected_structural_issues=sum(bool(r['structural_flags']) for r in rs),
            selected_nominal_text_matches=sum(r['original_sentence_same_as_current'] for r in rs),
            selected_duplicate_source_indices=len(indices_by_group[rid])-len(set(indices_by_group[rid])))
    table('all_253_trace.csv',audit);table('suspected_utterances.csv',[r for r in audit if r['priority_suspect']]);table('recording_summary.csv',list(group_info.values()))
    selected_status=Counter(r['verification_status'] for r in audit)
    summary=dict(scope=dict(train=190,validation=63,test_utterances_opened=0),recordings=len(group_info),statuses=dict(selected_status),
        structural_issue_utterances=sum(bool(r['structural_flags']) for r in audit),nominal_text_matches=sum(r['original_sentence_same_as_current'] for r in audit),
        copy_wav_hash_matches=sum(r['copy_hash_match'] for r in audit),duplicate_selected_audio_hash_sets=duplicates,
        auto_recoverable_confirmed=0,manual_listening_required_for_definitive_semantics=len(audit),priority_manual_review=sum(r['priority_suspect'] for r in audit),
        cause_unresolved_among_priority_suspects=sum(r['cause_unknown_for_suspect'] for r in audit),
        unflagged_chain_matches=sum(not r['priority_suspect'] for r in audit),
        count_definitions='manual semantics=all selected; priority is subset; unresolved count overlaps priority, not an additional exclusive category',
        raw_source_recordings_directly_verified=0,original_json_export_recordings_verified=5,
        limits='No raw continuous WAV/JSON available at supplied root. JSON export has whole-recording Transcript and recording-level StartPos/EndPos only. Counts/order cannot establish utterance alignment.',
        original_modified=False,training=False,model_inference=False)
    assert all(sha(p)==h for p,h in inputs.items())
    write('input_checksums.json',inputs);write('summary.json',summary);write('original_metadata_provenance.json',group_info)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
