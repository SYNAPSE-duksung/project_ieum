"""Investigate only 11 unresolved slots; replay saved JSON/listing evidence.

No downloads, audio reads, preprocessing, fuzzy matching or existing-file edits.
Six CYZ slots have no confirmed recording IDs: do not invent IDs/select six.
"""
import argparse
import base64
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('C:/ieum')


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',type=Path,default=Path(__file__).parent/'unresolved_investigation_evidence')
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args()
    out=ROOT/'results/full_expansion'
    outputs=['unresolved_investigation.csv','all_recordings_manifest_resolved.csv',
             'candidate_exclude_exact_matches_resolved.csv','candidate_exclude_all_code_overlap_resolved.csv',
             'candidate_resolved_summary.csv','cyz_recording_metadata_audit.csv','unresolved_resolution_summary.json']
    if any((out/name).exists() for name in outputs):
        raise ValueError('Existing target output; refusing overwrite')
    sources=[out/'all_recordings_manifest.csv',out/'manifest_missing_or_unresolved.csv',
             ROOT/'data/metadata/speaker_summary.csv',out/'speaker_expansion_classification.csv',
             ROOT/'scripts/03_preprocessing_pilot/all_metadata_drive.csv',
             args.evidence_dir/'json_evidence_complete.json',args.evidence_dir/'targeted_drive_listing.json']
    hashes={str(p):sha(p) for p in sources}
    original=read(sources[0])
    if len(original)!=7501 or len({r['recording_id'] for r in original})!=7501:
        raise ValueError('Unexpected initial manifest')
    summary={r['speaker_id']:r for r in read(sources[2])}
    classification={r['speaker_id']:r for r in read(sources[3])}
    metadata=read(sources[4])
    evidence=json.loads(sources[5].read_text(encoding='utf-8-sig'))
    json_by_id={}
    for entry in evidence:
        if entry.get('error') or not entry.get('b64'):
            raise ValueError('Incomplete JSON evidence')
        raw=base64.b64decode(entry['b64'])
        json_by_id[entry['id']]=dict(entry,obj=json.loads(raw.decode('utf-8-sig')),sha256=hashlib.sha256(raw).hexdigest())
    listing=json.loads(sources[6].read_text(encoding='utf-8-sig'))
    files={}
    for folder in listing['folders']:
        for f in folder['files']:
            files[f['id']]=dict(f,dataset=folder['dataset'],drive_path=folder['path']+'/'+f['name'])
    # Explicit candidate WAV IDs are grounded in exact physical WAV/JSON
    # basename pairs in the same dataset, and JSON File_id equals expected ID.
    corrections={
      'ID-01-01-N-JJR-03-F-64-KK':('1_R5wZL96SK7DZtPCUYcjK6E6vHIbBTiI','1UlJBrj2-uviIgBvQuymBQ9mNFEWzAIk0',
        'Drive WAV/JSON use disease-code token 11; JSON File_id and Disease_info retain 01; File_id, sex, age and playTime exactly identify expected metadata recording.'),
      'ID-01-15-N-LHS-04-M-72-CC':('1JOT-jpsB4ewql3TzFzHJ_HajIweDORDx','1LWTz50Y_v6g8FIUJ3CoYbd5TPdl66cAA',
        'Drive WAV/JSON filename age token is 99; JSON File_id and Patient_info.Age are 72 and playTime matches expected metadata; no speaker age inferred from filename.'),
      'ID-02-27-N-UCH-02-01-M-55-SU':('1H1-kQDcOju89XSdplsZP8kZ2HVfYWolZ','1mjaoaDIrTxxSYNfRO0Pvvd_KuqsDuBta',
        'WAV basename contains one extra closing parenthesis before .wav; JSON File_id, Patient_info and playTime exactly match expected recording; unique candidate after only that punctuation removal.'),
      'ID-02-27-N-BJJ-01-03-F-36-KK_01-030151부터_중복':('1cKT1qDOPViNdXRiOIH8NguG47VBG3GYe','14puSpVPOWJwFk0PAPc1dg5hs65UaVYlu',
        'Physical WAV/JSON basename renamed to KK2; JSON File_id retains exact long original suffix and playTime matches metadata; base KK and _중복 are distinct and not reused.'),
      'ID-02-27-N-BJJ-01-03-F-36-KK_01-030151부터':('1dpMJrl9_s6fWAHyjdzlEznKr9OsA7vEU','1bIvZTHO9Ac1fP6GiCDoxbwYQFkWZ33PO',
        'Physical WAV/JSON basename renamed to KK1; JSON File_id retains exact long original suffix and playTime matches metadata; no duration-only or name-similarity mapping.')}
    investigation=[]
    corrected=[dict(r) for r in original]
    byrid={r['recording_id']:r for r in corrected}
    for rid,(wid,jid,reason) in corrections.items():
        row=byrid[rid]
        w,j=files[wid],files[jid]
        obj=json_by_id[jid]['obj']
        patient=obj['Patient_info']
        if obj['File_id']!=rid+'.wav' or patient['Sex']!=row['sex'] or int(patient['Age'])!=int(row['age']):
            raise ValueError('Exact JSON identity validation failed: '+rid)
        if not math.isclose(float(obj['playTime']),float(row['duration_sec']),abs_tol=1e-6,rel_tol=0):
            raise ValueError('JSON duration differs: '+rid)
        physical_same=Path(w['name']).stem==Path(j['name']).stem
        punctuation_only=w['name']==rid+').wav' and j['name']==rid+'.json'
        if not (physical_same or punctuation_only):
            raise ValueError('Physical filename pairing not proven')
        if j['dataset']!=row['dataset'] or w['dataset']!={'VL01':'VS01','TL02':'TS02'}[row['dataset']]:
            raise ValueError('Dataset pairing conflict')
        investigation.append(dict(investigation_item_id=rid,speaker_id=row['speaker_id'],expected_recording_id=rid,
                                  dataset=row['dataset'],status='RESOLVABLE_EXACT_METADATA_MATCH',applied_to_manifest=True,
                                  wav_filename=w['name'],json_filename=j['name'],wav_drive_id=wid,json_drive_id=jid,
                                  wav_drive_location=w['drive_path'],json_drive_location=j['drive_path'],
                                  json_File_id=obj['File_id'],speaker_code=row['speaker_code'],sex=patient['Sex'],age=patient['Age'],
                                  duration_sec=obj['playTime'],disease_info=json.dumps(obj['Disease_info'],ensure_ascii=False),
                                  reason=reason,evidence='Exact JSON File_id + Patient_info + playTime; unique physical pairing in verified Drive listing',
                                  json_sha256=json_by_id[jid]['sha256']))
        row['original_expected_wav_filename']=row['wav_filename']
        row['original_expected_json_filename']=row['json_filename']
        row['json_File_id']=obj['File_id']
        for kind,f in [('wav',w),('json',j)]:
            row[kind+'_filename']=f['name']
            row[kind+'_drive_id']=f['id']
            row[kind+'_drive_path']=f['drive_path']
            row[kind+'_size_bytes']=str(int(f['size']))
        # Literal basename metric stays truthful for UCH; pairing_verified
        # includes exact-metadata verified physical-name exceptions.
        row['basename_pairing']=str(physical_same)
        row['json_file_id_verified']='True'
        row['json_file_id_verification']='Exact expected recording ID; physical filename alias verified through JSON'
        row['pairing_verified']='True'
        row['pairing_method']='EXACT_JSON_FILE_ID_AND_METADATA_PHYSICAL_NAME_ALIAS'
    for row in corrected:
        row.setdefault('pairing_verified',row['basename_pairing'])
        row.setdefault('pairing_method','EXACT_LITERAL_BASENAME')
    cyz_summary=summary['CYZ_F_65']
    cyz_metadata=[m for m in metadata if m['speaker']=='CYZ' and m['sex']=='F' and int(float(m['age']))==65]
    cyz_audit=[]
    for md in cyz_metadata:
        matches=[f for f in files.values() if f['dataset']=='VL01' and f['name']==Path(md['file_id']).with_suffix('.json').name]
        wavs=[f for f in files.values() if f['dataset']=='VS01' and f['name']==md['file_id']]
        if len(matches)!=1 or len(wavs)!=1:
            raise ValueError('CYZ exact listing ambiguity')
        j,w=matches[0],wavs[0]
        obj=json_by_id[j['id']]['obj']
        patient=obj['Patient_info']
        if obj['File_id']!=md['file_id'] or not math.isclose(float(obj['playTime']),float(md['play_time']),abs_tol=1e-6,rel_tol=0):
            raise ValueError('CYZ JSON/metadata conflict')
        cyz_audit.append(dict(recording_id=Path(md['file_id']).stem,speaker_id='CYZ_F_65',dataset='VL01',
                             wav_filename=w['name'],json_filename=j['name'],wav_drive_id=w['id'],json_drive_id=j['id'],
                             wav_drive_location=w['drive_path'],json_drive_location=j['drive_path'],json_File_id=obj['File_id'],
                             speaker_code='CYZ',sex=patient['Sex'],age=patient['Age'],region=patient['Area'],
                             duration_sec=obj['playTime'],disease_info=json.dumps(obj['Disease_info'],ensure_ascii=False),
                             json_version=obj['Meta_info'].get('Version'),json_revision_history=obj['Meta_info'].get('RevisionHistory'),
                             json_modified_time=json_by_id[j['id']]['modified_time'],json_sha256=json_by_id[j['id']]['sha256'],
                             matches_summary_region_case_exact=patient['Area']==cyz_summary['region'],
                             selection_status='NOT_SELECTED_METADATA_SCOPE_CONFLICT'))
    partitions={region:dict(recording_count=sum(r['region']==region for r in cyz_audit),
                           duration_sec=sum(r['duration_sec'] for r in cyz_audit if r['region']==region))
                for region in sorted({r['region'] for r in cyz_audit})}
    upper=partitions.get('KK',{})
    summary_reproduced=upper.get('recording_count')==int(cyz_summary['n_files']) and math.isclose(upper.get('duration_sec',0),float(cyz_summary['total_duration_sec']),abs_tol=1e-6)
    reason=('Summary has 6 recordings and 6423.564693877551 sec; exact-case JSON Patient_info.Area=KK subset reproduces both. '
            'All metadata/Drive contain 22 recordings: KK/KK2=6 and kk or kk (5)=16. '
            'All checked JSON Version values are v.0.1 with empty RevisionHistory. '
            'Original speaker_summary generator/selection provenance not located: case-sensitive region selection is a reproducible explanation, not a confirmed historical rule. '
            'No six recordings selected; all 22 retained in evidence audit.')
    for i in range(int(cyz_summary['n_files'])):
        investigation.append(dict(investigation_item_id=f'CYZ_EXPECTED_SLOT_{i+1:02d}',speaker_id='CYZ_F_65',
                                  expected_recording_id='',dataset='VL01',status='METADATA_VERSION_CONFLICT',
                                  applied_to_manifest=False,reason=reason,
                                  evidence='speaker_summary row + 22 all_metadata rows + 22 exact WAV/JSON listing pairs + 22 JSON File_id/Patient_info/playTime',
                                  note='Summary-level expected slot only; actual recording ID not established; slot label is not a fabricated recording ID'))
    if len(investigation)!=11:
        raise ValueError('Expected 11 investigation rows')
    for field in ['recording_id','wav_drive_id','json_drive_id']:
        values=[r[field] for r in corrected]
        if len(values)!=len(set(values)):
            raise ValueError('Duplicate corrected '+field)
    candidates=[('Candidate_1',['B_speaker_code_overlap_only','C_new_no_overlap'],'candidate_exclude_exact_matches_resolved.csv'),
                ('Candidate_2',['C_new_no_overlap'],'candidate_exclude_all_code_overlap_resolved.csv')]
    candidate_rows=[]
    candidate_summary=[]
    for label,cats,name in candidates:
        rs=[r for r in corrected if r['recovered_relation'] in cats]
        expected=[r for r in classification.values() if r['recovered_relation'] in cats]
        wb=sum(int(r['wav_size_bytes']) for r in rs)
        jb=sum(int(r['json_size_bytes']) for r in rs)
        candidate_rows.append((name,rs))
        candidate_summary.append(dict(candidate=label,speaker_count=len({r['speaker_id'] for r in rs}),recording_count=len(rs),
                                      pairing_verified_count=sum(r['pairing_verified']=='True' for r in rs),
                                      wav_bytes=wb,json_bytes=jb,total_bytes=wb+jb,GB=(wb+jb)/1e9,GiB=(wb+jb)/(2**30),
                                      duration_sec=sum(float(r['duration_sec']) for r in rs),
                                      expected_speaker_count=len(expected),expected_recording_count=sum(int(r['n_files']) for r in expected),
                                      remaining_expected_recordings=sum(int(r['n_files']) for r in expected)-len(rs),
                                      coverage='Exact size of corrected manifest only; six CYZ slots remain unresolved; complete candidate size NOT established'))
    result=dict(investigation_count=11,resolved_applied_count=5,remaining_unresolved_expected_slots=6,
                status_counts=dict(Counter(r['status'] for r in investigation)),
                corrected_speaker_count=len({r['speaker_id'] for r in corrected}),corrected_recording_count=len(corrected),
                literal_basename_pairing_count=sum(r['basename_pairing']=='True' for r in corrected),
                exact_metadata_or_basename_pairing_count=sum(r['pairing_verified']=='True' for r in corrected),
                duplicate_recording_id_count=0,cyz_partitions=partitions,
                cyz_summary_reproduced_by_exact_region_case=summary_reproduced,
                cyz_original_selection_rule_confirmed=False,cyz_recordings_added=0,
                candidate_summary=candidate_summary,input_sha256=hashes,
                evidence_observed_at=listing['observed_at'],
                limitations=['No WAV content downloaded/read; physical WAV duration not measured.',
                             'JSON File_id/Patient_info/playTime validate metadata pairing; biological identity not asserted.',
                             'CYZ selection provenance unconfirmed; no recordings selected/discarded.',
                             'Original manifest and original CSV/JSON/WAV files unchanged.'])
    if any(sha(p)!=hashes[str(p)] for p in sources):
        raise ValueError('Inputs changed')
    if not args.inspect:
        for name,rs in [('unresolved_investigation.csv',investigation),('all_recordings_manifest_resolved.csv',corrected),
                        ('candidate_resolved_summary.csv',candidate_summary),('cyz_recording_metadata_audit.csv',cyz_audit)]+candidate_rows:
            write(out/name,rs)
        with (out/'unresolved_resolution_summary.json').open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
