"""Metadata-only expansion preparation; no media download or preprocessing.

Drive snapshot is produced by connector list_folder(top_k=1000), recursively
visiting every returned subfolder below the eight verified dataset roots.
Replay uses that immutable snapshot; no credentials or media requests are used.
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT = Path('C:/ieum')
A = 'A_exact_recovered_match'
B = 'B_speaker_code_overlap_only'
C = 'C_new_no_overlap'
DATASETS = {'TS01':'TL01', 'TS02':'TL02', 'TS03':'TL03', 'VS01':'VL01'}
GROUPS = {'TL01':'뇌신경장애', 'VL01':'뇌신경장애', 'TL02':'언어·청각 장애', 'TL03':'후두장애'}


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def age(value):
    n = Decimal(value)
    if not n.is_finite() or n < 0 or n != int(n):
        raise ValueError(f'Invalid age: {value}')
    return str(int(n))


def canonical(value):
    value = value.strip().upper()
    if value.startswith('PN_'):
        value = value[3:]
    m = re.fullmatch(r'([A-Z0-9]+)[_-]([FM])[_-](\d+(?:\.0+)?)', value)
    if not m:
        raise ValueError(f'Invalid canonical ID: {value}')
    return f'{m[1]}_{m[2]}_{age(m[3])}'


def identity(filename):
    m = re.search(r'-([A-Za-z0-9]+)-(?:\d+-)*([FM])-(\d+)-', filename)
    if not m:
        raise ValueError(f'Unsupported recording filename: {filename}')
    return f'{m[1].upper()}_{m[2].upper()}_{age(m[3])}'


def write(path, rows, fallback):
    keys = list(dict.fromkeys(k for r in rows for k in r)) or fallback
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--snapshot', type=Path, default=Path(__file__).with_name('drive_listing_snapshot.json'))
    p.add_argument('--metadata', type=Path, default=ROOT/'scripts/03_preprocessing_pilot/all_metadata_drive.csv')
    p.add_argument('--output', type=Path, default=ROOT/'results/full_expansion')
    p.add_argument('--inspect', action='store_true', help='Validate/compute in memory; write nothing')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError(f'Existing output; refusing overwrite: {args.output}')
    paths = [args.root/'data/metadata/speaker_summary.csv',
             args.root/'data/metadata/all_pseudo_speaker_summary.csv',
             args.root/'results/speaker_matching/recovered_speaker_matches.csv',
             args.root/'results/pilot_selection/pilot_speakers.csv',
             args.root/'results/preprocessing_analysis/pilot15/pilot15_analysis_summary.json',
             args.metadata, args.snapshot]
    hashes = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}
    full, recovered, previous, pilots = [read(x) for x in paths[:4]]
    if len(full) != 835 or len({r['speaker_id'] for r in full}) != 835:
        raise ValueError('Expected 835 unique speakers')
    if sum(int(r['n_files']) for r in full) != 7507:
        raise ValueError('Expected 7507 summary recordings')
    if len(recovered) != 87 or len(previous) != 87:
        raise ValueError('Expected 87 recovered speakers and matching rows')
    full_by_canonical = {}
    for r in full:
        key = canonical(r['speaker_id'])
        if key != f"{r['speaker_code'].upper()}_{r['sex'].upper()}_{age(r['age'])}" or key in full_by_canonical:
            raise ValueError('Full canonical ID conflict/ambiguity')
        full_by_canonical[key] = r
    recovered_by_canonical, recovered_by_code = defaultdict(list), defaultdict(list)
    old = {r['recovered_pseudo_speaker_id']:r for r in previous}
    if len(old) != 87:
        raise ValueError('Duplicate recovered mapping ID')
    statuses = Counter()
    for r in recovered:
        key = canonical(r['pseudo_speaker_id'])
        if key != f"{r['speaker_code'].upper()}_{r['sex'].upper()}_{age(r['age'])}":
            raise ValueError('Recovered ID/components conflict')
        match = full_by_canonical.get(key)
        expected = 'MATCHED' if match else 'NOT_MATCHED'
        prior = old[r['pseudo_speaker_id']]
        if prior['recovered_canonical_id'] != key or prior['match_status'] != expected or prior['matched_speaker_id'] != (match['speaker_id'] if match else ''):
            raise ValueError('Contradiction with previous matching; stopping')
        statuses[expected] += 1
        recovered_by_canonical[key].append(r['pseudo_speaker_id'])
        recovered_by_code[key.split('_')[0]].append(r['pseudo_speaker_id'])
    if statuses != Counter(MATCHED=61, NOT_MATCHED=26):
        raise ValueError('Matching totals contradict 61/26/0')
    classification = []
    for r in full:
        key = canonical(r['speaker_id'])
        exact = recovered_by_canonical.get(key, [])
        overlap = recovered_by_code.get(r['speaker_code'].upper(), [])
        relation = A if exact else B if overlap else C
        classification.append(dict(**{k:r[k] for k in ['speaker_id','speaker_code','sex','age','dataset','n_files','total_duration_sec']},
                                   recovered_relation=relation, matched_pseudo_speaker_id='|'.join(exact),
                                   overlapping_recovered_codes=r['speaker_code'].upper() if overlap else '',
                                   overlapping_recovered_pseudo_speaker_ids='|'.join(overlap)))
    classes = {r['speaker_id']:r for r in classification}
    if sum(r['recovered_relation']==A for r in classification) != 61:
        raise ValueError('Exact recovered classification must be 61 speakers')
    def class_summary(rs):
        return dict(speaker_count=len(rs), expected_recording_count=sum(int(r['n_files']) for r in rs),
                    duration_sec=sum(float(r['total_duration_sec']) for r in rs),
                    duration_hour=sum(float(r['total_duration_sec']) for r in rs)/3600)
    summaries = [dict(recovered_relation=rel, **class_summary([r for r in classification if r['recovered_relation']==rel])) for rel in [A,B,C]]
    snapshot = json.loads(args.snapshot.read_text(encoding='utf-8-sig'))
    if snapshot['root_id'] != '19yxQbRr6XQm_j4FDbTh1JkFsNH6FcA9g':
        raise ValueError('Unexpected Drive root')
    inventory = defaultdict(list)
    listing_issues = []
    visited = {f['id'] for f in snapshot['folders']}
    for folder in snapshot['folders']:
        if folder.get('error') or len(folder['files']) >= folder['requested_limit']:
            listing_issues.append(dict(issue='ERROR_OR_POSSIBLY_TRUNCATED_LISTING',drive_path=folder['path']))
        for f in folder['files']:
            if f['kind'] == 'folder':
                if f['id'] not in visited:
                    listing_issues.append(dict(issue='UNVISITED_SUBFOLDER',drive_path=folder['path']+'/'+f['name']))
                continue
            inventory[(folder['dataset'],f['name'])].append(dict(f, drive_path=folder['path']+'/'+f['name']))
    metadata = read(args.metadata)
    manifest, unresolved = [], list(listing_issues)
    meta_keys = set()
    metadata_field_conflicts = 0
    for md in metadata:
        ds = DATASETS.get(md['dataset'])
        name = md['file_id']
        try:
            canonical_sid = identity(name)
            sid = full_by_canonical[canonical_sid]['speaker_id'] if canonical_sid in full_by_canonical else canonical_sid
        except ValueError as exc:
            unresolved.append(dict(issue=str(exc), recording_id=Path(name).stem))
            continue
        if canonical_sid != f"{md['speaker'].upper()}_{md['sex'].upper()}_{age(md['age'])}":
            metadata_field_conflicts += 1
        if sid not in classes or ds not in classes[sid]['dataset'].split(','):
            unresolved.append(dict(issue='METADATA_NOT_IN_835_SUMMARY', speaker_id=sid, recording_id=Path(name).stem,dataset=ds))
            continue
        key = (md['dataset'],name)
        if key in meta_keys:
            unresolved.append(dict(issue='DUPLICATE_METADATA_RECORDING', speaker_id=sid,recording_id=Path(name).stem,dataset=ds))
            continue
        meta_keys.add(key)
        cr = classes[sid]
        jname = Path(name).with_suffix('.json').name
        row = dict(speaker_id=sid,speaker_code=cr['speaker_code'],sex=cr['sex'],age=age(cr['age']),dataset=ds,
                   disability_group=GROUPS[ds], recovered_relation=cr['recovered_relation'],recording_id=Path(name).stem,
                   wav_filename=name,json_filename=jname,duration_sec=float(md['play_time']),
                   duration_source='all_metadata.csv play_time', canonical_source='exact recording filename code/sex/age',
                   basename_pairing=False,json_file_id_verified=False,json_file_id_verification='not read; metadata/listing only')
        for kind, source_ds, filename in [('wav',md['dataset'],name),('json',ds,jname)]:
            found = inventory[(source_ds,filename)]
            row[kind+'_drive_id'] = found[0]['id'] if len(found)==1 else ''
            row[kind+'_drive_path'] = found[0]['drive_path'] if len(found)==1 else ''
            row[kind+'_size_bytes'] = int(found[0]['size']) if len(found)==1 and found[0]['size'] is not None else None
            if len(found) != 1 or row[kind+'_size_bytes'] is None:
                unresolved.append(dict(issue='MISSING_OR_AMBIGUOUS_DRIVE_'+kind.upper(),speaker_id=sid,recording_id=row['recording_id'],dataset=ds,candidate_count=len(found)))
        row['basename_pairing'] = bool(row['wav_drive_id'] and row['json_drive_id'])
        manifest.append(row)
    # A speaker with an inconsistent recording universe cannot be resolved by
    # arbitrarily choosing recordings to reproduce the expected summary count.
    inconsistent_speakers = set()
    for sid,cr in classes.items():
        rs = [r for r in manifest if r['speaker_id']==sid]
        if len(rs)!=int(cr['n_files']) or not math.isclose(sum(r['duration_sec'] for r in rs),float(cr['total_duration_sec']),rel_tol=1e-6,abs_tol=.01):
            inconsistent_speakers.add(sid)
            unresolved.append(dict(issue='UNRESOLVED_SPEAKER_RECORDING_UNIVERSE',speaker_id=sid,
                                   expected_recording_count=int(cr['n_files']),metadata_recording_count=len(rs),
                                   expected_duration_sec=cr['total_duration_sec'],metadata_duration_sec=sum(r['duration_sec'] for r in rs)))
    for r in manifest:
        if r['speaker_id'] in inconsistent_speakers:
            unresolved.append(dict(issue='RECORDING_FROM_UNRESOLVED_SPEAKER_UNIVERSE',speaker_id=r['speaker_id'],
                                   recording_id=r['recording_id'],dataset=r['dataset']))
    manifest = [r for r in manifest if r['speaker_id'] not in inconsistent_speakers]
    seen = Counter(r['recording_id'] for r in manifest)
    for rid,count in seen.items():
        if count>1:
            unresolved.append(dict(issue='DUPLICATE_RECORDING_ID',recording_id=rid,count=count))
    for sid,cr in classes.items():
        rs = [r for r in manifest if r['speaker_id']==sid]
        if len(rs) != int(cr['n_files']):
            unresolved.append(dict(issue='SPEAKER_RECORDING_COUNT_DIFFERS',speaker_id=sid,expected=int(cr['n_files']),observed=len(rs)))
        if not math.isclose(sum(r['duration_sec'] for r in rs), float(cr['total_duration_sec']), rel_tol=1e-6, abs_tol=.01):
            unresolved.append(dict(issue='SPEAKER_DURATION_DIFFERS',speaker_id=sid,expected=float(cr['total_duration_sec']),observed=sum(r['duration_sec'] for r in rs)))
    used = {r[k+'_drive_id'] for r in manifest for k in ['wav','json'] if r[k+'_drive_id']}
    extras = [dict(issue='DRIVE_FILE_OUTSIDE_MAPPED_MANIFEST',drive_id=f['id'],drive_path=f['drive_path'])
              for files in inventory.values() for f in files if f['id'] not in used]
    unresolved += extras
    # Verify only JSON already present locally; never download or recreate JSON.
    existing = {r['recording_id']:r for r in read(args.root/'results/preprocessing_pilot/pilot_15_manifest.csv')}
    for r in manifest:
        local = existing.get(r['recording_id'])
        if local and local['json_drive_id']==r['json_drive_id']:
            path = Path(local['local_json_path'])
            if path.exists():
                obj = json.loads(path.read_text(encoding='utf-8-sig'))
                fid = obj.get('File_id')
                if isinstance(fid,str) and Path(fid).stem==r['recording_id']:
                    r['json_file_id_verified']=True
                    r['json_file_id_verification']='existing local JSON File_id basename exact match'
                else:
                    unresolved.append(dict(issue='EXISTING_JSON_FILE_ID_MISMATCH',recording_id=r['recording_id'],observed=str(fid)))
    pilotcheck = []
    for r in pilots:
        cr = classes[r['speaker_id']]
        pilotcheck.append(dict(speaker_id=r['speaker_id'],recovered_relation=cr['recovered_relation'],
                               candidate1_included=cr['recovered_relation'] in [B,C],candidate2_included=cr['recovered_relation']==C,
                               expected_C_verified=cr['recovered_relation']==C))
    if len(pilotcheck)!=15 or not all(r['expected_C_verified'] for r in pilotcheck):
        raise ValueError('Pilot relation contradicts conservative selection')
    def storage(label, cats):
        crs = [r for r in classification if r['recovered_relation'] in cats]
        rs = [r for r in manifest if r['recovered_relation'] in cats]
        complete = len(rs)==sum(int(r['n_files']) for r in crs) and all(r['basename_pairing'] and r['wav_size_bytes'] is not None and r['json_size_bytes'] is not None for r in rs) and all(seen[r['recording_id']]==1 for r in rs) and not listing_issues
        wb = sum(r['wav_size_bytes'] or 0 for r in rs)
        jb = sum(r['json_size_bytes'] or 0 for r in rs)
        return dict(scope=label,**class_summary(crs),mapped_recording_count=len(rs),
                    mapped_speaker_count=len({r['speaker_id'] for r in rs}),
                    basename_paired_count=sum(r['basename_pairing'] for r in rs),
                    mapped_duration_sec=sum(r['duration_sec'] for r in rs),
                    wav_bytes=wb if complete else None,
                    json_bytes=jb if complete else None,total_bytes=wb+jb if complete else None,
                    GB=(wb+jb)/1e9 if complete else None,GiB=(wb+jb)/(2**30) if complete else None,
                    observed_wav_bytes=wb,observed_json_bytes=jb,size_complete=complete,
                    size_source='Drive list_folder size metadata',coverage='835-speaker summary universe; extra Drive files separately recorded')
    storages = [storage('all_835',[A,B,C])]+[storage(rel,[rel]) for rel in [A,B,C]]
    candidates = [('Candidate_1',[B,C],'candidate_exclude_exact_matches.csv'),('Candidate_2',[C],'candidate_exclude_all_code_overlap.csv')]
    candidate_sizes = [storage(label,cats) for label,cats,_ in candidates]
    storages += candidate_sizes
    # Entire traversed original dataset root inventory, including files not in summary.
    unique_files = {f['id']:dict(f,dataset=key[0]) for key,files in inventory.items() for f in files}
    root_wav = [f for f in unique_files.values() if f['name'].lower().endswith('.wav')]
    root_json = [f for f in unique_files.values() if f['name'].lower().endswith('.json')]
    root_complete = not listing_issues and all(f['size'] is not None for f in root_wav+root_json)
    rw = sum(int(f['size'] or 0) for f in root_wav)
    rj = sum(int(f['size'] or 0) for f in root_json)
    storages.append(dict(scope='entire_traversed_drive_roots',wav_count=len(root_wav),json_count=len(root_json),
                         wav_bytes=rw if root_complete else None,json_bytes=rj if root_complete else None,
                         total_bytes=rw+rj if root_complete else None,GB=(rw+rj)/1e9 if root_complete else None,
                         GiB=(rw+rj)/(2**30) if root_complete else None,size_complete=root_complete,
                         coverage='All files returned below eight original dataset roots; includes out-of-summary files'))
    speeds = [(50,'50 Mbps'),(100,'100 Mbps'),(300,'300 Mbps'),(500,'500 Mbps'),(1000,'1 Gbps')]
    downloads = [dict(candidate=s['scope'],speed=label,total_bytes=s['total_bytes'],
                      estimated_download_hours=s['total_bytes']*8/(mbps*1e6)/3600 if s['size_complete'] else None,
                      note='Theoretical transfer only; excludes Drive limits, API/small-file overhead and retries')
                 for s in candidate_sizes for mbps,label in speeds]
    timing = json.loads(paths[4].read_text(encoding='utf-8-sig'))['timing']
    estimates = [dict(candidate=s['scope'],duration_hour=s['duration_hour'],
                      measured_pilot_processing_time_sec=timing['pilot_processing_time_sec'],
                      measured_sec_per_audio_hour=timing['processing_sec_per_audio_hour'],
                      estimated_processing_sec=s['duration_hour']*timing['processing_sec_per_audio_hour'],
                      estimated_processing_hours=s['duration_hour']*timing['processing_sec_per_audio_hour']/3600,
                      note='pilot 기반 extrapolation; download time 및 manual review time 제외; loading/VAD/artifact export included') for s in candidate_sizes]
    report = dict(classification=summaries,metadata_row_count=len(metadata),manifest_recording_count=len(manifest),
                  manifest_unique_speaker_count=len({r['speaker_id'] for r in manifest}),
                  manifest_unique_recording_count=len(seen),basename_paired_count=sum(r['basename_pairing'] for r in manifest),
                  json_file_id_verified_count=sum(r['json_file_id_verified'] for r in manifest),
                  metadata_component_conflict_count=metadata_field_conflicts,
                  issues_by_type=dict(Counter(r['issue'] for r in unresolved)),
                  non_extra_issues=[r for r in unresolved if r['issue'] not in ['DRIVE_FILE_OUTSIDE_MAPPED_MANIFEST','METADATA_NOT_IN_835_SUMMARY']],
                  drive_extra_file_count=len(extras),
                  unresolved_expected_recording_count=7507-len(manifest)+sum(not r['basename_pairing'] for r in manifest),
                  summary_recordings_without_identified_ids=7507-len(manifest),
                  identified_recordings_with_unresolved_pairing=sum(not r['basename_pairing'] for r in manifest),
                  candidates_are_provisional=True,candidate_summaries=candidate_sizes,
                  drive_storage=storages,download_time_estimates=downloads,preprocessing_time_estimates=estimates,
                  pilot_all_C=all(r['expected_C_verified'] for r in pilotcheck),input_sha256=hashes,
                  snapshot_observed_at=snapshot['observed_at'],
                  limitations=['Canonical mapping does not prove biological identity.',
                               'JSON contents not fetched; File_id verified only for already-local pilot JSON.',
                               'Neither candidate selected; no downloads or preprocessing performed.'])
    if any(hashlib.sha256(x.read_bytes()).hexdigest()!=hashes[str(x)] for x in paths):
        raise ValueError('Input changed during analysis')
    if not args.inspect:
        args.output.mkdir(parents=True,exist_ok=False)
        outputs = [('speaker_expansion_classification.csv',classification),('speaker_expansion_summary.csv',summaries),
                   ('all_recordings_manifest.csv',manifest),('manifest_missing_or_unresolved.csv',unresolved),
                   ('drive_storage_summary.csv',storages),('pilot_relation_check.csv',pilotcheck),
                   ('download_time_estimates.csv',downloads),('preprocessing_time_estimates.csv',estimates)]
        outputs += [(filename,[r for r in manifest if r['recovered_relation'] in cats]) for _,cats,filename in candidates]
        for name,rs in outputs:
            write(args.output/name,rs,['issue'])
        with (args.output/'full_expansion_preparation_summary.json').open('x',encoding='utf-8') as f:
            json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
