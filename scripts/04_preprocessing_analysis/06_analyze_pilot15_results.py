"""Read-only pilot analysis: CSV aggregation and existing segment WAV headers only."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('C:/ieum')
GROUPS = {'TL01': '뇌신경장애', 'VL01': '뇌신경장애',
          'TL02': '언어·청각 장애', 'TL03': '후두장애'}
STATUSES = ('SUCCESS', 'NEEDS_MANUAL_REVIEW', 'ERROR')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid {key}: {value}')
    return value


def integer(row, key):
    value = number(row, key)
    if value != int(value):
        raise ValueError(f'Noninteger {key}: {value}')
    return int(value)


def boolean(value):
    if value.lower() not in ('true', 'false'):
        raise ValueError(f'Invalid boolean: {value}')
    return value.lower() == 'true'


def ratio(a, b):
    return a / b if b else None


def status_summary(rows):
    counts = Counter(r['final_status'] for r in rows)
    return dict(speaker_count=len({r['speaker_id'] for r in rows}),
                recording_count=len(rows), **{s: counts[s] for s in STATUSES},
                success_rate=ratio(counts['SUCCESS'], len(rows)),
                manual_review_rate=ratio(counts['NEEDS_MANUAL_REVIEW'], len(rows)))


def usable(rows):
    success = [r for r in rows if r['final_status'] == 'SUCCESS']
    complete = all(r['_artifact_complete'] for r in success)
    total = sum(r['_total'] for r in success)
    valid = sum(r['_valid'] for r in success)
    return dict(success_recording_count=len(success), total_segments=total,
                valid_1_30s_segments=valid, valid_segment_ratio=ratio(valid, total),
                success_recording_audio_duration_sec=sum(number(r, 'audio_duration_sec') for r in success),
                success_segment_duration_sec=sum(r['_segment_duration'] for r in success) if complete else None,
                valid_segment_duration_sec=sum(r['_valid_duration'] for r in success) if complete else None,
                valid_segment_duration_hour=sum(r['_valid_duration'] for r in success)/3600 if complete else None,
                artifact_duration_complete=complete,
                duration_source='existing final-pass segment WAV headers; inclusive 1 <= seconds <= 30')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys or ['issue'])
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    if args.output.exists():
        raise ValueError(f'Output already exists; stopping without overwrite: {args.output}')
    rows, manifest = read_csv(args.results), read_csv(args.manifest)
    if len(rows) != 139:
        raise ValueError(f'WARNING: expected 139 result rows, got {len(rows)}; analysis stopped')
    if len(manifest) != 139:
        raise ValueError('Manifest must contain exactly 139 recordings')
    def index(records):
        keys = [r['recording_id'] for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate recording_id')
        return {r['recording_id']: r for r in records}
    mi, ri = index(manifest), index(rows)
    if set(mi) != set(ri) or len({r['speaker_id'] for r in rows}) != 15:
        raise ValueError('Manifest/result IDs differ or speaker count is not 15')
    inputs = [args.results, args.manifest]
    if args.metadata.exists():
        inputs.append(args.metadata)
    hashes = {str(p): digest(p) for p in inputs}
    artifacts, issues = [], []
    for r in rows:
        rid = r['recording_id']
        for key in ('speaker_id', 'disability_group'):
            if r[key] != mi[rid][key]:
                raise ValueError(f'Manifest mismatch {rid}: {key}')
        if r['final_status'] not in STATUSES:
            raise ValueError(f'Unknown status: {r}')
        r['_retry'] = boolean(r['retry_needed'])
        r['_total'] = integer(r, 'total_segments')
        r['_valid'] = integer(r, 'valid_1_30s_segments')
        r['_segment_duration'] = r['_valid_duration'] = 0.0
        r['_artifact_complete'] = True
        number(r, 'audio_duration_sec')
        number(r, 'processing_time_sec')
        if r['final_status'] != 'SUCCESS':
            continue
        passnum = integer(r, 'final_pass')
        if passnum not in (1, 2):
            raise ValueError(f'Invalid successful final pass: {rid}')
        if integer(r, f'vad{passnum}_audio_segment_count') != integer(r, 'text_segment_count'):
            raise ValueError(f'SUCCESS count mismatch: {rid}')
        passdir = args.results.parent / rid / f'vad{passnum}'
        try:
            segments = read_csv(passdir / 'segments.csv')
            names = [s['wav_name'] for s in segments]
            if len(names) != len(set(names)) or len(names) != r['_total']:
                raise ValueError('Artifact index count/uniqueness mismatch')
            if {p.name for p in passdir.glob('*.wav')} != set(names):
                raise ValueError('Artifact WAV inventory differs from segments.csv')
            measured_valid = 0
            for s in segments:
                path = passdir / s['wav_name']
                if path.resolve().parent != passdir.resolve():
                    raise ValueError('Unsafe artifact path')
                with wave.open(str(path), 'rb') as wav:
                    duration = wav.getnframes() / wav.getframerate()
                valid = 1 <= duration <= 30
                if abs(duration - number(s, 'duration_sec')) > 1e-6 or valid != boolean(s['valid_1_30s']):
                    raise ValueError(f'Artifact duration/validity differs: {path.name}')
                measured_valid += valid
                r['_segment_duration'] += duration
                r['_valid_duration'] += duration if valid else 0
                artifacts.append(dict(speaker_id=r['speaker_id'], disability_group=r['disability_group'],
                                      recording_id=rid, final_pass=passnum, wav_path=str(path),
                                      duration_sec=duration, valid_1_30s=valid))
            if measured_valid != r['_valid']:
                raise ValueError('Result CSV valid count differs from WAV headers')
        except Exception as exc:
            r['_artifact_complete'] = False
            issues.append(dict(recording_id=rid, issue=f'{type(exc).__name__}: {exc}'))

    groups, speakers = defaultdict(list), defaultdict(list)
    for r in rows:
        groups[r['disability_group']].append(r)
        speakers[r['speaker_id']].append(r)
    group_summary = [dict(disability_group=g, **status_summary(rs)) for g, rs in sorted(groups.items())]
    speaker_summary = [dict(speaker_id=s, disability_group=rs[0]['disability_group'],
                            **status_summary(rs), **usable(rs)) for s, rs in sorted(speakers.items())]
    speaker_usable = [dict(speaker_id=s, disability_group=rs[0]['disability_group'], **usable(rs))
                      for s, rs in sorted(speakers.items())]
    usable_summary = [dict(scope='overall', disability_group='ALL', **usable(rows))]
    usable_summary += [dict(scope='disability_group', disability_group=g, **usable(rs)) for g, rs in sorted(groups.items())]
    mismatches = []
    bins = Counter()
    directions = Counter()
    for r in rows:
        if r['final_status'] != 'NEEDS_MANUAL_REVIEW':
            continue
        p = integer(r, 'final_pass')
        audio = integer(r, f'vad{p}_audio_segment_count')
        textcount = integer(r, 'text_segment_count')
        signed = audio - textcount
        diff = abs(signed)
        bucket = next(label for limit, label in [(0, '0'), (1, '1'), (2, '2'), (5, '3~5'),
                      (10, '6~10'), (50, '11~50'), (100, '51~100'), (math.inf, '>100')] if diff <= limit)
        direction = 'audio_more' if signed > 0 else 'audio_less' if signed < 0 else 'equal'
        bins[bucket] += 1
        directions[direction] += 1
        mismatches.append(dict(speaker_id=r['speaker_id'], disability_group=r['disability_group'],
                               recording_id=r['recording_id'], final_pass=p,
                               audio_segment_count=audio, text_segment_count=textcount,
                               signed_difference=signed, absolute_difference=diff,
                               difference_bin=bucket, direction=direction, failure_reason=r['failure_reason']))
    mismatch_summary = [dict(category='absolute_difference', value=k, count=bins[k],
                             denominator=len(mismatches), ratio=ratio(bins[k], len(mismatches)))
                        for k in ['0', '1', '2', '3~5', '6~10', '11~50', '51~100', '>100']]
    mismatch_summary += [dict(category='direction', value=k, count=directions[k],
                              denominator=len(mismatches), ratio=ratio(directions[k], len(mismatches)))
                         for k in ['audio_more', 'audio_less', 'equal']]
    processing = [number(r, 'processing_time_sec') for r in rows]
    pilot_audio = sum(number(r, 'audio_duration_sec') for r in rows)
    vadtime = sum(number(r, k) for r in rows for k in ['vad1_processing_time_sec', 'vad2_processing_time_sec'] if r.get(k))
    timing = dict(pilot_processing_time_sec=sum(processing), mean_per_recording_sec=statistics.mean(processing),
                  median_per_recording_sec=statistics.median(processing), pilot_audio_duration_hour=pilot_audio/3600,
                  processing_sec_per_audio_hour=sum(processing)/(pilot_audio/3600),
                  pilot_vad_only_time_sec=vadtime, vad_only_sec_per_audio_hour=vadtime/(pilot_audio/3600),
                  processing_time_scope='processing_time includes pairing, JSON/audio loading, VAD and artifact export; excludes Drive downloads and manual review')
    scale = []
    full = None
    if args.metadata.exists():
        metadata = read_csv(args.metadata)
        if len({r['speaker_id'] for r in metadata}) != len(metadata):
            raise ValueError('Full metadata contains duplicate speaker_id; cannot sum safely')
        def scale_row(scope, label, records):
            duration = sum(number(r, 'total_duration_sec') for r in records)
            return dict(scope=scope, label=label, speaker_count=len(records),
                        recording_count=sum(integer(r, 'n_files') for r in records),
                        duration_sec=duration, duration_hour=duration/3600,
                        storage_estimate='not directly available',
                        allocation_note='Exact dataset combinations; multi-dataset recording/duration cannot be allocated to individual datasets from this CSV')
        full = scale_row('overall', 'ALL', metadata)
        full.update(duration_scaled_processing_reference_sec=full['duration_hour']*timing['processing_sec_per_audio_hour'],
                    duration_scaled_vad_only_reference_sec=full['duration_hour']*timing['vad_only_sec_per_audio_hour'],
                    estimate_note='Pilot duration-rate extrapolation only; workload and hardware dependent; excludes downloads/manual review')
        scale.append(full)
        bydataset, bygroup = defaultdict(list), defaultdict(list)
        for r in metadata:
            codes = sorted(set(c.strip() for c in r['dataset'].split(',') if c.strip()))
            bydataset[','.join(codes)].append(r)
            groupnames = sorted({GROUPS.get(c, f'UNKNOWN:{c}') for c in codes})
            bygroup[' + '.join(groupnames)].append(r)
        scale += [scale_row('dataset_combination', d, rs) for d, rs in sorted(bydataset.items())]
        scale += [scale_row('disability_group_combination', g, rs) for g, rs in sorted(bygroup.items())]
    else:
        issues.append(dict(recording_id='', issue='Full speaker metadata not available'))
    scale.append(dict(scope='pilot_timing', label='ALL', **timing, storage_estimate='not directly available'))
    rates = [r['success_rate'] for r in speaker_summary]
    summary = dict(**status_summary(rows), error_rate=ratio(sum(r['final_status']=='ERROR' for r in rows), len(rows)),
                   retry_needed_count=sum(r['_retry'] for r in rows),
                   retry_needed_rate=ratio(sum(r['_retry'] for r in rows), len(rows)),
                   vad1_success_count=sum(r['final_status']=='SUCCESS' and r['final_pass']=='1' for r in rows),
                   vad2_success_count=sum(r['final_status']=='SUCCESS' and r['final_pass']=='2' for r in rows),
                   group_summary=group_summary, speaker_success_rate_min=min(rates), speaker_success_rate_max=max(rates),
                   near_mismatch_count=sum(r['absolute_difference']<=2 for r in mismatches),
                   near_mismatch_rate=ratio(sum(r['absolute_difference']<=2 for r in mismatches), len(mismatches)),
                   large_mismatch_count=sum(r['absolute_difference']>10 for r in mismatches),
                   large_mismatch_rate=ratio(sum(r['absolute_difference']>10 for r in mismatches), len(mismatches)),
                   mismatch_rate_denominator='NEEDS_MANUAL_REVIEW recordings', mismatch_bins=dict(bins),
                   mismatch_directions=dict(directions), usable_data=usable(rows), timing=timing,
                   full_dataset=full, input_sha256=hashes, issues=issues,
                   limitations=['Count match is not verified semantic transcript alignment.',
                                'Original recording duration differs from retained segment duration.',
                                'Storage bytes are not directly available from full speaker metadata.',
                                'No preprocessing, correction, download or training was performed.'])
    if any(digest(p) != hashes[str(p)] for p in inputs):
        raise ValueError('Input changed during analysis; refusing to publish')
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('group_summary.csv', group_summary), ('speaker_summary.csv', speaker_summary),
                       ('usable_data_summary.csv', usable_summary), ('speaker_usable_data.csv', speaker_usable),
                       ('mismatch_summary.csv', mismatch_summary), ('mismatch_recordings.csv', mismatches),
                       ('full_dataset_scale_estimate.csv', scale), ('segment_duration_audit.csv', artifacts),
                       ('analysis_issues.csv', issues)]:
        write_csv(args.output / name, data)
    with (args.output / 'pilot15_analysis_summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/preprocessing_pilot/pilot15_run1/preprocessing_results.csv')
    parser.add_argument('--manifest', type=Path, default=ROOT/'results/preprocessing_pilot/pilot_15_manifest.csv')
    parser.add_argument('--metadata', type=Path, default=ROOT/'data/metadata/speaker_summary.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'results/preprocessing_analysis/pilot15')
    args = parser.parse_args()
    try:
        analyze(args)
    except Exception as exc:
        print(f'Analysis stopped: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
