"""Build a reproducible smoke report and duration-adjusted full-test time estimate."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics

import numpy as np
from scipy.optimize import nnls
from general_test import DEFAULT_ROOT, GeneralTest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    args = p.parse_args()
    dataset = GeneralTest(args.data_root)
    with sqlite3.connect(args.run/'checkpoint.sqlite3') as db:
        records = [json.loads(r[0]) for r in db.execute(
            'SELECT a.payload FROM attempts a JOIN (SELECT utterance,MAX(id) id FROM attempts GROUP BY utterance) b ON a.id=b.id')]
        attempt_count = db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]
    good = [r for r in records if r['status']=='success']
    if not good:
        raise ValueError('No successful smoke records')
    scores = json.loads((args.run/'scores.json').read_text(encoding='utf-8'))
    sessions = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(args.run.glob('session_*.json'))]
    by_group = defaultdict(list)
    for row in good:
        by_group[row['disability_group']].append(row)
    fit = {}
    estimate = 0
    for group, rows in by_group.items():
        # Nonnegative fixed overhead + duration slope; sparse smoke fit, not a confidence interval.
        x = np.array([[1,float(r['duration_sec'])] for r in rows])
        coef,_ = nnls(x,np.array([r['inference_sec'] for r in rows]))
        population = [r for r in dataset.test if r['disability_group']==group]
        seconds = len(population)*coef[0]+sum(float(r['duration_sec']) for r in population)*coef[1]
        fit[group] = dict(intercept_sec=float(coef[0]),seconds_per_audio_sec=float(coef[1]),
                          estimated_inference_hours=float(seconds/3600),population=len(population))
        estimate += seconds
    median_loading = statistics.median(r['loading_sec'] for r in good)
    adjusted = estimate+len(dataset.test)*median_loading
    summary = dict(utterances=len(records),successful=len(good),failures=len(records)-len(good),
                   pseudo_speakers=len({r['speaker_id'] for r in records}),
                   groups=sorted(by_group),duration_range_sec=[min(float(r['duration_sec']) for r in good),max(float(r['duration_sec']) for r in good)],
                   all_zip_transcripts_match=all(r.get('transcript_matches') for r in good),
                   acoustic_alignment_independently_verified=False,
                   normalized_cer=scores['overall']['normalized']['cer'],normalized_wer=scores['overall']['normalized']['wer'],
                   raw_cer=scores['overall']['raw']['cer'],raw_wer=scores['overall']['raw']['wer'],
                   inference_sec=dict(min=min(r['inference_sec'] for r in good),median=statistics.median(r['inference_sec'] for r in good),
                                      max=max(r['inference_sec'] for r in good),total=sum(r['inference_sec'] for r in good)),
                   loading_sec=dict(median=median_loading,total=sum(r['loading_sec'] for r in good)),
                   peak_allocated_mib=max(r['peak_allocated_mib'] for r in good),
                   peak_reserved_mib=max(r['peak_reserved_mib'] for r in good),
                   resume_sessions=[dict(processed=s['processed'],skipped=s['skipped'],pending=s['pending']) for s in sessions],
                   checkpoint_attempt_count=attempt_count,
                   no_duplicate_attempts=attempt_count==len(records),
                   full_test=dict(utterances=len(dataset.test),duration_hours=sum(float(r['duration_sec']) for r in dataset.test)/3600,
                                  duration_adjusted_inference_hours=estimate/3600,
                                  with_median_loading_hours=adjusted/3600,
                                  planning_range_hours=[adjusted/3600,adjusted*2/3600],group_fits=fit,
                                  caveat='18 duration-stratified samples, not random. Range is a planning heuristic, not a confidence interval. Drive hydration, thermal throttling, long decoding, retries and cache state can exceed it.'))
    output = args.run/'smoke_summary.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
