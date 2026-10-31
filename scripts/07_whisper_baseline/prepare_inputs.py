"""Select a reproducible diverse smoke set, or explicitly prepare the full test."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from general_test import DEFAULT_ROOT, GeneralTest


def key(row):
    return row['batch'] + '/' + row['audio_member']


def prepare(root, full=False):
    dataset = GeneralTest(root)
    if full:
        selected = dataset.test
    else:
        groups = defaultdict(list)
        for row in dataset.test:
            groups[row['disability_group']].append(row)
        selected = []
        used = set()
        for group, rows in sorted(groups.items()):
            # Six duration quantiles per disability group, unique speakers if possible.
            durations = sorted(float(r['duration_sec']) for r in rows)
            for quantile in (0, .2, .4, .6, .8, 1):
                target = durations[round((len(durations)-1)*quantile)]
                candidates = [r for r in rows if r['speaker_id'] not in used]
                if not candidates:
                    raise ValueError('Insufficient distinct test speakers for smoke selection')
                chosen = min(candidates, key=lambda r: (abs(float(r['duration_sec'])-target), key(r)))
                selected.append(chosen)
                used.add(chosen['speaker_id'])
    return dict(schema=1, scope='full' if full else 'smoke',
                manifest_sha256=dataset.manifest_sha256, speaker_split_sha256=dataset.speaker_sha256,
                total_test_rows=len(dataset.test), total_test_duration_sec=sum(float(r['duration_sec']) for r in dataset.test),
                rows=selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--full-test', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        p.error('Selection already exists')
    selection = prepare(args.data_root, args.full_test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(selection, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(scope=selection['scope'], selected=len(selection['rows']),
                          speakers=len({r['speaker_id'] for r in selection['rows']}),
                          groups=sorted({r['disability_group'] for r in selection['rows']}),
                          durations=[float(r['duration_sec']) for r in selection['rows']]), ensure_ascii=False))


if __name__ == '__main__':
    main()
