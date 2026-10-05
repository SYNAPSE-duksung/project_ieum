"""Select 15 reproducible pilot pseudo-speakers using local metadata only."""
import csv
import hashlib
import json
import random
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

RANDOM_SEED = 42
GROUPS = {"TL01": "뇌신경장애", "VL01": "뇌신경장애",
          "TL02": "언어·청각 장애", "TL03": "후두장애"}
QUOTAS = {"low": 1, "medium": 2, "high": 2}

def canonical(identifier):
    value = identifier.strip().upper()
    if value.startswith("PN_"):
        value = value[3:]
    match = re.fullmatch(r"([A-Z0-9]+)[_-]([FM])[_-](\d+(?:\.0+)?)", value)
    if not match:
        raise ValueError(f"Invalid ID: {identifier}")
    code, sex, age = match.groups()
    return f"{code}_{sex}_{int(Decimal(age))}"

def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)

def main():
    root = Path(__file__).resolve().parents[2]
    sources = [root / "data/metadata/speaker_summary.csv",
               root / "results/speaker_matching/recovered_speaker_matches.csv"]
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    full, recovered = map(read, sources)
    assert len(full) == 835 and len({r["speaker_id"] for r in full}) == 835
    assert len(recovered) == 87
    assert Counter(r["match_status"] for r in recovered) == {"MATCHED": 61, "NOT_MATCHED": 26}
    full_index = {}
    for row in full:
        row["canonical"] = canonical(row["speaker_id"])
        expected = f'{row["speaker_code"].strip().upper()}_{row["sex"].strip().upper()}_{int(Decimal(row["age"]))}'
        assert row["canonical"] == expected
        assert row["canonical"] not in full_index
        full_index[row["canonical"]] = row
        row["n_files"] = int(row["n_files"])
        assert row["n_files"] > 0
    recovered_ids = {canonical(r["recovered_canonical_id"]) for r in recovered}
    recovered_codes = {c.rsplit("_", 2)[0] for c in recovered_ids}
    matched_ids = {r["matched_speaker_id"] for r in recovered if r["match_status"] == "MATCHED"}
    assert len(matched_ids) == 61 and matched_ids <= {r["speaker_id"] for r in full}
    uncertain = []
    recovered_exact = set()
    for row in recovered:
        key = canonical(row["recovered_canonical_id"])
        assert key == canonical(row["recovered_pseudo_speaker_id"])
        if row["match_status"] == "MATCHED":
            assert canonical(row["matched_speaker_id"]) == key
        elif key in full_index:
            recovered_exact.add(full_index[key]["speaker_id"])
        else:
            uncertain.append(row)
    summary = []
    def stat(section, group, metric, value, details=""):
        summary.append(dict(section=section, disability_group=group,
                            metric=metric, value=value, details=details))
    eligible = []
    exclusions = Counter()
    by_group_exclusions = Counter()
    for row in full:
        mapped = {GROUPS[d.strip()] for d in row["dataset"].split(",") if d.strip() in GROUPS}
        group = next(iter(mapped)) if len(mapped) == 1 else "MULTIPLE_OR_UNKNOWN"
        if row["speaker_id"] in matched_ids:
            reason = "MATCHED_ID"
        elif row["speaker_id"] in recovered_exact:
            reason = "NOT_MATCHED_EXACT_ID"
        elif row["speaker_code"].strip().upper() in recovered_codes:
            reason = "RECOVERED_CODE_OVERLAP"
        elif len(mapped) != 1:
            reason = "MULTIPLE_OR_UNKNOWN_GROUP"
        else:
            row["disability_group"] = group
            eligible.append(row)
            continue
        exclusions[reason] += 1
        by_group_exclusions[(group, reason)] += 1
    stat("pool", "ALL", "total", len(full))
    for reason in ["MATCHED_ID", "NOT_MATCHED_EXACT_ID", "RECOVERED_CODE_OVERLAP", "MULTIPLE_OR_UNKNOWN_GROUP"]:
        stat("pool", "ALL", reason, exclusions[reason])
    stat("pool", "ALL", "eligible", len(eligible))
    stat("configuration", "ALL", "random_seed", RANDOM_SEED)
    stat("configuration", "ALL", "quotas", 5, json.dumps(QUOTAS))
    stat("configuration", "ALL", "group_basis", "dataset folder mapping",
         "Disability group is a dataset-level label, not a confirmed individual diagnosis. Disease omitted.")
    for row in uncertain:
        stat("uncertain_recovered_overlap", "ALL", row["recovered_pseudo_speaker_id"],
             row["recovered_canonical_id"], row.get("mismatch_details", ""))
    for (group, reason), count in sorted(by_group_exclusions.items()):
        stat("exclusions", group, reason, count)
    rng = random.Random(RANDOM_SEED)
    chosen = []
    used_codes = set()
    for group in ["뇌신경장애", "언어·청각 장애", "후두장애"]:
        pool = sorted([r for r in eligible if r["disability_group"] == group],
                      key=lambda r: (r["n_files"], r["speaker_id"]))
        if not pool:
            raise ValueError(f"No eligible candidates: {group}")
        q1 = quantile([r["n_files"] for r in pool], 1/3)
        q2 = quantile([r["n_files"] for r in pool], 2/3)
        stat("group_pool", group, "eligible", len(pool))
        stat("quantiles", group, "q33_n_files", q1, "low <= q33; medium q33 < n_files <= q67; high > q67")
        stat("quantiles", group, "q67_n_files", q2, "Linear interpolation; equal n_files stay in the same band")
        for row in pool:
            row["data_volume_group"] = "low" if row["n_files"] <= q1 else "medium" if row["n_files"] <= q2 else "high"
        for band, count in QUOTAS.items():
            candidates = [r for r in pool if r["data_volume_group"] == band]
            stat("volume_pool", group, band, len(candidates))
            rng.shuffle(candidates)
            selected = []
            for row in candidates:
                code = row["speaker_code"].strip().upper()
                if code not in used_codes:
                    selected.append(row)
                    used_codes.add(code)
                if len(selected) == count:
                    break
            if len(selected) != count:
                raise ValueError(f"Insufficient distinct codes: {group}/{band}; refusing quota fallback")
            chosen.extend(selected)
            stat("selected", group, band, len(selected))
    assert len(chosen) == 15 and len({r["speaker_id"] for r in chosen}) == 15
    id_collisions = {r["canonical"] for r in chosen} & recovered_ids
    code_collisions = {r["speaker_code"].strip().upper() for r in chosen} & recovered_codes
    assert not id_collisions and not code_collisions
    assert Counter(r["disability_group"] for r in chosen) == {g: 5 for g in set(GROUPS.values())}
    stat("validation", "ALL", "canonical_id_collisions", len(id_collisions))
    stat("validation", "ALL", "speaker_code_collisions", len(code_collisions))
    stat("validation", "ALL", "unique_pilot_codes", len(used_codes))
    for p in sources:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p]
        stat("source_sha256", "ALL", str(p), hashes[p], "unchanged")
    columns = ["speaker_id", "speaker_code", "sex", "age", "dataset",
               "disability_group", "n_files", "total_duration_min", "data_volume_group"]
    output = [{key: r[key] for key in columns} for r in chosen]
    destination = root / "results/pilot_selection"
    destination.mkdir(parents=True, exist_ok=True)
    write(destination / "pilot_speakers.csv", output)
    write(destination / "pilot_selection_summary.csv", summary)
    assert read(destination / "pilot_speakers.csv") == [{k: str(v) for k,v in r.items()} for r in output]
    print("Pool:", len(full), "exclusions:", dict(exclusions), "eligible:", len(eligible))
    print("Uncertain recovered overlap:", len(uncertain))
    for row in output:
        print(row["disability_group"], row["speaker_id"], row["dataset"],
              row["n_files"], round(float(row["total_duration_min"]), 2), row["data_volume_group"])
    print("Canonical ID collisions:", len(id_collisions), "speaker_code collisions:", len(code_collisions))
    print("Source hashes unchanged; saved CSV readback passed.")

if __name__ == "__main__":
    main()

