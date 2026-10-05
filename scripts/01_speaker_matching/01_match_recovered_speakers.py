"""Exact canonical-ID matching; never modifies source CSVs or uses fuzzy matching."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path


def normalize_age(value):
    age = Decimal(str(value).strip())
    if not age.is_finite() or age < 0 or age != age.to_integral_value():
        raise ValueError(f"Invalid integer age: {value!r}")
    return str(int(age))


def canonicalize(identifier):
    value = identifier.strip().upper()
    # Only the known recovery prefix is removed; arbitrary prefixes are not guessed.
    if value.startswith("PN_"):
        value = value[3:]
    match = re.fullmatch(r"([A-Z0-9]+)[_-]([FM])[_-](\d+(?:\.0+)?)", value)
    if not match:
        raise ValueError(f"Unsupported pseudo-speaker ID: {identifier!r}")
    code, sex, age = match.groups()
    return f"{code}_{sex}_{normalize_age(age)}"


def read_rows(path, id_column):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {id_column, "speaker_code", "sex", "age"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path}: {required}")
        rows = list(reader)
    for row in rows:
        canonical = canonicalize(row[id_column])
        components = (row["speaker_code"].strip().upper(),
                      row["sex"].strip().upper(), normalize_age(row["age"]))
        if canonical != "_".join(components):
            raise ValueError(f"ID/metadata conflict: {row[id_column]}")
        row["canonical_id"] = canonical
        row["components"] = components
    return rows


def diagnose(components, full_rows):
    code, sex, age = components
    same_code = [r for r in full_rows if r["components"][0] == code]
    if not same_code:
        return "SPEAKER_CODE_ABSENT", "speaker_code absent; sex/age comparison unavailable"
    same_code_sex = [r for r in same_code if r["components"][1] == sex]
    if same_code_sex:
        ages = sorted({r["components"][2] for r in same_code_sex}, key=int)
        return "AGE_DIFFERS", "same speaker_code/sex; available ages=" + "|".join(ages)
    sexes = sorted({r["components"][1] for r in same_code})
    ages = sorted({r["components"][2] for r in same_code}, key=int)
    return "SEX_DIFFERS", "same speaker_code; available sexes=" + "|".join(sexes) + "; available ages=" + "|".join(ages)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.project_root
    full_path = root / "data/metadata/speaker_summary.csv"
    recovered_path = root / "data/metadata/all_pseudo_speaker_summary.csv"
    before = {p: digest(p) for p in (full_path, recovered_path)}
    full = read_rows(full_path, "speaker_id")
    recovered = read_rows(recovered_path, "pseudo_speaker_id")
    if len(full) != 835 or len({r["speaker_id"] for r in full}) != 835:
        raise ValueError("Expected 835 unique full pseudo-speakers")
    if len(recovered) != 87 or len({r["pseudo_speaker_id"] for r in recovered}) != 87:
        raise ValueError("Expected 87 unique recovered pseudo-speakers")
    index = defaultdict(list)
    for row in full:
        index[row["canonical_id"]].append(row)
    output = []
    for row in recovered:
        candidates = index[row["canonical_id"]]
        status = "MATCHED" if len(candidates) == 1 else "NOT_MATCHED" if not candidates else "AMBIGUOUS"
        kind, detail = diagnose(row["components"], full) if not candidates else ("", "")
        output.append({
            "recovered_pseudo_speaker_id": row["pseudo_speaker_id"],
            "recovered_canonical_id": row["canonical_id"],
            "matched_speaker_id": candidates[0]["speaker_id"] if len(candidates) == 1 else "",
            "match_status": status,
            "candidate_count": len(candidates),
            "candidate_speaker_ids": "|".join(r["speaker_id"] for r in candidates),
            "mismatch_type": kind,
            "mismatch_details": detail,
        })
    counts = Counter(r["match_status"] for r in output)
    assert sum(counts.values()) == 87
    target = root / "results/speaker_matching/recovered_speaker_matches.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    if any(digest(p) != before[p] for p in before):
        raise RuntimeError("Source CSV changed during analysis")
    print(f"Full: {len(full)}; recovered: {len(recovered)}")
    for status in ("MATCHED", "NOT_MATCHED", "AMBIGUOUS"):
        print(f"{status}: {counts[status]}")
    print("NOT_MATCHED diagnostics (exact component comparisons only; no identity inference):")
    for row in output:
        if row["match_status"] == "NOT_MATCHED":
            print(f"{row['recovered_pseudo_speaker_id']}: {row['mismatch_type']}; {row['mismatch_details']}")
    print("Mismatch counts:", json.dumps(dict(Counter(r["mismatch_type"] for r in output if r["match_status"] == "NOT_MATCHED"))))
    print(f"Result: {target}")
    print("Source SHA256 unchanged")


if __name__ == "__main__":
    main()
