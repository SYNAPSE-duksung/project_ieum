"""Static repository inspection and metadata audit only. Never executes notebooks/audio."""
import argparse
import csv
import json
import re
import urllib.request
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
COMMIT = "4624249a217e85ff91752c2f689eca4ede15686a"
REPO = "yoona-J/Diagnosis-Aware_Multitask_Fine-Tuning_of_Whisper_for_DSR"

def download_repo():
    paths = ["Preprocess/README.md", "Fine-Tuning/README.md"]
    paths += [f"Preprocess/[Distribution]Preprocessing_{name}.ipynb" for name in ["Stroke", "Cerebral_Palsy", "Peripheral_Neuropathy"]]
    paths += [f"Fine-Tuning/[Distribution]FineTuning_{name}.ipynb" for name in ["Stroke", "Cerebral_Palsy", "Peripheral_Neuropathy", "GeneralModel"]]
    for path in paths:
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/" + quote(path)
        raw = urllib.request.urlopen(url, timeout=40).read()
        target = ROOT / "reference_code" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        if path.endswith(".ipynb"):
            nb = json.loads(raw)
            lines = []
            for index, cell in enumerate(nb["cells"], 1):
                if cell["cell_type"] == "code":
                    lines.append(f"\n### CELL {index} (1-based notebook cell index)")
                    for line_index, line in enumerate("".join(cell["source"]).splitlines(), 1):
                        lines.append(f"C{index}:L{line_index}: {line}")
            target.with_suffix(".code.txt").write_text("\n".join(lines), encoding="utf-8")
        print("Saved static source:", path)

def audit_metadata():
    with (ROOT / "all_metadata_drive.csv").open(encoding="utf-8-sig", newline="") as f:
        metadata = list(csv.DictReader(f))
    project = ROOT.parents[1]
    with (project / "results/pilot_selection/pilot_speakers.csv").open(encoding="utf-8-sig", newline="") as f:
        pilots = list(csv.DictReader(f))
    result = []
    for pilot in pilots:
        dataset = {"TL01":"TS01", "TL02":"TS02", "TL03":"TS03", "VL01":"VS01"}[pilot["dataset"]]
        found = []
        for row in metadata:
            match = re.search(r"-([A-Za-z0-9]+)-(?:\d+-)*([FM])-(\d+)-", row["file_id"], re.I)
            if not match:
                continue
            key = f"{match[1].upper()}_{match[2].upper()}_{int(match[3])}"
            if key == pilot["speaker_id"] and row["dataset"] == dataset:
                found.append(row)
        ids = [r["file_id"] for r in found]
        duration = sum((Decimal(r["play_time"]) for r in found), Decimal(0)) / 60
        item = dict(speaker_id=pilot["speaker_id"], dataset=dataset,
                    metadata_rows=len(found), unique_wav_names=len(set(ids)),
                    metadata_duration_min=str(duration),
                    summary_n_files=int(pilot["n_files"]), summary_duration_min=pilot["total_duration_min"],
                    file_ids=ids, diseases=sorted({r["disease"] for r in found}))
        result.append(item)
        print(json.dumps(item, ensure_ascii=False))
    (ROOT / "pilot_metadata_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-reference", action="store_true")
    parser.add_argument("--audit-metadata", action="store_true")
    args = parser.parse_args()
    if args.download_reference:
        download_repo()
    if args.audit_metadata:
        audit_metadata()

