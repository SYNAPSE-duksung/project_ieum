"""Audit bounded Drive inventory against exact pilot IDs; no WAV access."""
import csv
import json
import re
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def identity(name):
    match = re.search(r"-([A-Za-z0-9]+)-(?:\d+-)*([FM])-(\d+)-", name, re.I)
    return f"{match[1].upper()}_{match[2].upper()}_{int(match[3])}" if match else None

def main():
    inventory = json.loads((ROOT / "pilot_drive_inventory.json").read_text(encoding="utf-8-sig"))
    metadata = json.loads((ROOT / "pilot_metadata_audit.json").read_text(encoding="utf-8"))
    with (ROOT.parents[1] / "results/pilot_selection/pilot_speakers.csv").open(encoding="utf-8-sig", newline="") as f:
        pilots = list(csv.DictReader(f))
    result = []
    for pilot in pilots:
        prefix = {"TL02": ("TS","TL"), "TL03": ("TS","TL"), "VL01": ("VS","VL")}[pilot["dataset"]]
        category = "2" if pilot["dataset"] == "TL02" else "3" if pilot["dataset"] == "TL03" else ""
        audio, labels = [], []
        for folder in inventory:
            assert folder["observed_count"] < folder["requested_limit"], "Possible partial folder listing"
            if category and not folder["label"][2:].startswith(category):
                continue
            for file in folder["files"]:
                if identity(file["name"]) != pilot["speaker_id"]:
                    continue
                record = dict(file, folder_id=folder["id"], folder_label=folder["label"])
                if folder["label"].startswith(prefix[0]) and file["name"].endswith(".wav"):
                    audio.append(record)
                if folder["label"].startswith(prefix[1]) and file["name"].endswith(".json"):
                    labels.append(record)
        audio_names = {f["name"] for f in audio}
        label_wav_names = {f["name"][:-5]+".wav" for f in labels}
        audit = next(r for r in metadata if r["speaker_id"] == pilot["speaker_id"])
        expected = set(audit["file_ids"])
        item = dict(audit, actual_wav_count=len(audio), actual_json_count=len(labels),
                    complete_pairs=len(audio_names & label_wav_names),
                    duplicate_wav_names=[n for n,c in Counter(f["name"] for f in audio).items() if c > 1],
                    missing_expected_wav=sorted(expected-audio_names),
                    missing_expected_json=sorted(expected-label_wav_names),
                    extra_wav=sorted(audio_names-expected), extra_json=sorted(label_wav_names-expected),
                    audio=audio, labels=labels)
        result.append(item)
        print(pilot["speaker_id"], "WAV",len(audio),"JSON",len(labels),"pairs",item["complete_pairs"],
              "duration_min",round(float(audit["metadata_duration_min"]),3),
              "missing",item["missing_expected_wav"],item["missing_expected_json"],
              "extra",item["extra_wav"],item["extra_json"])
    (ROOT / "pilot_drive_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()

