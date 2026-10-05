"""Compare JSON metadata with summarized duration; no audio decoding or preprocessing."""
import csv
import json
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def main():
    with (ROOT / "all_metadata_drive.csv").open(encoding="utf-8-sig", newline="") as f:
        metadata = {row["file_id"]: row for row in csv.DictReader(f)}
    results = []
    for path in sorted((ROOT / "json_samples").glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        text = str(obj.get("Transcript","")).strip()
        if "/" in text:
            mode, parts = "slash", text.split("/")
        elif re.search(r"[.?!]", text):
            mode, parts = "punctuation", re.split(r"[.?!]", text)
        else:
            mode, parts = "whitespace", text.split()
        parts = [p.strip() for p in parts if p.strip()]
        wav = obj["File_id"]
        seconds = float(obj["playTime"])
        summary_seconds = float(metadata[wav]["play_time"])
        assert abs(seconds-summary_seconds) < 1e-6
        result = dict(file=path.name, file_id=wav, duration_sec=seconds,
                      duration_min=seconds/60, sampling_rate=obj["Meta_info"]["SamplingRate"],
                      disease_info=obj["Disease_info"], patient_info=obj["Patient_info"],
                      test_info=obj.get("Test_info"), transcript_chars=len(text),
                      text_split_mode=mode, text_segment_count=len(parts), summary_seconds=summary_seconds)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    lke = [json.loads(p.read_text(encoding="utf-8-sig")) for p in sorted((ROOT/"json_samples").glob("*LKE*.json"))]
    if len(lke)==2:
        print("LKE same Transcript:", lke[0]["Transcript"] == lke[1]["Transcript"],
              "same playTime:",lke[0]["playTime"]==lke[1]["playTime"],
              "(not proof of duplicate audio)")
    (ROOT/"json_sample_audit.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__ == "__main__":
    main()

