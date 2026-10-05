"""Read saved bounded Drive metadata samples; never downloads audio or preprocesses."""
import base64
import csv
import io
import json
from pathlib import Path

def main():
    for path in sorted(Path(__file__).parent.glob("sample_*.json")):
        sample = json.loads(path.read_text(encoding="utf-8-sig"))
        print("\nSample:", sample["label"], sample["title"], sample["url"])
        if sample["mime"] == "application/json":
            obj = json.loads(base64.b64decode(sample["b64"]).decode("utf-8-sig"))
            print("Top-level keys:", ", ".join(obj))
            for key, value in obj.items():
                if isinstance(value, dict):
                    print(key + ":", json.dumps(value, ensure_ascii=False))
            print("File_id:", obj.get("File_id"), "playTime:", obj.get("playTime"))
        elif sample["mime"] == "text/csv":
            reader = csv.reader(io.StringIO(sample["text"].lstrip("\ufeff")))
            print("Columns:", next(reader))
            print("First row:", next(reader, []))
            print("Bounded header/sample only; no row totals inferred.")

if __name__ == "__main__":
    main()

