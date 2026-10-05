"""Static AST/syntax/import checks only: never loads or processes any audio."""
import ast
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "04_run_preprocessing_smoke_test.py"
source = TARGET.read_text(encoding="utf-8-sig")
compile(source, str(TARGET), "exec")
tree = ast.parse(source)

# Verify the numerical VAD implementation against the pinned reference.
reference = ROOT / "reference_code/Preprocess/[Distribution]Preprocessing_Stroke.ipynb"
nb = json.loads(reference.read_text(encoding="utf-8-sig"))
cell = "".join(nb["cells"][9]["source"])
reference_func = next(n for n in ast.parse(cell).body
                      if isinstance(n, ast.FunctionDef) and n.name == "vad_segment_by_energy")
actual_func = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "vad_segment_by_energy")

class IgnorePrint(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "print":
                return ast.Pass()
        return self.generic_visit(node)

assert ast.dump(IgnorePrint().visit(reference_func), include_attributes=False) == ast.dump(
    IgnorePrint().visit(actual_func), include_attributes=False), "VAD algorithm differs from reference"
spec = importlib.util.spec_from_file_location("smoke_test_static_import", TARGET)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # __name__ differs from __main__; no main() call.
assert module.VAD1 == dict(frame_ms=500, silence_duration_sec=5, alpha=0.3)
assert module.VAD2 == dict(frame_ms=600, silence_duration_sec=3, alpha=0.4)
with (ROOT.parents[1] / "results/preprocessing_pilot/smoke_test_manifest.csv").open(
        encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 3 and {r["speaker_id"] for r in rows} == set(module.EXPECTED)
for row in rows:
    group, wav = module.EXPECTED[row["speaker_id"]]
    assert row["disability_group"] == group
    assert Path(row["wav_path"]).name == wav
    assert Path(row["json_path"]).name == Path(wav).with_suffix(".json").name
    assert "EDIT_LOCAL_PATH" in row["wav_path"] and "EDIT_LOCAL_PATH" in row["json_path"]
assert set(["speaker_id", "disability_group", "recording_id", "audio_duration_sec",
            "text_segment_count", "vad1_audio_segment_count", "vad1_match",
            "vad2_audio_segment_count", "vad2_match", "final_status", "failure_reason",
            "valid_1_30s_segments", "total_segments", "valid_segment_ratio",
            "processing_time_sec", "implementation_safety_fix"]) <= set(module.RESULT_FIELDS)
print("PASS: syntax; pinned-reference VAD AST equivalence (print-only change); module import; manifest; result schema")
print("No WAV read, VAD call, transcript processing or real smoke-test run performed.")

