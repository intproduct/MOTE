"""The handoff must detect corrupt evidence and never advertise an unbuilt launcher."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("handoff_verifier", ROOT / "scripts/verify_quantum_handoff.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_evidence_detects_tampering_and_path_escape(tmp_path):
    artifact = tmp_path / "result.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = {"files": [{"path": "result.csv", "bytes": artifact.stat().st_size,
                           "sha256": module.file_hash(artifact)}]}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert not module.verify_evidence(tmp_path)["errors"]
    artifact.write_text("value\n2\n", encoding="utf-8")
    assert "mismatch" in module.verify_evidence(tmp_path)["errors"][0]
    manifest["files"][0]["path"] = "../escape.csv"
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert "Unsafe" in module.verify_evidence(tmp_path)["errors"][0]


def test_machine_plans_are_joint_and_have_unique_outputs():
    assert not module.verify_plans(ROOT / "research/qwen_mote_joint_handoff_20261030")["errors"]
