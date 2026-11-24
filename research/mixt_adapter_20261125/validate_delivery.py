"""Verify source provenance and the checkpoint emitted by the real controller."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from fitmotn.checkpointing import load_fitmotn_metadata, validate_checkpoint, extract_patch_state_dict
from fitmotn.eval.restore import restore_fitmotn_model
from fitmotn.utils.files import sha256_file

torch.set_num_threads(8)
source, imported, checkpoint = map(Path, sys.argv[1:4])
metadata = load_fitmotn_metadata(imported)
report = {"import_verification": metadata["import_verification"]}
assert all(sha256_file(str(source / name)) == expected
           for name, expected in metadata["source_artifact_sha256"].items())
report["source_artifacts_unchanged"] = True
report["source_weights_sha256"] = {p.name: sha256_file(str(p)) for p in source.glob("*.safetensors")}
validation = validate_checkpoint(checkpoint, verify_hashes=True, require_exact_resume="sft")
assert validation["ok"], validation
report["controller_checkpoint_integrity"] = validation
model, tokenizer, saved = restore_fitmotn_model(checkpoint, device="cpu")
actual_patch = extract_patch_state_dict(model, saved["layers_to_patch"])
assert set(actual_patch) == set(saved["patch_state_dict"])
assert all(torch.equal(actual_patch[name], value) for name, value in saved["patch_state_dict"].items())
report["trained_patch_reload_tensors_exact"] = True
report["updated_router_tensors"] = sum(not torch.equal(value, metadata["patch_state_dict"][name])
                                       for name, value in actual_patch.items() if ".gate." in name)
assert report["updated_router_tensors"] > 0
with torch.no_grad():
    logits = model(**tokenizer("1+1=", return_tensors="pt"), use_cache=False).logits
assert torch.isfinite(logits).all()
report["trained_checkpoint_forward_finite"] = True
report["ok"] = True
Path(__file__).with_name("delivery_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
