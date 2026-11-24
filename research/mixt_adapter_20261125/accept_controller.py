"""One real controller update with a synthetic frozen acceptance dataset."""
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer
from fitmotn.config import load_config
from fitmotn.data.release import build_frozen_sft_release
from fitmotn.train.controller import run_fitmotn_training

torch.set_num_threads(8)
source, imported, output = map(lambda s: Path(s).resolve(), sys.argv[1:4])
output.mkdir(parents=True, exist_ok=True)
release = output / "synthetic_acceptance_release"
tok = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=True)
if not release.exists():
    build_frozen_sft_release([dict(source_name="adapter_acceptance", source_revision="v1",
        source_split="test", source_row_id="1", prompt="1+1=", target="2")], release,
        tokenizer=tok, max_length=16, pipeline_version="adapter_acceptance_v1")
config = json.loads((ROOT / "fitmotn_config.pretrained_mixt.example.json").read_text(encoding="utf-8"))
config["model"].update(model_path=str(source), device="cpu", use_amp=False)
config["data"].update(seq_len_run=16, frozen_sft_release_dir=str(release))
config["train"].update(resume_weights_from=str(imported), steps=1, extra_updates=1,
    grad_accum=1, benchmark_train_only=True, save_every_updates=1, log_every=1,
    eval_every_updates=0, early_stop=False)
config["eval"] = {"run_baseline_eval": False}
config["output"].update(root_dir=str(output), run_name="controller_acceptance")
path = output / "acceptance_config.json"
path.write_text(json.dumps(config, indent=2), encoding="utf-8")
run_fitmotn_training(load_config(path))
