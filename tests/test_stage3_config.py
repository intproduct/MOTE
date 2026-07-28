from __future__ import annotations

import json
from pathlib import Path

from MOTE.config.loader import load_config_from_json


ROOT = Path(__file__).resolve().parents[1]


def test_stage3_single_card_config_is_frozen_strict_bounded_and_no_code(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    cfg = load_config_from_json(ROOT / "configs" / "fitmotn_k8_nodecay_global_8e_sft_v2_rc1.json")
    assert cfg.data.seq_len_run == 1280
    assert cfg.data.use_frozen_sft_release is True
    assert cfg.data.frozen_sft_exclusive is True
    assert cfg.data.source_sampling_mode == "deterministic_strict"
    assert cfg.data.source_max_epochs == 1
    assert cfg.data.dataloader_num_workers == 0
    assert cfg.data.fail_on_dynamic_skip is True
    assert cfg.data.reasoning_format == "raw"
    assert cfg.data.use_code is False
    assert cfg.data.use_wiki_local is False
    assert cfg.data.use_fineweb is False
    assert "no_code" in cfg.output.run_name


def test_source_manifest_lists_all_required_adapters_and_no_code_plane():
    manifest = json.loads((ROOT / "configs" / "sft_v2_rc1_sources.example.json").read_text(encoding="utf-8"))
    adapters = {source["adapter"] for source in manifest["sources"]}
    assert adapters == {
        "gsm8k_main",
        "gsm8k_socratic",
        "svamp",
        "synthetic_arithmetic",
        "hendrycks_math",
        "metamath",
        "openr1",
        "numinamath",
        "openthoughts",
    }
    assert manifest["pretrain_data_plane"]["the_stack"]["status"] == "disabled"
