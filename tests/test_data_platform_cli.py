from __future__ import annotations

import json

import pytest

from MOTE.cli.bind_training_release import build_parser as build_bind_parser
from MOTE.cli.build_clean_release import main as build_clean_main
from MOTE.cli.cache_hf_dataset import main as cache_hf_main


def test_clean_release_cli_builds_without_tokenizer_arguments(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"text": "hello governed data"}) + "\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "format": "fitmotn_data_registry_v1",
                "pipeline_version": "clean-v1",
                "sources": [
                    {
                        "source_name": "demo",
                        "kind": "jsonl",
                        "path": str(source),
                        "split": "train",
                        "revision": "fixed-commit",
                        "data_plane": "pretraining",
                        "adapter": "text",
                        "fields": {"text": "text"},
                        "license": "apache-2.0",
                        "allowed_uses": ["training"],
                        "acquired_at": "2026-07-29T00:00:00Z"
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert build_clean_main(["--registry", str(registry), "--output_dir", str(tmp_path / "clean")]) == 0


def test_bind_cli_exposes_planes_buckets_and_safe_packing_policy():
    args = build_bind_parser().parse_args(
        [
            "--clean_release_dir", "clean", "--output_dir", "bound", "--tokenizer", "model",
            "--tokenizer_revision", "commit", "--max_length", "1280", "--planes", "pretraining",
            "--length_buckets", "256,512,1280", "--packing_mode", "pretrain_greedy",
            "--workers", "24", "--prefetch_factor", "3"
        ]
    )
    assert args.packing_mode == "pretrain_greedy"
    assert args.max_length == 1280
    assert args.workers == 24
    assert args.prefetch_factor == 3


def test_hf_cache_cli_rejects_vague_revision_before_network_access(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        cache_hf_main(
            [
                "--dataset", "demo/data", "--revision", "main", "--output_dir", str(tmp_path / "cache"),
                "--license", "apache-2.0", "--allowed_use", "training"
            ]
        )
