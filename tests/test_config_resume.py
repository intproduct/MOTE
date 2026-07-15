from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if "MOTE" not in sys.modules:
    mote_pkg = types.ModuleType("MOTE")
    mote_pkg.__path__ = [str(ROOT)]
    sys.modules["MOTE"] = mote_pkg
if "MOTE.config" not in sys.modules:
    config_pkg = types.ModuleType("MOTE.config")
    config_pkg.__path__ = [str(ROOT / "config")]
    sys.modules["MOTE.config"] = config_pkg

from MOTE.config.loader import load_config, load_config_from_json
from MOTE.config.schema import RLConfig


BASE_PORTABLE_PAYLOAD = {
    "model": {"model_path": "${MODEL_ROOT}/Qwen3-0.6B"},
    "data": {
        "tok_shard_dir": "${DATA_ROOT}/wiki24_tok",
        "datas_dir": "${DATA_ROOT}",
        "fineweb_cache_path": "${CACHE_ROOT}/fineweb_sample10bt",
        "code_cache_path": "${CACHE_ROOT}/the_stack_v2",
        "gsm8k_cache_path": "${CACHE_ROOT}/gsm8k_main",
        "math_cache_root": "${CACHE_ROOT}/hendrycks_math",
    },
    "output": {"root_dir": "${OUTPUT_ROOT}/fitmotn_runs"},
}


def with_base(payload):
    merged = deepcopy(BASE_PORTABLE_PAYLOAD)
    for section, values in payload.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section].update(values)
        else:
            merged[section] = values
    return merged


class ConfigResumeTests(unittest.TestCase):
    def setUp(self):
        self._old_env = {key: os.environ.get(key) for key in ["MODEL_ROOT", "DATA_ROOT", "OUTPUT_ROOT", "CACHE_ROOT", "PROJECT_ROOT"]}
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        os.environ["MODEL_ROOT"] = str(root / "models")
        os.environ["DATA_ROOT"] = str(root / "data")
        os.environ["OUTPUT_ROOT"] = str(root / "outputs")
        os.environ["CACHE_ROOT"] = str(root / "cache")
        os.environ["PROJECT_ROOT"] = str(root / "project")

    def tearDown(self):
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def test_old_json_uses_schema_default_with_warning_event(self):
        payload = {"train": {"lr_scheduler_type": "constant"}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            cfg = load_config_from_json(path)

        events = getattr(cfg, "_path_audit_events", [])
        self.assertTrue(any(event["key"] == "model.model_path" and event["source"] == "default" and event["severity"] == "warning" for event in events))

    def test_json_without_resume_fields_loads_when_paths_are_configured(self):
        payload = with_base({"train": {"lr_scheduler_type": "constant"}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            cfg = load_config_from_json(path)

        self.assertIsNone(cfg.train.resume_fitmotn_from)
        self.assertIsNone(cfg.train.resume_weights_from)
        self.assertIsNone(cfg.train.resume_checkpoint_from)
        self.assertEqual(cfg.train.checkpoint_keep_last_n, 3)
        self.assertTrue(cfg.train.save_on_stage_transition)
        self.assertFalse(cfg.rl.enabled)
        self.assertEqual(cfg.rl.mode, "gsm8k_grpo")
        self.assertEqual(cfg.train.resume_stage, "auto")
        self.assertIsNone(cfg.train.extra_updates)
        self.assertTrue(cfg.train.stage2_only_on_resume)
        self.assertFalse(cfg.rl.enable_usage_tracking)
        self.assertEqual(cfg.data.reasoning_format, "raw")
        self.assertFalse(cfg.data.reasoning_chat_enable_thinking)
        self.assertIsNone(cfg.data.reasoning_chat_system_prompt)
        self.assertTrue(cfg.data.reasoning_chat_use_generation_prompt_for_labels)
        self.assertEqual(cfg.rl.prompt_format, "raw")
        self.assertFalse(cfg.rl.chat_enable_thinking)
        self.assertIsNone(cfg.rl.chat_system_prompt)
        self.assertEqual(cfg.rl.log_memory_every, 1)
        self.assertEqual(cfg.rl.logprob_micro_batch_size, 1)
        self.assertEqual(cfg.rl.rollout_micro_batch_size, 0)
        self.assertTrue(cfg.rl.rollout_use_cache)
        self.assertTrue(cfg.rl.rollout_inference_mode)
        self.assertTrue(cfg.rl.rollout_log_timing)
        self.assertEqual(cfg.rl.rollout_max_prompt_tokens, 0)
        self.assertFalse(cfg.rl.gradient_checkpointing)
        self.assertEqual(cfg.rl.empty_cache_every, 0)
        self.assertTrue(cfg.rl.skip_zero_advantage_updates)
        self.assertEqual(cfg.rl.max_zero_advantage_rollout_retries, 8)
        self.assertEqual(cfg.rl.zero_advantage_retry_action, "warn_continue")
        self.assertIsNone(cfg.rl.resume_weights_from)
        self.assertIsNone(cfg.rl.resume_checkpoint_from)
        self.assertEqual(cfg.rl.checkpoint_keep_last_n, 3)

    def test_legacy_resume_aliases_weight_resume(self):
        payload = with_base({"train": {"resume_fitmotn_from": "checkpoint-1"}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config_from_json(path)

        self.assertEqual(cfg.train.resume_fitmotn_from, cfg.train.resume_weights_from)
        self.assertIsNone(cfg.train.resume_checkpoint_from)

    def test_exact_and_weight_resume_are_mutually_exclusive(self):
        cases = [
            {
                "train": {
                    "resume_weights_from": "checkpoint-1",
                    "resume_checkpoint_from": "checkpoint-2",
                }
            },
            {
                "rl": {
                    "resume_weights_from": "checkpoint-1",
                    "resume_checkpoint_from": "checkpoint-2",
                }
            },
        ]
        for payload in cases:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "config.json"
                path.write_text(json.dumps(with_base(payload)), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                    load_config_from_json(path)

    def test_checkpoint_policy_fields_validate(self):
        cases = [
            ({"train": {"save_every_updates": 0}}, "train.save_every_updates"),
            ({"train": {"checkpoint_keep_last_n": -1}}, "retention"),
            (
                {"rl": {"enabled": True, "max_steps": 1, "checkpoint_temp_max_age_sec": -1}},
                "checkpoint_temp_max_age_sec",
            ),
        ]
        for payload, message in cases:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "config.json"
                path.write_text(json.dumps(with_base(payload)), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_config_from_json(path)

    def test_invalid_resume_stage_raises(self):
        payload = with_base({"train": {"resume_stage": "stage_c"}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "train.resume_stage must be one of"):
                load_config_from_json(path)

    def test_explicit_model_keys_are_tracked(self):
        payload = with_base({"model": {"k_in": 7}, "train": {"extra_updates": 10}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            cfg = load_config_from_json(path)

        self.assertIn("k_in", getattr(cfg, "_explicit_model_keys"))
        self.assertIn("extra_updates", getattr(cfg, "_explicit_train_keys"))

    def test_rl_config_parses_and_roundtrips(self):
        rl_payload = asdict(RLConfig(enabled=True, max_steps=3, trainable_mode="motn_only", train_json="train.jsonl"))
        payload = with_base({"train": {"lr_scheduler_type": "constant"}, "rl": rl_payload})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            cfg = load_config_from_json(path)
            roundtrip_path = Path(tmpdir) / "roundtrip.json"
            roundtrip_path.write_text(json.dumps(with_base({"rl": asdict(cfg.rl), "train": {"lr_scheduler_type": "constant"}})), encoding="utf-8")
            roundtrip = load_config_from_json(roundtrip_path)

        self.assertTrue(cfg.rl.enabled)
        self.assertEqual(cfg.rl.max_steps, 3)
        self.assertEqual(cfg.rl.trainable_mode, "motn_only")
        self.assertEqual(asdict(roundtrip.rl), asdict(cfg.rl))

    def test_rl_memory_fields_parse_and_validate(self):
        payload = with_base({
            "train": {"lr_scheduler_type": "constant"},
            "rl": {
                "enabled": True,
                "max_steps": 1,
                "train_json": "train.jsonl",
                "enable_usage_tracking": True,
                "log_memory_every": 0,
                "logprob_micro_batch_size": 2,
                "rollout_micro_batch_size": 0,
                "rollout_use_cache": False,
                "rollout_inference_mode": False,
                "rollout_log_timing": False,
                "rollout_max_prompt_tokens": 128,
                "gradient_checkpointing": True,
                "empty_cache_every": 3,
                "skip_zero_advantage_updates": False,
                "max_zero_advantage_rollout_retries": 5,
                "zero_advantage_retry_action": "raise",
            },
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config_from_json(path)

        self.assertTrue(cfg.rl.enable_usage_tracking)
        self.assertEqual(cfg.rl.log_memory_every, 0)
        self.assertEqual(cfg.rl.logprob_micro_batch_size, 2)
        self.assertEqual(cfg.rl.rollout_micro_batch_size, 0)
        self.assertFalse(cfg.rl.rollout_use_cache)
        self.assertFalse(cfg.rl.rollout_inference_mode)
        self.assertFalse(cfg.rl.rollout_log_timing)
        self.assertEqual(cfg.rl.rollout_max_prompt_tokens, 128)
        self.assertTrue(cfg.rl.gradient_checkpointing)
        self.assertEqual(cfg.rl.empty_cache_every, 3)
        self.assertFalse(cfg.rl.skip_zero_advantage_updates)
        self.assertEqual(cfg.rl.max_zero_advantage_rollout_retries, 5)
        self.assertEqual(cfg.rl.zero_advantage_retry_action, "raise")

    def test_chat_format_fields_parse_and_validate(self):
        payload = with_base({
            "train": {"lr_scheduler_type": "constant"},
            "data": {
                "reasoning_format": "chat",
                "reasoning_chat_enable_thinking": True,
                "reasoning_chat_system_prompt": "sys",
                "reasoning_chat_use_generation_prompt_for_labels": False,
            },
            "rl": {
                "enabled": True,
                "max_steps": 1,
                "train_json": "train.jsonl",
                "prompt_format": "chat",
                "chat_enable_thinking": True,
                "chat_system_prompt": "rl sys",
            },
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config_from_json(path)

        self.assertEqual(cfg.data.reasoning_format, "chat")
        self.assertTrue(cfg.data.reasoning_chat_enable_thinking)
        self.assertEqual(cfg.data.reasoning_chat_system_prompt, "sys")
        self.assertFalse(cfg.data.reasoning_chat_use_generation_prompt_for_labels)
        self.assertEqual(cfg.rl.prompt_format, "chat")
        self.assertTrue(cfg.rl.chat_enable_thinking)
        self.assertEqual(cfg.rl.chat_system_prompt, "rl sys")

    def test_invalid_rl_config_raises(self):
        cases = [
            ({"rl": {"enabled": True, "mode": "other", "max_steps": 1}}, "rl.mode must be one of"),
            ({"rl": {"enabled": True, "trainable_mode": "name_match", "max_steps": 1}}, "rl.trainable_mode must be one of"),
            ({"rl": {"enabled": False, "run_after_sft": True}}, "rl.run_after_sft=true requires rl.enabled=true"),
            (
                {"rl": {"enabled": True, "run_after_sft": True, "max_steps": 1, "resume_weights_from": "checkpoint-1"}},
                "cannot be combined with an RL resume checkpoint",
            ),
            ({"rl": {"enabled": True, "max_steps": 0}}, "rl.max_steps must be > 0"),
            ({"rl": {"enabled": True, "max_steps": 1, "log_memory_every": -1}}, "rl.log_memory_every must be >= 0"),
            ({"rl": {"enabled": True, "max_steps": 1, "logprob_micro_batch_size": -1}}, "rl.logprob_micro_batch_size must be >= 0"),
            ({"rl": {"enabled": True, "max_steps": 1, "rollout_micro_batch_size": -1}}, "rl.rollout_micro_batch_size must be >= 0"),
            ({"rl": {"enabled": True, "max_steps": 1, "rollout_max_prompt_tokens": -1}}, "rl.rollout_max_prompt_tokens must be >= 0"),
            ({"rl": {"enabled": True, "max_steps": 1, "empty_cache_every": -1}}, "rl.empty_cache_every must be >= 0"),
            ({"rl": {"enabled": True, "max_steps": 1, "max_zero_advantage_rollout_retries": 0}}, "rl.max_zero_advantage_rollout_retries must be >= 1"),
            ({"rl": {"enabled": True, "max_steps": 1, "zero_advantage_retry_action": "stop"}}, "rl.zero_advantage_retry_action must be one of"),
            ({"data": {"reasoning_format": "xml"}}, "data.reasoning_format must be one of"),
            ({"rl": {"enabled": True, "max_steps": 1, "prompt_format": "xml"}}, "rl.prompt_format must be one of"),
        ]
        for payload, message in cases:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / "config.json"
                    path.write_text(json.dumps(with_base(payload)), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_config_from_json(path)

    def test_env_paths_and_optional_resume_paths_resolve(self):
        payload = with_base({
            "train": {"resume_fitmotn_from": "relative_sft_ckpt"},
            "rl": {"resume_from": "${OUTPUT_ROOT}/resume", "train_json": "data/train.jsonl"},
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config_from_json(path)

        self.assertTrue(Path(cfg.model.model_path).is_absolute())
        self.assertEqual(cfg.model.model_path, str(Path(os.environ["MODEL_ROOT"]).resolve() / "Qwen3-0.6B"))
        self.assertEqual(cfg.output.root_dir, str(Path(os.environ["OUTPUT_ROOT"]).resolve() / "fitmotn_runs"))
        self.assertEqual(cfg.rl.resume_from, str(Path(os.environ["OUTPUT_ROOT"]).resolve() / "resume"))
        self.assertEqual(cfg.train.resume_fitmotn_from, str(Path(os.environ["PROJECT_ROOT"]).resolve() / "relative_sft_ckpt"))
        self.assertEqual(cfg.rl.train_json, str(Path(os.environ["PROJECT_ROOT"]).resolve() / "data/train.jsonl"))

    def test_work_machine_path_warns_but_loads(self):
        legacy_model_path = "/" + "work" + "/project/models/Qwen3-0.6B"
        payload = with_base({"model": {"model_path": legacy_model_path}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            cfg = load_config_from_json(path)

        self.assertEqual(cfg.model.model_path, legacy_model_path)
        self.assertTrue(any(event["key"] == "model.model_path" and event["severity"] == "warning" for event in getattr(cfg, "_path_audit_events", [])))

    def test_home_machine_path_raises(self):
        forbidden_model_path = "/" + "home" + "/user/models/Qwen3-0.6B"
        payload = with_base({"model": {"model_path": forbidden_model_path}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsafe path"):
                load_config_from_json(path)

    def test_alias_config_resolves_model_and_gsm8k_dataset(self):
        payload = {
            "model_alias": "qwen3_8b",
            "dataset_alias": "gsm8k",
            "output": {"root_dir": "${OUTPUT_ROOT}/alias_run"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg = load_config_from_json(path)

        self.assertEqual(cfg.model.model_path, str(Path(os.environ["MODEL_ROOT"]).resolve() / "Qwen3-8B"))
        self.assertEqual(cfg.data.gsm8k_cache_path, str(Path(os.environ["CACHE_ROOT"]).resolve() / "gsm8k_main"))
        self.assertFalse(cfg.data.use_fineweb)
        self.assertFalse(cfg.data.use_code)

    def test_debug_audit_records_config_override_and_alias_sources(self):
        old_debug = os.environ.get("FITMOTN_PATH_DEBUG")
        os.environ["FITMOTN_PATH_DEBUG"] = "1"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "config.json"
                path.write_text(
                    json.dumps({
                        "model_alias": "qwen3_8b",
                        "dataset_alias": "gsm8k",
                        "output": {"root_dir": "${OUTPUT_ROOT}/from_config"},
                    }),
                    encoding="utf-8",
                )
                cfg = load_config(path, overrides={"output": {"root_dir": "${OUTPUT_ROOT}/from_override"}})
        finally:
            if old_debug is None:
                os.environ.pop("FITMOTN_PATH_DEBUG", None)
            else:
                os.environ["FITMOTN_PATH_DEBUG"] = old_debug

        events = getattr(cfg, "_path_audit_events", [])
        self.assertTrue(any(event["key"] == "model.model_path" and event["source"] == "alias" for event in events))
        self.assertTrue(any(event["key"] == "data.gsm8k_cache_path" and event["source"] == "alias" for event in events))
        self.assertTrue(any(event["key"] == "output.root_dir" and event["source"] == "override" for event in events))


if __name__ == "__main__":
    unittest.main()
