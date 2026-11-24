"""Import a trained partner MixT checkpoint as a weight-preserving MoTE run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..checkpointing import extract_patch_state_dict, save_fitmotn_metadata, transactional_save_checkpoint
from ..patching import patch_qwen_ffn_layers, set_trainable_patch_only
from ..pretrained_mixt import resolve_mixt_layers
from ..runtime import align_mixt_generation_config
from ..utils.files import sha256_file


def import_checkpoint(model_path, output_dir, *, device="cpu", dtype="bfloat16", gate_arch="linear",
                      train_scope="gate_only", verify_prompts=("1+1=",), seed=0):
    source, output = Path(model_path).resolve(), Path(output_dir).resolve()
    if output == source or source in output.parents:
        raise ValueError("Import output must be outside the immutable source checkpoint")
    if output.exists():
        raise FileExistsError(f"Import output already exists: {output}")
    torch.manual_seed(int(seed))
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, local_files_only=True)
    model, loading = AutoModelForCausalLM.from_pretrained(
        source, trust_remote_code=True, local_files_only=True, dtype=getattr(torch, dtype),
        output_loading_info=True, attn_implementation="eager")
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Source MixT did not load completely: {loading}")
    model.to(device).eval()
    align_mixt_generation_config(model, tokenizer)
    layers = resolve_mixt_layers(model, "mixt")
    probes = [tokenizer(prompt, return_tensors="pt").to(device) for prompt in verify_prompts]
    with torch.no_grad():
        expected = [model(**inputs, use_cache=False).logits.cpu() for inputs in probes]
    config = dict(patch_backend="pretrained_mixt", gate_type="softmax", gate_arch=gate_arch,
                  topk=4, E=4, d=2, k_in=8, drop_tokens=False, global_expert_enabled=False,
                  temperature=1.0, dtype=getattr(torch, dtype),
                  pretrained_mixt={"version": 1, "uniform_init": True, "train_scope": train_scope})
    patch_qwen_ffn_layers(model, layers, config, torch.device(device), dtype=getattr(torch, dtype))
    verification = []
    with torch.no_grad():
        for inputs, reference in zip(probes, expected):
            actual = model(**inputs, use_cache=False).logits.cpu()
            delta = (actual.float() - reference.float())
            row = {"tokens": int(inputs["input_ids"].numel()), "max_abs": float(delta.abs().max()),
                   "relative_l2": float(delta.norm() / reference.float().norm().clamp_min(1e-12))}
            # Uniform native routing is expected to preserve storage-precision
            # arithmetic; fail conversion if the full model changes at all.
            if not torch.equal(actual, reference):
                raise RuntimeError(f"Native MixT import changed initial model outputs: {row}")
            verification.append(row)
    set_trainable_patch_only(model)
    artifact_names = ["config.json", "mixt_manifest.json", "mixt_projector.py",
                      "modeling_qwen3_5_mixt.py", "tokenizer.json", "chat_template.jinja",
                      "generation_config.json", "tokenizer_config.json", "model.safetensors.index.json"]
    artifact_names.extend(path.name for path in sorted(source.glob("*.safetensors")))
    provenance = {name: sha256_file(str(source / name))
                  for name in artifact_names
                  if (source / name).is_file()}
    metadata = dict(base_model_path=str(source), layers_to_patch=layers,
                    patch_backend="pretrained_mixt", patch_cfg=config, motn_cfg=config,
                    patch_state_dict=extract_patch_state_dict(model, layers),
                    resolved_model_dtype=dtype, trust_remote_code=True,
                    attn_implementation="eager",
                    source_artifact_sha256=provenance, import_verification=verification,
                    import_seed=int(seed), optimizer_state_inherited=False,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    total_parameters=sum(p.numel() for p in model.parameters()))
    def writer(path):
        save_fitmotn_metadata(path, metadata)
        tokenizer.save_pretrained(path)
        model.generation_config.save_pretrained(path)
    transactional_save_checkpoint(output, writer, checkpoint_kind="weights_only", update_step=0,
                                  stage_name="native_mixt_import")
    return {"output": str(output), "layers": layers, "verification": verification,
            "trainable_parameters": metadata["trainable_parameters"],
            "total_parameters": metadata["total_parameters"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16")
    parser.add_argument("--gate-arch", choices=["linear", "mlp", "quantum"], default="linear")
    parser.add_argument("--train-scope", choices=["gate_only", "gate_experts", "all"], default="gate_only")
    parser.add_argument("--verify-prompt", action="append")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    result = import_checkpoint(args.model, args.output, device=args.device, dtype=args.dtype,
                               gate_arch=args.gate_arch, train_scope=args.train_scope,
                               verify_prompts=args.verify_prompt or ["1+1="], seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
