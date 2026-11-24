"""Real single-GPU native vLLM parity probe in the isolated WSL environment."""
import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path

def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    torch.set_num_threads(8)
    start = time.monotonic()
    report = dict(torch=torch.__version__, transformers=transformers.__version__, tensor_parallel_size=1,
                  isolated_nvml_fallback=True, stages={})
    def save():
        report["elapsed_seconds"] = time.monotonic() - start
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True, local_files_only=True)
        prompts = ["1+1=", "Explain why the sum of two even integers is even."]
        token_prompts = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        baseline = AutoModelForCausalLM.from_pretrained(args.source, trust_remote_code=True, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
        refs = []
        with torch.no_grad():
            for tokens in token_prompts:
                ids = torch.tensor([tokens], device="cuda:0")
                generated = baseline.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True,
                                              pad_token_id=tokenizer.pad_token_id)
                refs.append(generated[0, len(tokens):].tolist())
        del baseline, ids, generated
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        report["reference_tokens"] = refs
        report["stages"]["reference"] = True
        save()
        import vllm
        from vllm import LLM, SamplingParams
        from fitmotn.rl.vllm_qwen35_native import register_qwen35_text_native
        register_qwen35_text_native()
        report["vllm"] = vllm.__version__
        save()
        engine = LLM(model=args.export, tokenizer=args.export, trust_remote_code=True, dtype="bfloat16",
            tensor_parallel_size=1, gpu_memory_utilization=.75, max_model_len=4096, max_num_seqs=2,
            max_num_batched_tokens=512, enforce_eager=True, enable_prefix_caching=False,
            disable_log_stats=True, seed=0, model_impl="auto")
        report["stages"]["engine_loaded"] = True
        save()
        sampling = SamplingParams(temperature=0.0, max_tokens=8)
        data = [{"prompt_token_ids": tokens} for tokens in token_prompts]
        tick = time.monotonic()
        outputs = engine.generate(data, sampling)
        report["generation_wall_seconds"] = time.monotonic() - tick
        actual = [list(output.outputs[0].token_ids) for output in outputs]
        repeated = engine.generate(data, sampling)
        singles = [list(engine.generate([prompt], sampling)[0].outputs[0].token_ids) for prompt in data]
        report["vllm_tokens"] = actual
        report["first_token_matches"] = [a[0] == b[0] for a, b in zip(actual, refs)]
        report["all_tokens_match"] = actual == refs
        report["repeated_generation_identical"] = actual == [list(output.outputs[0].token_ids) for output in repeated]
        report["single_sequence_matches_batch"] = singles == actual
        report["prompt_tokens_preserved"] = [list(output.prompt_token_ids) == prompt for output, prompt in zip(outputs, token_prompts)]
        report["initialization_and_generation_ok"] = True
        report["ok"] = report["all_tokens_match"] and report["single_sequence_matches_batch"] and report["repeated_generation_identical"] and all(report["prompt_tokens_preserved"])
        save()
        assert report["ok"], report
    except Exception as exc:
        report.update(ok=False, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save()

if __name__ == "__main__":
    main()
