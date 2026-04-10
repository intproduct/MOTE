from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    model_path: str = "/work/home/sugang2025/qxfang/models/Qwen3-0.6B"
    layers_to_patch: str = "last_quarter"
    device: str = "cuda:0"
    use_amp: bool = True
    torch_dtype: str = "auto"
    trust_remote_code: bool = True

    E: int = 16
    d: int = 2
    k_in: int = 5
    topk: int = 4
    gate_type: str = "topk"
    temperature: float = 1.0
    capacity_factor: float = 1.35
    min_capacity: int = 8
    drop_tokens: bool = True
    drop_policy: str = "probs"
    aux_coeff: float = 3e-2
    zloss_coeff: float = 0.0
    jitter_eps: float = 0.0
    use_ste: bool = True
    pos_strategy: str = "random"
    init_gamma: float = 0.6
    warmup_ratio: float = 0.5
    warmup_stride: int = 1


@dataclass
class DataConfig:
    tok_shard_dir: str = "/work/home/sugang2025/qxfang/wiki24_tok"
    datas_dir: str = "/work/home/sugang2025/qxfang/Datas"
    seq_len_run: int = 1024
    dataloader_num_workers: int = 0

    fineweb_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/fineweb_sample10bt"
    code_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/the_stack_v2"
    gsm8k_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/gsm8k_main"
    gsm8k_socratic_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/gsm8k_socratic"
    svamp_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/svamp"
    metamath_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/metamathqa"
    mmlu_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/mmlu_all"
    math_cache_root: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/hendrycks_math"
    openr1_math_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/openr1_math_220k_default"
    numinamath_cot_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/numinamath_cot"
    openthoughts_math_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/openthoughts_114k_math"
    bespoke_stratos_cache_path: str = "/work/home/sugang2025/qxfang/Datas/hf_cache/bespoke_stratos_17k"

    fineweb_hf_name: str = "HuggingFaceFW/fineweb"
    fineweb_hf_config: str = "sample-10BT"
    fineweb_text_field: str = "text"

    code_hf_name: str = "bigcode/the-stack-v2"
    code_hf_config: str = ""
    code_text_field: str = "content"

    gsm8k_hf_name: str = "openai/gsm8k"
    gsm8k_hf_config: str = "main"
    gsm8k_socratic_hf_name: str = "openai/gsm8k"
    gsm8k_socratic_hf_config: str = "socratic"

    svamp_hf_name: str = "garrethlee/svamp"
    svamp_hf_config: str = ""

    metamath_hf_name: str = "meta-math/MetaMathQA"
    metamath_hf_config: str = ""
    metamath_max_samples: int = 20000

    mmlu_hf_name: str = "cais/mmlu"
    mmlu_hf_config: str = "all"
    mmlu_split: str = "dev"
    mmlu_train_split: str = "auxiliary_train"

    math_hf_name: str = "EleutherAI/hendrycks_math"
    math_split: str = "train"
    openr1_math_hf_name: str = "open-r1/OpenR1-Math-220k"
    openr1_math_hf_config: str = "default"
    numinamath_cot_hf_name: str = "AI-MO/NuminaMath-CoT"
    numinamath_cot_hf_config: str = ""
    openthoughts_math_hf_name: str = "open-r1/OpenThoughts-114k-math"
    openthoughts_math_hf_config: str = ""
    bespoke_stratos_hf_name: str = "bespokelabs/Bespoke-Stratos-17k"
    bespoke_stratos_hf_config: str = "default"

    use_wiki_local: bool = True
    use_fineweb: bool = True
    use_code: bool = True
    use_gsm8k_train: bool = True
    use_gsm8k_socratic_train: bool = False
    use_svamp_train: bool = False
    use_synthetic_arithmetic_train: bool = False
    use_metamath_train: bool = False
    use_math_train: bool = True
    use_mmlu_train: bool = False
    use_openr1_math: bool = False
    use_numinamath_cot: bool = False
    use_openthoughts_math: bool = False
    use_bespoke_stratos: bool = False

    wt_wiki: float = 1.0
    wt_fineweb: float = 1.0
    wt_code: float = 1.0
    wt_gsm8k: float = 1.0
    wt_gsm8k_socratic: float = 1.0
    wt_svamp: float = 0.8
    wt_synthetic_arithmetic: float = 1.0
    wt_metamath: float = 0.3
    wt_math: float = 1.0
    wt_mmlu: float = 1.0
    wt_openr1_math: float = 3.0
    wt_numinamath_cot: float = 2.2
    wt_openthoughts_math: float = 0.7
    wt_bespoke_stratos: float = 0.35

    reasoning_max_chars: int = 12000
    reasoning_max_approx_tokens: int = 3000
    openthoughts_max_chars: int = 7000
    openthoughts_max_approx_tokens: int = 1800
    synthetic_arithmetic_num_samples: int = 80000
    synthetic_arithmetic_seed: int = 42
    prefer_short_reasoning: bool = True
    skip_overlong_reasoning_samples: bool = True
    reasoning_supervision_mode: str = "answer_only"


@dataclass
class ApproxInitConfig:
    enabled: bool = False
    mode: str = "identity"
    subset_mode: str = "warmup_sliding_only"
    steps_per_proj: int = 300
    batch_size: int = 256
    lr: float = 3e-4
    use_fp32: bool = True
    early_stop_patience: int = 30
    early_stop_min_delta: float = 1e-6
    target_rel_l2: Optional[float] = None
    target_cos: Optional[float] = None
    identity_chunk_size: int = 256
    save_metrics: bool = True
    metrics_jsonl_name: str = "approx_init.jsonl"
    summary_json_name: str = "approx_init_summary.json"
    temp_gate_type: str = "softmax"
    temp_temperature: float = 1.5
    temp_drop_tokens: bool = False
    temp_aux_coeff: float = 0.0
    temp_zloss_coeff: float = 0.0
    fallback_hook_collect_batches: int = 8
    fallback_hook_max_len: int = 256
    init_random_std_scale: float = 0.3
    expert_warmup_scaling_enabled: bool = False
    random_expert_init_scale: float = 0.1
    random_expert_warmup_steps: int = 500
    random_expert_warmup_schedule: str = "linear"


@dataclass
class TrainConfig:
    batch_size: int = 4
    grad_accum: int = 1
    steps: int = 4000
    epochs: float = 1.0
    epoch_samples: int = 200_000
    lr: float = 3e-5
    log_every: int = 50
    save_every_updates: int = 1000
    eval_every_updates: int = 1000
    max_grad_norm: float = 1.0

    stage_a_ratio: float = 0.60
    stage_a_pretrain_ratio: float = 0.70
    stage_a_task_ratio: float = 0.30
    task_bucket_mode: str = "flat"
    stage_a_core_task_ratio: float = 0.0
    stage_a_aux_task_ratio: float = 0.0
    stage_b_mode: str = "mixed"
    stage_b_pretrain_ratio: float = 0.45
    stage_b_task_ratio: float = 0.55
    stage_b_core_task_ratio: float = 0.0
    stage_b_aux_task_ratio: float = 0.0
    stage_b_disable_pretrain: bool = False
    stage_b_reasoning_boost: float = 1.0

    begin_t: float = 1.5
    end_t: float = 0.8
    gate_freeze_steps: int = 2000
    usage_dump_every: int = 500
    usage_light_every: Optional[int] = None
    usage_light_jsonl_every: Optional[int] = None
    usage_report_every: Optional[int] = None
    heavy_log_every: int = 500
    enable_usage_runtime_tracking: Optional[bool] = None
    enable_usage_report: Optional[bool] = None
    enable_heavy_runtime_stats: Optional[bool] = None
    enable_grad_param_norm: Optional[bool] = None
    enable_cuda_snapshot: Optional[bool] = None
    train_jsonl_every: Optional[int] = None
    benchmark_train_only: bool = False

    lr_warmup: bool = False
    lr_warmup_steps: int = 1000
    warmup_ratio: float = 0.5

    early_stop: bool = True
    early_stop_abs_gsm8k: float = 0.45
    early_stop_abs_mmlu: float = 0.45

    seed: int = 0
    report_to: List[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    run_baseline_eval: bool = True
    lm_eval_batch_size: int = 1
    lm_eval_num_fewshot_mmlu: int = 5
    lm_eval_num_fewshot_gsm8k: int = 8
    lm_eval_num_fewshot_math: int = 4
    lm_eval_device: str = "cuda:0"

    baseline_small_tasks: List[str] = field(default_factory=lambda: ["gsm8k", "mmlu"])
    final_tasks: List[str] = field(default_factory=lambda: ["gsm8k", "mmlu"])

    early_limit_gsm8k: int = 32
    early_limit_mmlu: int = 64
    baseline_small_limit_gsm8k: int = 64
    baseline_small_limit_mmlu: int = 128
    final_limit_gsm8k: int = 0
    final_limit_mmlu: int = 512
    final_limit_math: int = 256

    early_max_gen_toks_gsm8k: int = 256
    final_max_gen_toks_gsm8k: int = 256
    final_max_gen_toks_math: int = 256


@dataclass
class OutputConfig:
    root_dir: str = "./fitmotn_outputs"
    run_name: Optional[str] = None
    overwrite_output_dir: bool = False


@dataclass
class FitMoTNConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    approx_init: ApproxInitConfig = field(default_factory=ApproxInitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
