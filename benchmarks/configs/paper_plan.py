from typing import Dict, List


LLAMA31_MODEL_PATH = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
QWEN_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

PRIMARY_MODEL = "llama31"
PRIMARY_MODEL_PATH = LLAMA31_MODEL_PATH
SECONDARY_MODEL = "qwen25"
SECONDARY_MODEL_PATH = QWEN_MODEL_PATH

MODEL_ROLES: Dict[str, object] = {
    "primary": {
        "name": "Meta-Llama-3.1-8B-Instruct",
        "path": LLAMA31_MODEL_PATH,
        "reason": "ShadowKV officially supports Llama-3.1-8B, making related-system comparison cleaner.",
    },
    "secondary": {
        "name": "Qwen2.5-7B-Instruct",
        "path": QWEN_MODEL_PATH,
        "reason": "Keep the completed Qwen evidence as a cross-model robustness check instead of the only mainline.",
    },
}

P2_MAINLINE: Dict[str, object] = {
    "group": "p2_only_compress",
    "name": "P2 mainline",
    "variant": "varA",
    "retain_ratio": 0.10,
    "offload_budget_blocks_max": 320,
    "prefetch_budget_blocks_max": 320,
    "chunk_size": 512,
    "sink_len": 16,
    "snapkv_observation_len": 16,
    "p2_sink_tokens": 16,
    "p2_recent_tokens": 16,
    "p2_target_free_blocks": 0,
    "wm_high_ratio": 0.40,
    "decode_path_mode": "rebuild",
    "decode_paged_flash_enabled": False,
}

FORMAL_BASE_METHOD_ORDER: List[str] = ["vllm", "off_compress", "p2_only_compress"]
FORMAL_EXTENDED_METHOD_ORDER: List[str] = ["vllm", "vllm_offload", "off_compress", "p2_only_compress", "shadowkv"]
FORMAL_METHOD_ORDER: List[str] = FORMAL_BASE_METHOD_ORDER

FROZEN_INTERNAL_RUNTIME: Dict[str, object] = {
    "gpu_mem_frac": 0.20,
    "prefill_batch_size": 4,
    "chunk_size": 512,
}

FROZEN_VLLM_RUNTIME: Dict[str, object] = {
    "gpu_memory_utilization": 0.90,
}

FROZEN_VLLM_OFFLOAD_RUNTIME: Dict[str, object] = {
    "gpu_memory_utilization": 0.90,
    "swap_space_candidates": [16, 32],
    "cpu_offload_gb_candidates": [8, 12],
}

FROZEN_SINGLE_SEQ_ROWS: List[Dict[str, object]] = [
    {
        "method": "off_raw",
        "best_gpu_mem_frac": 0.38,
        "L_quality_max": 49000,
        "tokens_s_at_L_quality_max": 0.4578,
        "p95_ms_at_L_quality_max": 208.740,
        "source": "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/results/paper/off_raw_rerun_fixed/off_raw_memfrac_0.38.json",
    },
    {
        "method": "off_compress",
        "best_gpu_mem_frac": 0.12,
        "L_quality_max": 115000,
        "tokens_s_at_L_quality_max": 0.1739,
        "p95_ms_at_L_quality_max": 82.303,
        "source": "/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/results/paper/single_seq_qwen_neighbors_rerun_fixed/off_compress_memfrac_0.12_stepdown.json",
    },
]

APPENDIX_TABLES: List[Dict[str, str]] = [
    {
        "name": "historical_runs",
        "title": "Pre-fix historical runs",
        "purpose": "Appendix only; documents the search trajectory before runner fixes.",
    },
    {
        "name": "depth_table",
        "title": "Per-depth strict pass/fail table",
        "purpose": "Appendix only; shows depth=0.1/0.5/0.9 behavior near the boundary.",
    },
]

STATUS_DEFINITIONS: Dict[str, str] = {
    "Success": "valid completion rate = 1.0 and no OOM/runtime failure",
    "Degraded": "completion rate > 0 but valid completion rate < 1.0, without catastrophic system failure",
    "Failed/OOM": "completion rate = 0, or any run terminated by OOM/timeout/runtime failure such that the point is not stably serviceable",
}

VALID_DEFINITIONS: Dict[str, str] = {
    "synthetic_fixed_point": "request completed normally and output format is legal",
    "concurrency_frontier": "request completed normally and output format is legal",
    "sharegpt": "request completed normally and output format is legal",
    "passkey_under_load": "request completed normally and the target answer field is parseable",
    "longbench": "request completed normally and the task scoring script can score the sample",
}

METRIC_DEFINITIONS: Dict[str, str] = {
    "TTFT": "request arrival to first token emission",
    "ITL": "per-request average inter-token latency, aggregated across requests with p95/p99",
    "completion_rate": "fraction of requests that terminate normally",
    "valid_completion_rate": "fraction of requests that terminate normally and satisfy the phase-specific valid definition",
    "OOM_failure_count": "count of OOM, timeout, runtime failure, or other system-level failures",
    "min_free_blocks": "minimum free KV blocks observed during the run",
    "min_free_block_ratio": "min_free_blocks / N_total, stored in internal logs only",
    "wall_clock_total_runtime": "wall-clock runtime of the run, stored in internal logs only",
}

DRY_RUN_POINTS: List[Dict[str, int]] = [
    {"input_len": 32768, "concurrency": 16, "max_new_tokens": 1024},
    {"input_len": 8192, "concurrency": 64, "max_new_tokens": 1024},
]

RELATED_BASELINE_FEASIBILITY: Dict[str, object] = {
    "priority": ["shadowkv", "vllm_offload", "kvpr"],
    "gate": "join Phase 3/4 only if the baseline can stably run 32K x 16 x 1024 within one day",
    "points": [
        {"name": "32k_c16_n1024", "input_len": 32768, "concurrency": 16, "max_new_tokens": 1024},
        {"name": "32k_c64_n1024", "input_len": 32768, "concurrency": 64, "max_new_tokens": 1024},
    ],
    "shadowkv": {
        "isolation": "separate baseline directory/environment; do not modify the main conda environment",
        "blocks_mainline": False,
    },
    "vllm_offload": {
        "runtime": FROZEN_VLLM_OFFLOAD_RUNTIME,
        "blocks_mainline": False,
    },
    "kvpr": {
        "scope": "installation/model-support/32K-serving feasibility only",
        "blocks_mainline": False,
    },
}

LLAMA31_MAINLINE_SMOKE: Dict[str, object] = {
    "purpose": "Confirm Llama3.1 can replace Qwen2.5 as the primary system model before full Phase 3/4.",
    "model_path": LLAMA31_MODEL_PATH,
    "warm_start": {
        "internal": {
            "gpu_mem_frac": 0.20,
            "prefill_batch_size": 4,
            "chunk_size": 512,
            "p2_target_free_blocks": 0,
            "wm_high_ratio": 0.40,
        },
        "vllm": {
            "gpu_memory_utilization": 0.90,
        },
    },
    "smoke_points": [
        {"name": "internal_path_check", "input_len": 2048, "concurrency": 1, "max_new_tokens": 8},
        {"name": "mainline_gate", "input_len": 32768, "concurrency": 16, "max_new_tokens": 1024},
    ],
    "post_smoke_calibration_candidates": {
        "gpu_mem_frac": [0.18, 0.20, 0.22],
        "wm_high_ratio": [0.40],
        "vllm_gpu_memory_utilization": [0.90, 0.95],
    },
}

CALIBRATION_32K: Dict[str, object] = {
    "title": "32K serving memory-knob calibration",
    "input_len": 32768,
    "max_new_tokens": 512,
    "concurrency": 16,
    "methods": ["vllm", "off_compress", "p2_only_compress"],
    "method_labels": {
        "vllm": "vLLM",
        "off_compress": "off_compress",
        "p2_only_compress": "P2 mainline",
    },
    "gpu_mem_frac_grid": [0.28, 0.36, 0.44, 0.52, 0.60, 0.68],
    "vllm_gpu_memory_utilization_grid": [0.75, 0.80, 0.85, 0.90, 0.95],
    "rough_repeats": 2,
    "confirm_top_k": 2,
    "confirm_extra_repeats": 1,
    "selection_priority": [
        "OOM/failure count = 0",
        "valid completion rate",
        "completion rate",
        "tokens/s",
        "TTFT p99 (min)",
    ],
}

SYNTHETIC_FIXED_POINTS: List[Dict[str, int]] = [
    {"name": "32k_c16_n1024", "input_len": 32768, "concurrency": 16, "max_new_tokens": 1024},
    {"name": "16k_c32_n1024", "input_len": 16384, "concurrency": 32, "max_new_tokens": 1024},
    {"name": "8k_c64_n1024", "input_len": 8192, "concurrency": 64, "max_new_tokens": 1024},
]

SYNTHETIC_MAIN_TABLE_FIELDS: List[str] = [
    "Status",
    "completion rate",
    "valid completion rate",
    "OOM/failure count",
    "tokens/s",
    "min free blocks",
]

CONCURRENCY_FRONTIER_32K: Dict[str, object] = {
    "input_len": 32768,
    "max_new_tokens": 1024,
    "concurrency_list": [1, 2, 4, 8, 12, 16, 24, 32, 48, 64],
    "repeats": 2,
    "frontier_reason_enum": ["stable", "degraded_valid", "oom", "timeout", "runtime_failure"],
}

PASSKEY_UNDER_LOAD_32K: Dict[str, object] = {
    "input_len": 32000,
    "max_new_tokens": 32,
    "depths": [0.1, 0.5, 0.9],
    "keys_per_depth": 20,
    "concurrency_list": [1, 4, 8, 16],
    "report_order": ["completion rate", "valid rate", "EM"],
}

SHAREGPT_SERVING: Dict[str, object] = {
    "dataset_name": "ShareGPT_V3_unfiltered_cleaned_split.json",
    "sample_count": 200,
    "prompt_len_range": [4096, 32768],
    "target_len_clip": [64, 512],
    "concurrency_list": [16, 32, 64],
    "repeats": 2,
}

LONGBENCH: Dict[str, object] = {
    "tasks": [
        "passage_retrieval_en",
        "multifieldqa_en",
        "hotpotqa",
        "2wikimqa",
        "musique",
    ],
    "sanity_samples_per_task": 20,
    "full_samples_per_task": 100,
    "max_prompt_tokens": 32768,
    "max_new_tokens": 64,
    "concurrency_list": [1, 16],
}

PHASE_ORDER: List[str] = [
    "single_seq_frozen",
    "dry_run",
    "calibration32k_qwen",
    "synthetic_qwen",
    "frontier32k_qwen",
    "passkey32k_load_qwen",
    "sharegpt_qwen",
    "longbench_qwen",
]

QUALITY_PASSKEY_V3: Dict[str, object] = {
    "gate": {"context_lengths": [8192, 32768], "depths": [0.0, 0.5, 1.0], "keys_per_depth": 5, "methods": ["hf_vanilla", "off_compress_page16", "p2_page16_offline"]},
    "formal": {"context_lengths": [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072], "depths": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "keys_per_depth": 20, "methods": ["hf_vanilla", "off_compress_page16", "p2_page16_offline"]},
    "normalize": "strip_whitespace_only_case_sensitive_punctuation_sensitive",
}

QUALITY_LONGBENCH_V3: Dict[str, object] = {
    "gate_tasks": ["passage_retrieval_en", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"],
    "formal_tasks": ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"],
    "gate_samples_per_task": 20,
    "formal_samples_per_task": 100,
    "max_prompt_tokens": 32768,
    "gate_max_new_tokens": 64,
    "formal_max_gen": "LongBench official dataset2maxlen",
    "methods": ["hf_vanilla", "off_compress_page16", "p2_page16_offline"],
}

