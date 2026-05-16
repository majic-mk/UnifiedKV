import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ALLOC_CONF_ENABLED = (
    str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")).strip() == "expandable_segments:True"
)

import torch
from transformers import AutoTokenizer

RUNNERS_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = RUNNERS_DIR.parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
CORE_DIR = PROJECT_ROOT / "core"
CONFIGS_DIR = BENCHMARKS_DIR / "configs"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(CONFIGS_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIGS_DIR))

from engine import ManagedInferenceEngine
from strategy_groups import (
    ALL_GROUPS,
    ARCHIVE_DIR,
    COMMON_BASE,
    DEFAULT_MAINLINE_GROUPS,
    GROUP_ARGS,
    LOCAL_MODEL_PATH,
    RESULTS_DIR,
)

DEFAULT_INPUT_LENGTHS = [8192, 16384, 32768]
DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16]
DEFAULT_FRONTIER_CONCURRENCY = [1, 2, 4, 8, 16, 24, 32, 48, 64]
DEFAULT_FRONTIER_CONCURRENCY_MAP = {
    8192: [128, 96, 64, 48, 32, 24, 16, 8, 4, 2, 1],
    16384: [64, 48, 32, 24, 16, 8, 4, 2, 1],
    32768: [32, 24, 16, 8, 4, 2, 1],
}
DEFAULT_PERF_FRONTIER_CONCURRENCY_MAP = {
    8192: [256, 224, 192, 160, 128, 96, 80, 64, 48, 32, 24, 16, 8, 4, 2, 1],
    16384: [128, 112, 96, 80, 64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1],
    32768: [96, 80, 64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1],
}
DEFAULT_GPU_MEM_FRAC_MAP = {8192: 0.80, 16384: 0.60, 32768: 0.40}
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_GPU_MEM_FRAC_FALLBACK_STEP = 0.10
DEFAULT_GPU_MEM_FRAC_MIN = 0.20
DEFAULT_FRONTIER_GPU_MEM_FRAC_MAX = 0.92
DEFAULT_FRONTIER_GPU_MEM_FRAC_RESOLUTION = 0.04
DEFAULT_FRONTIER_REFINE_WINDOW = 0.06
DEFAULT_FRONTIER_REFINE_STEP = 0.01
DEFAULT_FRONTIER_SEARCH_MAX_NEW_TOKENS = 64
DEFAULT_FRONTIER_FINAL_EVAL_MAX_NEW_TOKENS = 256
DEFAULT_FRONTIER_SEARCH_REPEATS = 1
DEFAULT_SAFE_CUDA_FREE_GB_MIN = 1.5
DEFAULT_SAFE_CUDA_FREE_FRAC_MIN = 0.06
DEFAULT_REPEATS = 1
DEFAULT_PERF_FRONTIER_GPU_MEM_FRACS = [0.92, 0.90, 0.88, 0.86, 0.84]
DEFAULT_OUT_PREFIX = str(RESULTS_DIR / "capacity_table")
PROMPT_PROFILE = "synthetic_token_bucket_v1"
METRIC_PROFILE = "exp1_capacity_perf"

OOM_KEYWORDS = (
    "out of memory",
    "cuda out of memory",
    "cannot evict enough blocks",
    "cublas_status_alloc_failed",
    "cuda error",
    "no free blocks",
    "allocation failed",
)


def parse_int_list(s: str) -> List[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("empty integer list")
    return out


def parse_float_list(s: str) -> List[float]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("empty float list")
    return out


def parse_groups(s: str) -> List[str]:
    groups: List[str] = []
    seen = set()
    for part in str(s or "").split(","):
        group = str(part).strip()
        if not group:
            continue
        if group not in GROUP_ARGS:
            raise ValueError(f"unknown group: {group}")
        if group in seen:
            continue
        seen.add(group)
        groups.append(group)
    if not groups:
        raise ValueError("empty groups list")
    return groups


def parse_concurrency_map(s: str) -> Dict[int, List[int]]:
    text = str(s or "").strip()
    if not text:
        return {}
    out: Dict[int, List[int]] = {}
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, vals = chunk.split(":", 1)
        out[int(key.strip())] = [int(x.strip()) for x in vals.split(",") if str(x).strip()]
    return out


def enrich_frontier_candidates(cands: Sequence[int]) -> List[int]:
    ranked = sorted({int(x) for x in cands if int(x) > 0}, reverse=True)
    extras: List[int] = []
    for hi, lo in zip(ranked, ranked[1:]):
        if hi == 128 and lo == 64 and 96 not in ranked:
            extras.append(96)
        elif hi == 64 and lo == 32 and 48 not in ranked:
            extras.append(48)
        elif hi == 32 and lo == 16 and 24 not in ranked:
            extras.append(24)
    return sorted(set(ranked + extras), reverse=True)


def frontier_candidates_for_input(args: argparse.Namespace, input_len: int) -> List[int]:
    mapping = parse_concurrency_map(getattr(args, "frontier_concurrency_map", ""))
    if int(input_len) in mapping:
        return enrich_frontier_candidates(mapping[int(input_len)])
    if int(input_len) in DEFAULT_FRONTIER_CONCURRENCY_MAP:
        return enrich_frontier_candidates(DEFAULT_FRONTIER_CONCURRENCY_MAP[int(input_len)])
    return enrich_frontier_candidates(parse_int_list(args.frontier_concurrency_candidates))


def perf_frontier_candidates_for_input(args: argparse.Namespace, input_len: int) -> List[int]:
    mapping = parse_concurrency_map(getattr(args, "perf_frontier_concurrency_map", ""))
    if int(input_len) in mapping:
        return enrich_frontier_candidates(mapping[int(input_len)])
    if int(input_len) in DEFAULT_PERF_FRONTIER_CONCURRENCY_MAP:
        return enrich_frontier_candidates(DEFAULT_PERF_FRONTIER_CONCURRENCY_MAP[int(input_len)])
    return enrich_frontier_candidates(parse_int_list(args.frontier_concurrency_candidates))


def parse_frac_map(s: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        key, value = part.split(":", 1)
        out[int(key.strip())] = float(value.strip())
    if not out:
        raise ValueError("empty gpu_mem_frac map")
    return out


def align_prompt_tolerance(target_tokens: int) -> int:
    return max(1, min(int(round(target_tokens * 0.02)), 256))


def is_oom_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in OOM_KEYWORDS)


def cleanup_engine(engine: Optional[ManagedInferenceEngine], reason: str = "") -> None:
    if engine is None:
        return
    try:
        if hasattr(engine, "has_pending_requests") and engine.has_pending_requests():
            req_map = dict(getattr(engine, "_requests", {}) or {})
            for req in req_map.values():
                try:
                    engine._mark_request_failed(req, RuntimeError(reason or "exp1_cleanup"))
                except Exception:
                    pass
    except Exception:
        pass
    try:
        if hasattr(engine, "_reset_online_runtime"):
            engine._reset_online_runtime(clear_request_counter=False)
    except Exception:
        pass
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def build_engine(
    model_name: str,
    gpu_mem_frac: float,
    group: str,
    max_new_tokens: int,
    chunk_size: int = 0,
    prefill_batch_size: int = 0,
    decode_micro_batch_size: int = 0,
    decode_active_cap_initial: int = 0,
    max_decode_active_cap: int = 0,
) -> ManagedInferenceEngine:
    args = dict(COMMON_BASE)
    args["model_name"] = model_name
    args["gpu_mem_frac"] = float(gpu_mem_frac)
    args["max_new_tokens"] = int(max_new_tokens)
    if int(chunk_size) > 0:
        args["chunk_size"] = int(chunk_size)
    if int(prefill_batch_size) > 0:
        args["prefill_batch_size"] = int(prefill_batch_size)
    args["decode_micro_batch_size"] = int(max(0, decode_micro_batch_size))
    args["decode_active_cap_initial"] = int(max(0, decode_active_cap_initial))
    args["max_decode_active_cap"] = int(max(0, max_decode_active_cap))
    args.update(GROUP_ARGS[group])
    return ManagedInferenceEngine(**args)


def compute_memory_stats(engine: ManagedInferenceEngine, gpu_mem_frac_effective: float) -> Tuple[float, float]:
    total = float(torch.cuda.get_device_properties(0).total_memory)
    reserve = 0.10 * total
    cfg = engine.model.config
    num_layers = int(cfg.num_hidden_layers)
    block_size = int(engine.scheduler.pool.B)
    num_kv_heads = int(engine.scheduler.num_kv_heads)
    head_dim = int(engine.scheduler.head_dim)
    bytes_per_block = float(2 * num_layers * block_size * num_kv_heads * head_dim * 2)
    kv_budget = float(engine.scheduler.pool.N_total) * bytes_per_block
    frac = max(1e-9, float(gpu_mem_frac_effective))
    kv_available = kv_budget / frac
    model_used = max(0.0, 0.90 * total - kv_available)
    return model_used / (1024 ** 3), kv_available / (1024 ** 3)


def compression_profile_for_group(group: str) -> str:
    args = dict(COMMON_BASE)
    args.update(GROUP_ARGS[group])
    sink = int(args.get("sink_len", 16))
    snapkv_obs = int(args.get("snapkv_observation_len", 16))
    retain = float(args.get("retain_ratio", 1.0))
    p2_sink = int(args.get("p2_sink_tokens", 16))
    p2_recent = int(args.get("p2_recent_tokens", 16))
    return f"sink={sink};snapkv_obs={snapkv_obs};retain={retain:.3f};p2_sink={p2_sink};p2_recent={p2_recent}"


def build_prompts_for_cell(tokenizer, target_tokens: int, concurrency: int) -> Tuple[List[str], List[int]]:
    prompts: List[str] = []
    actuals: List[int] = []
    tol = align_prompt_tolerance(target_tokens)
    base_user = (
        "You are a systems assistant. Read the long context and continue with a coherent, technical explanation "
        "about KV-cache management, prefill/decode scheduling, batching, and memory pressure handling. "
        "Do not answer with a list of bullets only; write normal explanatory prose."
    )
    filler = (
        " Context fragment discusses cache locality, request interleaving, scheduler fairness, pinned-memory copies, "
        "windowed retention, and latency-throughput tradeoffs in long-context serving."
    )
    for seq_id in range(int(concurrency)):
        user_text = f"[seq={seq_id}] {base_user}"
        prompt = user_text
        actual = 0
        for _ in range(12):
            prompt = user_text
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_text}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt = user_text
            actual = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            diff = int(target_tokens) - int(actual)
            if abs(diff) <= tol:
                break
            if diff > 0:
                user_text += filler * max(1, diff // 24)
            else:
                trim = max(64, min(len(user_text) // 8, abs(diff) * 4))
                user_text = user_text[:-trim] if trim < len(user_text) else user_text
        prompts.append(prompt)
        actuals.append(int(actual))
    return prompts, actuals


def aggregate_metrics(metrics_list: Sequence[Dict], wall_list: Sequence[float]) -> Dict:
    def avg_num(key: str) -> float:
        vals = [float(m.get(key, 0.0)) for m in metrics_list]
        return float(mean(vals)) if vals else 0.0

    def avg_int(key: str) -> int:
        vals = [int(m.get(key, 0)) for m in metrics_list]
        return int(round(mean(vals))) if vals else 0

    def avg_status_int(key: str) -> int:
        vals = [
            int(m.get(key, dict(m.get("decode_window_status", {}) or {}).get(key, 0)))
            for m in metrics_list
        ]
        return int(round(mean(vals))) if vals else 0

    def avg_status_first_step(key: str) -> int:
        vals = [
            int(m.get(key, dict(m.get("decode_window_status", {}) or {}).get(key, -1)))
            for m in metrics_list
        ]
        valid = [v for v in vals if v >= 0]
        if not valid:
            return -1
        return int(round(mean(valid)))

    fallback_count = sum(int(m.get("decode_path_fallback_count", 0)) for m in metrics_list)
    fallback_topk: Dict[str, int] = {}
    for m in metrics_list:
        for k, v in dict(m.get("decode_path_fallback_reason_topk", {}) or {}).items():
            fallback_topk[str(k)] = fallback_topk.get(str(k), 0) + int(v)

    return {
        "tokens_per_sec": avg_num("tokens_per_sec"),
        "decode_step_p95_ms": avg_num("decode_step_p95_ms"),
        "decode_min_n_free": min(int(m.get("decode_min_n_free", 0)) for m in metrics_list) if metrics_list else 0,
        "prefill_min_n_free": min(int(m.get("prefill_min_n_free", 0)) for m in metrics_list) if metrics_list else 0,
        "global_min_n_free": min(int(m.get("global_min_n_free", 0)) for m in metrics_list) if metrics_list else 0,
        "kv_total_blocks": max(int(m.get("kv_total_blocks", 0)) for m in metrics_list) if metrics_list else 0,
        "kv_peak_used_blocks": max(int(m.get("kv_peak_used_blocks", 0)) for m in metrics_list) if metrics_list else 0,
        "thrash_win16": avg_num("thrash_win16"),
        "decode_append_fail_count": avg_int("decode_append_fail_count"),
        "decode_backpressure_events": avg_int("decode_backpressure_events"),
        "decode_retry_timeout_fail_count": avg_int("decode_retry_timeout_fail_count"),
        "decode_no_progress_steps": avg_int("decode_no_progress_steps"),
        "wm_low": max(int(m.get("wm_low", 0)) for m in metrics_list) if metrics_list else 0,
        "wm_high": max(int(m.get("wm_high", 0)) for m in metrics_list) if metrics_list else 0,
        "p2_low_threshold": max(int(m.get("p2_low_threshold", 0)) for m in metrics_list) if metrics_list else 0,
        "p2_target_free_blocks": max(int(m.get("p2_target_free_blocks", 0)) for m in metrics_list) if metrics_list else 0,
        "p2_active_steps": avg_status_int("p2_active_steps"),
        "p2_candidate_steps": avg_status_int("p2_candidate_steps"),
        "p2_recovery_fail_windows": avg_status_int("p2_recovery_fail_windows"),
        "p2_no_candidate_steps": avg_status_int("p2_no_candidate_steps"),
        "p2_attempted_steps": avg_status_int("p2_attempted_steps"),
        "p2_success_steps": avg_status_int("p2_success_steps"),
        "p2_ready_candidate_steps": avg_status_int("p2_ready_candidate_steps"),
        "p2_decode_candidate_steps": avg_status_int("p2_decode_candidate_steps"),
        "p2_expected_reclaim_blocks": avg_status_int("p2_expected_reclaim_blocks"),
        "p2_gain_success_steps": avg_status_int("p2_gain_success_steps"),
        "p2_gain_fail_steps": avg_status_int("p2_gain_fail_steps"),
        "p2_skipped_low_benefit_steps": avg_status_int("p2_skipped_low_benefit_steps"),
        "first_p2_step": avg_status_first_step("first_p2_step"),
        "offloader_delta": dict(metrics_list[-1].get("offloader_delta", {})) if metrics_list else {},
        "avg_decode_microbatch_size": avg_num("avg_decode_microbatch_size"),
        "decode_path_selected": str(metrics_list[-1].get("decode_path_selected", "")) if metrics_list else "",
        "decode_path_fallback_count": int(fallback_count),
        "decode_path_fallback_reason_topk": fallback_topk,
        "peak_gpu_mem_allocated_gb": max(float(m.get("peak_gpu_mem_allocated_gb", m.get("peak_cuda_mem_gb", 0.0))) for m in metrics_list) if metrics_list else 0.0,
        "cuda_free_min_gb": min(float(m.get("cuda_free_min_gb", 0.0)) for m in metrics_list) if metrics_list else 0.0,
        "cuda_free_post_cleanup_min_gb": min(float(m.get("cuda_free_post_cleanup_min_gb", 0.0)) for m in metrics_list) if metrics_list else 0.0,
        "cuda_free_post_cleanup_last_gb": float(metrics_list[-1].get("cuda_free_post_cleanup_last_gb", 0.0)) if metrics_list else 0.0,
        "decode_cuda_free_post_cleanup_recent_min_gb": min(float(m.get("decode_cuda_free_post_cleanup_recent_min_gb", 0.0)) for m in metrics_list) if metrics_list else 0.0,
        "cuda_alloc_peak_gb": max(float(m.get("cuda_alloc_peak_gb", 0.0)) for m in metrics_list) if metrics_list else 0.0,
        "cuda_reserved_peak_gb": max(float(m.get("cuda_reserved_peak_gb", 0.0)) for m in metrics_list) if metrics_list else 0.0,
        "cuda_total_gb": max(float(m.get("cuda_total_gb", 0.0)) for m in metrics_list) if metrics_list else 0.0,
        "p2_cuda_pressure_signal_steps": avg_status_int("p2_cuda_pressure_signal_steps"),
        "ensure_fail_event_count": sum(int(m.get("ensure_fail_event_count", 0)) for m in metrics_list),
        "ensure_fail_ready_count": sum(int(m.get("ensure_fail_ready_count", 0)) for m in metrics_list),
        "ensure_fail_layer2_count": sum(int(m.get("ensure_fail_layer2_count", 0)) for m in metrics_list),
        "ensure_fail_layer3_count": sum(int(m.get("ensure_fail_layer3_count", 0)) for m in metrics_list),
        "ensure_fail_unknown_count": sum(int(m.get("ensure_fail_unknown_count", 0)) for m in metrics_list),
        "ensure_fail_by_layer": dict(metrics_list[-1].get("ensure_fail_by_layer", {})) if metrics_list else {},
        "ensure_fail_events_sample": list(metrics_list[-1].get("ensure_fail_events_sample", [])) if metrics_list else [],
        "wall_ms": float(mean(wall_list)) if wall_list else 0.0,
        "success_rate": 1.0 if metrics_list else 0.0,
    }


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_step_jsonl_partial(path: str) -> Optional[Dict]:
    p = Path(str(path or "").strip())
    if not p.exists() or not p.is_file():
        return None
    rows: List[Dict] = []
    try:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return None
    if not rows:
        return None
    rows.sort(key=lambda r: float(r.get("ts", 0.0)))
    first_ts = float(rows[0].get("ts", 0.0))
    last_ts = float(rows[-1].get("ts", first_ts))
    elapsed_s = max(1e-6, last_ts - first_ts)
    total_decode_tokens = int(sum(int(r.get("decode_tokens", 0)) for r in rows))
    step_lat_ms: List[float] = []
    for prev, cur in zip(rows, rows[1:]):
        dt_ms = max(0.0, (float(cur.get("ts", 0.0)) - float(prev.get("ts", 0.0))) * 1000.0)
        step_lat_ms.append(dt_ms)
    last = rows[-1]
    first_p2_step = -1
    for row in rows:
        step_id = int(row.get("step", 0))
        if first_p2_step < 0 and int(row.get("p2_attempted_steps", 0)) > 0:
            first_p2_step = step_id
    return {
        "summary_mode": "partial_failure",
        "partial_from_step_jsonl": 1,
        "partial_step_count": int(len(rows)),
        "tokens_per_sec": float(round(float(total_decode_tokens) / elapsed_s, 4)),
        "decode_step_p95_ms": float(round(ManagedInferenceEngine._percentile(step_lat_ms, 0.95), 3)) if step_lat_ms else 0.0,
        "decode_min_n_free": int(min(int(r.get("decode_min_n_free", 0)) for r in rows)),
        "wm_low": int(max(int(r.get("wm_low", 0)) for r in rows)),
        "wm_high": int(max(int(r.get("wm_high", 0)) for r in rows)),
        "p2_low_threshold": int(max(int(r.get("p2_low_threshold", 0)) for r in rows)),
        "decode_active_cap_boot": int(max(int(r.get("decode_active_cap_boot", 0)) for r in rows)),
        "decode_active_cap_final": int(last.get("decode_active_cap", 0)),
        "decode_active_cap_min_seen": int(min(int(r.get("decode_active_cap_min_seen", 0)) for r in rows if int(r.get("decode_active_cap_min_seen", 0)) > 0)) if any(int(r.get("decode_active_cap_min_seen", 0)) > 0 for r in rows) else 0,
        "thrash_win16": float(last.get("thrash_win16", 0.0)),
        "decode_backpressure_events": int(max(int(r.get("decode_backpressure_events", 0)) for r in rows)),
        "prefill_backpressure_events": int(max(int(r.get("prefill_backpressure_events", 0)) for r in rows)),
        "prefill_batch_failed_steps": int(max(int(r.get("prefill_batch_failed_steps", 0)) for r in rows)),
        "prefill_chunk_failed_steps": int(max(int(r.get("prefill_chunk_failed_steps", 0)) for r in rows)),
        "prefill_activate_failed_steps": int(max(int(r.get("prefill_activate_failed_steps", 0)) for r in rows)),
        "prefill_pause_steps": int(max(int(r.get("prefill_pause_steps", 0)) for r in rows)),
        "p2_active_steps": int(max(int(r.get("p2_active_steps", 0)) for r in rows)),
        "p2_candidate_steps": int(max(int(r.get("p2_candidate_steps", 0)) for r in rows)),
        "p2_attempted_steps": int(max(int(r.get("p2_attempted_steps", 0)) for r in rows)),
        "p2_success_steps": int(max(int(r.get("p2_success_steps", 0)) for r in rows)),
        "p2_no_candidate_steps": int(max(int(r.get("p2_no_candidate_steps", 0)) for r in rows)),
        "first_p2_step": int(first_p2_step),
        "cuda_free_min_gb": float(min(float(r.get("cuda_free_min_gb", 0.0)) for r in rows)),
        "cuda_free_post_cleanup_min_gb": float(min(float(r.get("cuda_free_post_cleanup_min_gb", 0.0)) for r in rows)),
        "cuda_free_post_cleanup_last_gb": float(last.get("cuda_free_post_cleanup_last_gb", 0.0)),
        "decode_cuda_free_post_cleanup_recent_min_gb": float(min(float(r.get("decode_cuda_free_post_cleanup_recent_min_gb", 0.0)) for r in rows)),
        "p2_cuda_pressure_signal_steps": int(max(int(r.get("p2_cuda_pressure_signal_steps", 0)) for r in rows)),
        "wall_ms": float(round(elapsed_s * 1000.0, 3)),
    }


def worker_main(args: argparse.Namespace) -> None:
    result = {
        "group": args.worker_group,
        "input_len": int(args.worker_input_len),
        "concurrency": int(args.worker_concurrency),
        "gpu_mem_frac_initial": float(args.worker_gpu_mem_frac_initial),
        "gpu_mem_frac_effective": float(args.worker_gpu_mem_frac_initial),
        "model_memory_gb": 0.0,
        "kv_available_gb": 0.0,
        "success": 0,
        "success_rate": 0.0,
        "oom": 0,
        "tokens_per_sec": 0.0,
        "decode_step_p95_ms": 0.0,
        "decode_min_n_free": 0,
        "prefill_min_n_free": 0,
        "global_min_n_free": 0,
        "wm_low": 0,
        "wm_high": 0,
        "p2_low_threshold": 0,
        "kv_total_blocks": 0,
        "kv_peak_used_blocks": 0,
        "thrash_win16": 0.0,
        "decode_append_fail_count": 0,
        "decode_backpressure_events": 0,
        "decode_retry_timeout_fail_count": 0,
        "decode_no_progress_steps": 0,
        "p2_active_steps": 0,
        "p2_candidate_steps": 0,
        "p2_recovery_fail_windows": 0,
        "p2_no_candidate_steps": 0,
        "p2_attempted_steps": 0,
        "p2_success_steps": 0,
        "first_p2_step": -1,
        "offloader_delta": {},
        "avg_decode_microbatch_size": 0.0,
        "decode_path_selected": "",
        "decode_path_fallback_count": 0,
        "decode_path_fallback_reason_topk": {},
        "peak_gpu_mem_allocated_gb": 0.0,
        "cuda_free_min_gb": 0.0,
        "cuda_free_post_cleanup_min_gb": 0.0,
        "cuda_free_post_cleanup_last_gb": 0.0,
        "decode_cuda_free_post_cleanup_recent_min_gb": 0.0,
        "cuda_alloc_peak_gb": 0.0,
        "cuda_reserved_peak_gb": 0.0,
        "cuda_total_gb": 0.0,
        "p2_cuda_pressure_signal_steps": 0,
        "error_reason": "",
        "summary_mode": "pending",
        "compression_profile": compression_profile_for_group(args.worker_group),
        "prompt_profile": PROMPT_PROFILE,
        "metric_profile": METRIC_PROFILE,
        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        "actual_prompt_tokens": [],
        "wall_ms": 0.0,
        "worker_fixed_gpu_mem_frac": int(bool(getattr(args, "worker_fixed_gpu_mem_frac", False))),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    prompts, actual_prompt_tokens = build_prompts_for_cell(tokenizer, int(args.worker_input_len), int(args.worker_concurrency))
    result["actual_prompt_tokens"] = actual_prompt_tokens
    step_jsonl_path = str(getattr(args, 'worker_step_jsonl', '') or '').strip()
    step_jsonl_every = max(1, int(getattr(args, 'worker_step_jsonl_every', 1) or 1))

    def step_callback(step_stats: Dict) -> None:
        if not step_jsonl_path:
            return
        step_id = int(step_stats.get('step', 0))
        if step_jsonl_every > 1 and (step_id % step_jsonl_every) != 0:
            return
        rec = {
            'ts': time.time(),
            'step': step_id,
            'group': args.worker_group,
            'input_len': int(args.worker_input_len),
            'concurrency': int(args.worker_concurrency),
            'decode_scheduled': int(step_stats.get('decode_scheduled', 0)),
            'decode_tokens': int(step_stats.get('decode_tokens', 0)),
            'decode_microbatches': int(step_stats.get('decode_microbatches', 0)),
            'materialized_blocks': int(step_stats.get('materialized_blocks', 0)),
            'missing_blocks_scheduled': int(step_stats.get('missing_blocks_scheduled', 0)),
            'decode_active_cap': int(step_stats.get('decode_active_cap', 0)),
            'decode_active_cap_boot': int(step_stats.get('decode_active_cap_boot', 0)),
            'decode_active_cap_min_seen': int(step_stats.get('decode_active_cap_min_seen', 0)),
            'thrash_win16': float(step_stats.get('thrash_win16', 0.0)),
            'n_free': int(step_stats.get('n_free', 0)),
            'decode_min_n_free': int(step_stats.get('decode_min_n_free', 0)),
            'wm_low': int(step_stats.get('wm_low', 0)),
            'wm_high': int(step_stats.get('wm_high', 0)),
            'p2_low_threshold': int(step_stats.get('p2_low_threshold', 0)),
            'ready_decode_resident_blocks': int(step_stats.get('ready_decode_resident_blocks', 0)),
            'decode_active_resident_blocks': int(step_stats.get('decode_active_resident_blocks', 0)),
            'cuda_free_post_cleanup_min_gb': float(step_stats.get('cuda_free_post_cleanup_min_gb', 0.0)),
            'cuda_free_post_cleanup_last_gb': float(step_stats.get('cuda_free_post_cleanup_last_gb', 0.0)),
            'cuda_free_min_gb': float(step_stats.get('cuda_free_min_gb', 0.0)),
            'decode_cuda_free_post_cleanup_recent_min_gb': float(step_stats.get('decode_cuda_free_post_cleanup_recent_min_gb', 0.0)),
            'p2_cuda_pressure_signal_steps': int(step_stats.get('p2_cuda_pressure_signal_steps', 0)),
            'p2_active_steps': int(step_stats.get('p2_active_steps', 0)),
            'p2_candidate_steps': int(step_stats.get('p2_candidate_steps', 0)),
            'p2_attempted_steps': int(step_stats.get('p2_attempted_steps', 0)),
            'p2_success_steps': int(step_stats.get('p2_success_steps', 0)),
            'p2_no_candidate_steps': int(step_stats.get('p2_no_candidate_steps', 0)),
            'p2_ready_candidate_steps': int(step_stats.get('p2_ready_candidate_steps', 0)),
            'p2_decode_candidate_steps': int(step_stats.get('p2_decode_candidate_steps', 0)),
            'p2_expected_reclaim_blocks': int(step_stats.get('p2_expected_reclaim_blocks', 0)),
            'p2_gain_success_steps': int(step_stats.get('p2_gain_success_steps', 0)),
            'p2_gain_fail_steps': int(step_stats.get('p2_gain_fail_steps', 0)),
            'p2_skipped_low_benefit_steps': int(step_stats.get('p2_skipped_low_benefit_steps', 0)),
            'p2_fail_streak': int(step_stats.get('p2_fail_streak', 0)),
            'p2_last_candidate_count': int(step_stats.get('p2_last_candidate_count', 0)),
            'p2_managed_active': int(step_stats.get('p2_managed_active', 0)),
            'p2_recover_streak': int(step_stats.get('p2_recover_streak', 0)),
            'decode_backpressure_events': int(step_stats.get('decode_backpressure_events', 0)),
            'prefill_backpressure_events': int(step_stats.get('prefill_backpressure_events', 0)),
            'prefill_batch_failed_steps': int(step_stats.get('prefill_batch_failed_steps', 0)),
            'prefill_chunk_failed_steps': int(step_stats.get('prefill_chunk_failed_steps', 0)),
            'prefill_activate_failed_steps': int(step_stats.get('prefill_activate_failed_steps', 0)),
            'prefill_pause_steps': int(step_stats.get('prefill_pause_steps', 0)),
            'prefill_batch_merge_peak_gb': float(step_stats.get('prefill_batch_merge_peak_gb', 0.0) or 0.0),
            'prefill_batch_input_peak_gb': float(step_stats.get('prefill_batch_input_peak_gb', 0.0) or 0.0),
            'prefill_batch_forward_peak_gb': float(step_stats.get('prefill_batch_forward_peak_gb', 0.0) or 0.0),
            'prefill_batch_slice_peak_gb': float(step_stats.get('prefill_batch_slice_peak_gb', 0.0) or 0.0),
            'prefill_chunk_input_peak_gb': float(step_stats.get('prefill_chunk_input_peak_gb', 0.0) or 0.0),
            'prefill_chunk_forward_peak_gb': float(step_stats.get('prefill_chunk_forward_peak_gb', 0.0) or 0.0),
        }
        with open(step_jsonl_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')

    frac = float(args.worker_gpu_mem_frac_initial)
    engine = None
    try:
        while frac >= float(args.gpu_mem_frac_min) - 1e-9:
            try:
                engine = build_engine(
                    args.model_name,
                    frac,
                    args.worker_group,
                    int(args.max_new_tokens),
                    chunk_size=int(getattr(args, 'chunk_size', 0) or 0),
                    prefill_batch_size=int(getattr(args, 'prefill_batch_size', 0) or 0),
                    decode_micro_batch_size=int(args.decode_micro_batch_size),
                    decode_active_cap_initial=int(args.decode_active_cap_initial),
                    max_decode_active_cap=int(args.max_decode_active_cap),
                )
                model_memory_gb, kv_available_gb = compute_memory_stats(engine, frac)
                metrics_list = []
                wall_list = []
                for _ in range(int(args.repeats)):
                    t0 = time.perf_counter()
                    outputs, metrics = engine.generate(prompts, return_metrics=True, step_callback=step_callback)
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    if not isinstance(outputs, list) or len(outputs) != int(args.worker_concurrency):
                        raise RuntimeError(f"unexpected_output_count:{len(outputs) if isinstance(outputs, list) else 'non_list'}")
                    metrics_list.append(metrics)
                    wall_list.append(wall_ms)
                agg = aggregate_metrics(metrics_list, wall_list)
                result.update({
                    "gpu_mem_frac_effective": float(frac),
                    "model_memory_gb": float(round(model_memory_gb, 4)),
                    "kv_available_gb": float(round(kv_available_gb, 4)),
                    "success": 1,
                    "success_rate": float(agg["success_rate"]),
                    "oom": 0,
                    "tokens_per_sec": float(agg["tokens_per_sec"]),
                    "decode_step_p95_ms": float(agg["decode_step_p95_ms"]),
                    "decode_min_n_free": int(agg["decode_min_n_free"]),
                    "prefill_min_n_free": int(agg["prefill_min_n_free"]),
                    "global_min_n_free": int(agg["global_min_n_free"]),
                    "wm_low": int(agg["wm_low"]),
                    "wm_high": int(agg["wm_high"]),
                    "p2_low_threshold": int(agg["p2_low_threshold"]),
                    "p2_target_free_blocks": int(agg["p2_target_free_blocks"]),
                    "kv_total_blocks": int(agg["kv_total_blocks"]),
                    "kv_peak_used_blocks": int(agg["kv_peak_used_blocks"]),
                    "thrash_win16": float(agg["thrash_win16"]),
                    "decode_append_fail_count": int(agg["decode_append_fail_count"]),
                    "decode_backpressure_events": int(agg["decode_backpressure_events"]),
                    "decode_retry_timeout_fail_count": int(agg["decode_retry_timeout_fail_count"]),
                    "decode_no_progress_steps": int(agg["decode_no_progress_steps"]),
                    "p2_active_steps": int(agg["p2_active_steps"]),
                    "p2_candidate_steps": int(agg["p2_candidate_steps"]),
                    "p2_recovery_fail_windows": int(agg["p2_recovery_fail_windows"]),
                    "p2_no_candidate_steps": int(agg["p2_no_candidate_steps"]),
                    "p2_attempted_steps": int(agg["p2_attempted_steps"]),
                    "p2_success_steps": int(agg["p2_success_steps"]),
                    "p2_ready_candidate_steps": int(agg["p2_ready_candidate_steps"]),
                    "p2_decode_candidate_steps": int(agg["p2_decode_candidate_steps"]),
                    "p2_expected_reclaim_blocks": int(agg["p2_expected_reclaim_blocks"]),
                    "p2_gain_success_steps": int(agg["p2_gain_success_steps"]),
                    "p2_gain_fail_steps": int(agg["p2_gain_fail_steps"]),
                    "p2_skipped_low_benefit_steps": int(agg["p2_skipped_low_benefit_steps"]),
                    "first_p2_step": int(agg["first_p2_step"]),
                    "offloader_delta": dict(agg["offloader_delta"]),
                    "avg_decode_microbatch_size": float(agg["avg_decode_microbatch_size"]),
                    "decode_path_selected": str(agg["decode_path_selected"]),
                    "decode_path_fallback_count": int(agg["decode_path_fallback_count"]),
                    "decode_path_fallback_reason_topk": dict(agg["decode_path_fallback_reason_topk"]),
                    "peak_gpu_mem_allocated_gb": float(agg["peak_gpu_mem_allocated_gb"]),
                    "cuda_free_min_gb": float(agg["cuda_free_min_gb"]),
                    "cuda_free_post_cleanup_min_gb": float(agg["cuda_free_post_cleanup_min_gb"]),
                    "cuda_free_post_cleanup_last_gb": float(agg["cuda_free_post_cleanup_last_gb"]),
                    "decode_cuda_free_post_cleanup_recent_min_gb": float(agg["decode_cuda_free_post_cleanup_recent_min_gb"]),
                    "cuda_alloc_peak_gb": float(agg["cuda_alloc_peak_gb"]),
                    "cuda_reserved_peak_gb": float(agg["cuda_reserved_peak_gb"]),
                    "cuda_total_gb": float(agg["cuda_total_gb"]),
                    "p2_cuda_pressure_signal_steps": int(agg["p2_cuda_pressure_signal_steps"]),
                    "ensure_fail_event_count": int(agg["ensure_fail_event_count"]),
                    "ensure_fail_ready_count": int(agg["ensure_fail_ready_count"]),
                    "ensure_fail_layer2_count": int(agg["ensure_fail_layer2_count"]),
                    "ensure_fail_layer3_count": int(agg["ensure_fail_layer3_count"]),
                    "ensure_fail_unknown_count": int(agg["ensure_fail_unknown_count"]),
                    "ensure_fail_by_layer": dict(agg["ensure_fail_by_layer"]),
                    "ensure_fail_events_sample": list(agg["ensure_fail_events_sample"]),
                    "wall_ms": float(round(agg["wall_ms"], 3)),
                    "summary_mode": "success",
                })
                cleanup_engine(engine, "exp1_worker_success")
                engine = None
                write_json(Path(args.worker_out), result)
                return
            except Exception as exc:
                oom = is_oom_error(exc)
                result["error_reason"] = str(exc)
                result["oom"] = int(oom)
                result["gpu_mem_frac_effective"] = float(max(frac, args.gpu_mem_frac_min))
                partial = summarize_step_jsonl_partial(step_jsonl_path)
                if partial:
                    result.update(partial)
                if engine is not None:
                    cleanup_engine(engine, result["error_reason"] or "exp1_worker_exception")
                    engine = None
                fallback_step = float(args.gpu_mem_frac_fallback_step)
                fixed_frac_mode = bool(getattr(args, "worker_fixed_gpu_mem_frac", False)) or fallback_step <= 0.0
                if (
                    oom
                    and (not fixed_frac_mode)
                    and fallback_step > 0.0
                    and frac - fallback_step >= float(args.gpu_mem_frac_min) - 1e-9
                ):
                    frac = round(frac - fallback_step, 10)
                    continue
                write_json(Path(args.worker_out), result)
                return
    finally:
        if engine is not None:
            cleanup_engine(engine, "exp1_worker_finally")
    write_json(Path(args.worker_out), result)


def write_progress(progress_path: Path, rec: Dict) -> None:
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def derive_capacity_summary(rows: Sequence[Dict], input_lengths: Sequence[int], groups: Sequence[str]) -> List[Dict]:
    out = []
    for group in groups:
        rec = {"group": group}
        for input_len in input_lengths:
            ok = [int(r["concurrency"]) for r in rows if r["group"] == group and int(r["input_len"]) == int(input_len) and int(r["success"]) == 1]
            rec[f"max_supported_concurrency_{input_len}"] = max(ok) if ok else 0
        out.append(rec)
    return out


def case_is_runnable(rec: Dict) -> bool:
    return (
        int(rec.get("success", 0)) == 1
        and int(rec.get("oom", 0)) == 0
        and float(rec.get("success_rate", 0.0)) >= 0.999
        and int(rec.get("decode_retry_timeout_fail_count", 0)) == 0
        and int(rec.get("decode_no_progress_steps", 0)) == 0
    )


def case_is_best_safe(rec: Dict, safe_cuda_free_gb_min: float, safe_cuda_free_frac_min: float) -> bool:
    total_gb = float(rec.get("cuda_total_gb", 0.0) or 0.0)
    min_free = float(rec.get("cuda_free_min_gb", 0.0) or 0.0)
    free_req = max(float(safe_cuda_free_gb_min), float(safe_cuda_free_frac_min) * total_gb) if total_gb > 0 else float(safe_cuda_free_gb_min)
    return (
        int(rec.get("success", 0)) == 1
        and int(rec.get("oom", 0)) == 0
        and float(rec.get("success_rate", 0.0)) >= 0.999
        and int(rec.get("decode_retry_timeout_fail_count", 0)) == 0
        and int(rec.get("decode_no_progress_steps", 0)) == 0
        and min_free >= free_req
    )


def frontier_search_for_case(script_path: Path, args: argparse.Namespace, group: str, input_len: int, concurrency: int, progress_path: Optional[Path] = None) -> Tuple[Optional[Dict], Optional[Dict], List[Dict]]:
    def run_case(frac: float, eval_max_new_tokens: int, eval_repeats: int, stage: str, frontier_kind: str) -> Dict:
        frac = round(float(frac), 6)
        with tempfile.NamedTemporaryFile(prefix="exp1_case_", suffix=".json", delete=False) as tf:
            worker_out = tf.name
        if progress_path is not None:
            write_progress(progress_path, {
                "event": "frontier_attempt_begin",
                "group": group,
                "input_len": int(input_len),
                "concurrency": int(concurrency),
                "gpu_mem_frac_initial": float(frac),
                "eval_stage": str(stage),
                "eval_max_new_tokens": int(eval_max_new_tokens),
                "eval_repeats": int(eval_repeats),
                "frontier_kind": str(frontier_kind),
            })
        cmd = [
            sys.executable,
            str(script_path),
            "--worker",
            "--model-name", args.model_name,
            "--max-new-tokens", str(int(eval_max_new_tokens)),
            "--repeats", str(int(eval_repeats)),
            "--gpu-mem-frac-fallback-step", "1.0",
            "--gpu-mem-frac-min", str(frac),
            "--chunk-size", str(int(getattr(args, "chunk_size", 0) or 0)),
            "--prefill-batch-size", str(int(getattr(args, "prefill_batch_size", 0) or 0)),
            "--decode-micro-batch-size", str(int(args.decode_micro_batch_size)),
            "--decode-active-cap-initial", str(int(args.decode_active_cap_initial)),
            "--max-decode-active-cap", str(int(args.max_decode_active_cap)),
            "--worker-group", group,
            "--worker-input-len", str(int(input_len)),
            "--worker-concurrency", str(int(concurrency)),
            "--worker-gpu-mem-frac-initial", str(frac),
            "--worker-out", worker_out,
        ]
        if str(getattr(args, 'worker_step_jsonl', '') or '').strip():
            cmd.extend([
                '--worker-step-jsonl', worker_out + '.steps.jsonl',
                '--worker-step-jsonl-every', str(int(getattr(args, 'worker_step_jsonl_every', 1) or 1)),
            ])
        proc = subprocess.run(cmd, cwd=str(script_path.parent), capture_output=True, text=True)
        try:
            rec = json.loads(Path(worker_out).read_text(encoding="utf-8"))
        except Exception:
            rec = {
                "group": group,
                "input_len": int(input_len),
                "concurrency": int(concurrency),
                "gpu_mem_frac_initial": frac,
                "gpu_mem_frac_effective": frac,
                "success": 0,
                "success_rate": 0.0,
                "oom": 0,
                "tokens_per_sec": 0.0,
                "decode_step_p95_ms": 0.0,
                "cuda_free_min_gb": 0.0,
                "cuda_free_post_cleanup_min_gb": 0.0,
                "cuda_free_post_cleanup_last_gb": 0.0,
                "decode_cuda_free_post_cleanup_recent_min_gb": 0.0,
                "wm_low": 0,
                "wm_high": 0,
                "p2_low_threshold": 0,
                "cuda_total_gb": 0.0,
                "p2_cuda_pressure_signal_steps": 0,
                "kv_total_blocks": 0,
                "kv_peak_used_blocks": 0,
                "error_reason": f"worker_failed: rc={proc.returncode}; stderr_tail={proc.stderr[-400:]}",
            }
        finally:
            try:
                Path(worker_out).unlink(missing_ok=True)
            except Exception:
                pass
        rec["eval_stage"] = str(stage)
        rec["eval_max_new_tokens"] = int(eval_max_new_tokens)
        rec["eval_repeats"] = int(eval_repeats)
        rec["frontier_kind"] = str(frontier_kind)
        return rec

    def search_best(kind: str, accept_fn) -> Tuple[Optional[Dict], List[Dict]]:
        lo = float(args.frontier_gpu_mem_frac_min)
        hi = float(args.frontier_gpu_mem_frac_max)
        coarse_resolution = float(args.frontier_gpu_mem_frac_resolution)
        refine_window = float(args.frontier_refine_window)
        refine_step = float(args.frontier_refine_step)
        search_max_new_tokens = int(args.frontier_search_max_new_tokens)
        final_eval_max_new_tokens = int(args.frontier_final_eval_max_new_tokens)
        search_repeats = int(args.frontier_search_repeats)
        final_eval_repeats = int(args.repeats)
        local_attempts: List[Dict] = []
        best: Optional[Dict] = None

        while hi - lo >= coarse_resolution - 1e-9:
            mid = round((lo + hi) / 2.0, 6)
            rec = run_case(mid, search_max_new_tokens, search_repeats, "search", kind)
            local_attempts.append(rec)
            if accept_fn(rec):
                best = rec
                lo = mid + coarse_resolution
            else:
                hi = mid - coarse_resolution

        if best is not None and refine_step > 0.0 and refine_window > 0.0:
            best_frac = float(best.get("gpu_mem_frac_effective", best.get("gpu_mem_frac_initial", lo)))
            upper = min(float(args.frontier_gpu_mem_frac_max), best_frac + refine_window)
            frac = round(best_frac + refine_step, 6)
            while frac <= upper + 1e-9:
                rec = run_case(frac, search_max_new_tokens, search_repeats, "search", kind)
                local_attempts.append(rec)
                if accept_fn(rec):
                    best = rec
                    frac = round(frac + refine_step, 6)
                    continue
                break

        if best is None:
            return None, local_attempts

        best_frac = float(best.get("gpu_mem_frac_effective", best.get("gpu_mem_frac_initial", 0.0)))
        final_step = refine_step if refine_step > 0.0 else coarse_resolution
        if final_step <= 0.0:
            final_step = 0.01
        successful_search_fracs = sorted({
            round(float(rec.get("gpu_mem_frac_effective", rec.get("gpu_mem_frac_initial", 0.0))), 6)
            for rec in local_attempts
            if rec.get("eval_stage") == "search" and accept_fn(rec)
        }, reverse=True)
        final_candidates: List[float] = []
        seen_final_fracs = set()
        def add_final_frac(frac: float) -> None:
            frac = round(float(frac), 6)
            if frac < float(args.frontier_gpu_mem_frac_min) - 1e-9:
                return
            if frac in seen_final_fracs:
                return
            seen_final_fracs.add(frac)
            final_candidates.append(frac)
        for frac in successful_search_fracs:
            add_final_frac(frac)
        probe = round(best_frac - final_step, 6)
        while probe >= float(args.frontier_gpu_mem_frac_min) - 1e-9:
            add_final_frac(probe)
            probe = round(probe - final_step, 6)

        final_error_reason = ""
        for candidate_frac in final_candidates:
            final_rec = run_case(candidate_frac, final_eval_max_new_tokens, final_eval_repeats, "final", kind)
            local_attempts.append(final_rec)
            if accept_fn(final_rec):
                final_rec[f"search_gpu_mem_frac_best_{kind}"] = best_frac
                final_rec[f"final_gpu_mem_frac_best_{kind}"] = candidate_frac
                return final_rec, local_attempts
            final_error_reason = str(final_rec.get("error_reason", f"final_eval_not_{kind}"))

        best["final_eval_failed"] = 1
        best["final_eval_error_reason"] = final_error_reason or f"final_eval_not_{kind}"
        best[f"search_gpu_mem_frac_best_{kind}"] = best_frac
        return None, local_attempts

    safe_accept = lambda rec: case_is_best_safe(rec, float(args.safe_cuda_free_gb_min), float(args.safe_cuda_free_frac_min))
    runnable_accept = lambda rec: case_is_runnable(rec)

    safe_best, safe_attempts = search_best("safe", safe_accept)
    runnable_best, runnable_attempts = search_best("runnable", runnable_accept)
    return safe_best, runnable_best, safe_attempts + runnable_attempts

def write_frontier_csv(path: Path, rows: Sequence[Dict]) -> None:
    header = [
        "group", "input_len",
        "max_supported_concurrency_safe", "gpu_mem_frac_best_safe", "tokens_per_sec_safe", "decode_step_p95_ms_safe", "cuda_free_min_gb_safe", "kv_peak_ratio_safe", "first_p2_step_safe", "p2_active_steps_safe",
        "max_supported_concurrency_runnable", "gpu_mem_frac_best_runnable", "tokens_per_sec_runnable", "decode_step_p95_ms_runnable", "cuda_free_min_gb_runnable", "kv_peak_ratio_runnable", "first_p2_step_runnable", "p2_active_steps_runnable"
    ]
    def peak_ratio(rec: Dict) -> float:
        peak = int(rec.get("kv_peak_used_blocks", 0))
        total = max(1, int(rec.get("kv_total_blocks", 0)))
        return float(peak / total)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rec in rows:
            safe = dict(rec.get("safe", {}))
            run = dict(rec.get("runnable", {}))
            vals = [
                str(rec.get("group", "")),
                str(int(rec.get("input_len", 0))),
                str(int(rec.get("max_supported_concurrency_safe", 0))),
                f"{float(rec.get('gpu_mem_frac_best_safe', 0.0)):.3f}",
                f"{float(safe.get('tokens_per_sec', 0.0)):.4f}",
                f"{float(safe.get('decode_step_p95_ms', 0.0)):.3f}",
                f"{float(safe.get('cuda_free_min_gb', 0.0)):.4f}",
                f"{peak_ratio(safe):.6f}",
                str(int(safe.get('first_p2_step', -1))),
                str(int(safe.get('p2_active_steps', 0))),
                str(int(rec.get("max_supported_concurrency_runnable", 0))),
                f"{float(rec.get('gpu_mem_frac_best_runnable', 0.0)):.3f}",
                f"{float(run.get('tokens_per_sec', 0.0)):.4f}",
                f"{float(run.get('decode_step_p95_ms', 0.0)):.3f}",
                f"{float(run.get('cuda_free_min_gb', 0.0)):.4f}",
                f"{peak_ratio(run):.6f}",
                str(int(run.get('first_p2_step', -1))),
                str(int(run.get('p2_active_steps', 0))),
            ]
            f.write(",".join(vals) + "\n")

def orchestrator_frontier_main(args: argparse.Namespace) -> None:
    input_lengths = parse_int_list(args.input_lengths)
    selected_groups = parse_groups(args.groups)
    out_prefix = Path(args.out_prefix)
    progress_path = out_prefix.with_suffix(".progress.jsonl")
    result_json = out_prefix.with_name(out_prefix.name + "_results.json")
    frontier_csv = out_prefix.with_name(out_prefix.name + "_frontier.csv")
    attempts_json = out_prefix.with_name(out_prefix.name + "_attempts.json")
    progress_path.write_text("", encoding="utf-8")
    rows: List[Dict] = []
    attempts_all: List[Dict] = []
    script_path = Path(__file__).resolve()
    for input_len in input_lengths:
        for group in selected_groups:
            safe_found: Optional[Dict] = None
            runnable_found: Optional[Dict] = None
            safe_conc = 0
            runnable_conc = 0
            concurrency_candidates = frontier_candidates_for_input(args, int(input_len))
            for concurrency in concurrency_candidates:
                safe_best, runnable_best, attempts = frontier_search_for_case(script_path, args, group, int(input_len), int(concurrency), progress_path=progress_path)
                attempts_all.extend(attempts)
                for rec in attempts:
                    write_progress(progress_path, {"event": "frontier_attempt", **rec})
                if runnable_found is None and runnable_best is not None:
                    runnable_found = dict(runnable_best)
                    runnable_conc = int(concurrency)
                    runnable_found["gpu_mem_frac_best_runnable"] = float(runnable_best.get("gpu_mem_frac_effective", runnable_best.get("gpu_mem_frac_initial", 0.0)))
                    runnable_found["max_supported_concurrency_runnable"] = int(concurrency)
                    write_progress(progress_path, {"event": "frontier_selected_runnable", **runnable_found})
                if safe_found is None and safe_best is not None:
                    safe_found = dict(safe_best)
                    safe_conc = int(concurrency)
                    safe_found["gpu_mem_frac_best_safe"] = float(safe_best.get("gpu_mem_frac_effective", safe_best.get("gpu_mem_frac_initial", 0.0)))
                    safe_found["max_supported_concurrency_safe"] = int(concurrency)
                    write_progress(progress_path, {"event": "frontier_selected_safe", **safe_found})
                if safe_found is not None and runnable_found is not None:
                    break
            rec = {
                "group": group,
                "input_len": int(input_len),
                "safe": safe_found or {},
                "runnable": runnable_found or {},
                "max_supported_concurrency_safe": int(safe_conc),
                "gpu_mem_frac_best_safe": float((safe_found or {}).get("gpu_mem_frac_best_safe", 0.0)),
                "max_supported_concurrency_runnable": int(runnable_conc),
                "gpu_mem_frac_best_runnable": float((runnable_found or {}).get("gpu_mem_frac_best_runnable", 0.0)),
            }
            rows.append(rec)
            write_progress(progress_path, {"event": "frontier_selected", **rec})
    payload = {
        "meta": {
            "task": "exp1_capacity_frontier",
            "mode": "frontier",
            "model_name": args.model_name,
            "groups": selected_groups,
            "input_lengths": input_lengths,
            "frontier_concurrency_candidates": {str(int(k)): frontier_candidates_for_input(args, int(k)) for k in input_lengths},
            "gpu_mem_frac_range": [float(args.frontier_gpu_mem_frac_min), float(args.frontier_gpu_mem_frac_max)],
            "gpu_mem_frac_resolution": float(args.frontier_gpu_mem_frac_resolution),
            "chunk_size": int(getattr(args, "chunk_size", 0) or 0),
            "prefill_batch_size": int(getattr(args, "prefill_batch_size", 0) or 0),
            "frontier_refine_window": float(args.frontier_refine_window),
            "frontier_refine_step": float(args.frontier_refine_step),
            "safe_cuda_free_gb_min": float(args.safe_cuda_free_gb_min),
            "safe_cuda_free_frac_min": float(args.safe_cuda_free_frac_min),
            "max_new_tokens": int(args.max_new_tokens),
            "frontier_search_max_new_tokens": int(args.frontier_search_max_new_tokens),
            "frontier_final_eval_max_new_tokens": int(args.frontier_final_eval_max_new_tokens),
            "frontier_search_repeats": int(args.frontier_search_repeats),
            "frontier_semantics": ["safe", "runnable"],
            "prompt_profile": PROMPT_PROFILE,
            "metric_profile": METRIC_PROFILE,
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        },
        "frontier_rows": rows,
        "attempts": attempts_all,
    }
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    attempts_json.write_text(json.dumps({"attempts": attempts_all}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_frontier_csv(frontier_csv, rows)
    print(f"Saved: {result_json}")


def write_perf_frontier_csv(path: Path, rows: Sequence[Dict]) -> None:
    header = [
        "group",
        "input_len",
        "max_supported_concurrency_runnable",
        "gpu_mem_frac_effective",
        "tokens_per_sec",
        "decode_step_p95_ms",
        "cuda_free_min_gb",
        "cuda_free_post_cleanup_min_gb",
        "cuda_free_post_cleanup_last_gb",
        "decode_cuda_free_post_cleanup_recent_min_gb",
        "kv_peak_ratio",
        "p2_active_steps",
        "p2_no_candidate_steps",
        "p2_attempted_steps",
        "p2_success_steps",
        "p2_cuda_pressure_signal_steps",
    ]

    def peak_ratio(rec: Dict) -> float:
        peak = int(rec.get("kv_peak_used_blocks", 0))
        total = max(1, int(rec.get("kv_total_blocks", 0)))
        return float(peak / total)

    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rec in rows:
            run = dict(rec.get("runnable", {}))
            vals = [
                str(rec.get("group", "")),
                str(int(rec.get("input_len", 0))),
                str(int(rec.get("max_supported_concurrency_runnable", 0))),
                f"{float(rec.get('gpu_mem_frac_effective', 0.0)):.3f}",
                f"{float(run.get('tokens_per_sec', 0.0)):.4f}",
                f"{float(run.get('decode_step_p95_ms', 0.0)):.3f}",
                f"{float(run.get('cuda_free_min_gb', 0.0)):.4f}",
                f"{float(run.get('cuda_free_post_cleanup_min_gb', 0.0)):.4f}",
                f"{float(run.get('cuda_free_post_cleanup_last_gb', 0.0)):.4f}",
                f"{float(run.get('decode_cuda_free_post_cleanup_recent_min_gb', 0.0)):.4f}",
                f"{peak_ratio(run):.6f}",
                str(int(run.get('p2_active_steps', 0))),
                str(int(run.get('p2_no_candidate_steps', 0))),
                str(int(run.get('p2_attempted_steps', 0))),
                str(int(run.get('p2_success_steps', 0))),
                str(int(run.get('p2_cuda_pressure_signal_steps', 0))),
            ]
            f.write(",".join(vals) + "\n")


def orchestrator_perf_frontier_main(args: argparse.Namespace) -> None:
    input_lengths = parse_int_list(args.input_lengths)
    frac_candidates = parse_float_list(args.perf_frontier_gpu_mem_fracs)
    selected_groups = parse_groups(args.groups)
    out_prefix = Path(args.out_prefix)
    progress_path = out_prefix.with_suffix(".progress.jsonl")
    result_json = out_prefix.with_name(out_prefix.name + "_results.json")
    frontier_csv = out_prefix.with_name(out_prefix.name + "_frontier.csv")
    attempts_json = out_prefix.with_name(out_prefix.name + "_attempts.json")
    progress_path.write_text("", encoding="utf-8")

    rows: List[Dict] = []
    attempts_all: List[Dict] = []
    script_path = Path(__file__).resolve()

    for input_len in input_lengths:
        concurrency_candidates = perf_frontier_candidates_for_input(args, int(input_len))
        for group in selected_groups:
            best_run: Optional[Dict] = None
            best_conc = 0
            for concurrency in concurrency_candidates:
                conc_attempts: List[Dict] = []
                for frac in frac_candidates:
                    frac = round(float(frac), 6)
                    with tempfile.NamedTemporaryFile(prefix="exp1_case_", suffix=".json", delete=False) as tf:
                        worker_out = tf.name
                    write_progress(progress_path, {
                        "event": "perf_frontier_attempt_begin",
                        "group": group,
                        "input_len": int(input_len),
                        "concurrency": int(concurrency),
                        "gpu_mem_frac_initial": float(frac),
                        "eval_max_new_tokens": int(args.max_new_tokens),
                        "eval_repeats": int(args.repeats),
                    })
                    cmd = [
                        sys.executable,
                        str(script_path),
                        "--worker",
                        "--model-name", args.model_name,
                        "--max-new-tokens", str(int(args.max_new_tokens)),
                        "--repeats", str(int(args.repeats)),
                        "--gpu-mem-frac-fallback-step", "1.0",
                        "--gpu-mem-frac-min", str(frac),
                        "--chunk-size", str(int(getattr(args, "chunk_size", 0) or 0)),
                        "--prefill-batch-size", str(int(getattr(args, "prefill_batch_size", 0) or 0)),
                        "--decode-micro-batch-size", str(int(args.decode_micro_batch_size)),
                        "--decode-active-cap-initial", str(int(args.decode_active_cap_initial)),
                        "--max-decode-active-cap", str(int(args.max_decode_active_cap)),
                        "--worker-group", group,
                        "--worker-input-len", str(int(input_len)),
                        "--worker-concurrency", str(int(concurrency)),
                        "--worker-gpu-mem-frac-initial", str(frac),
                        "--worker-out", worker_out,
                    ]
                    proc = subprocess.run(cmd, cwd=str(script_path.parent), capture_output=True, text=True)
                    try:
                        rec = json.loads(Path(worker_out).read_text(encoding="utf-8"))
                    except Exception:
                        rec = {
                            "group": group,
                            "input_len": int(input_len),
                            "concurrency": int(concurrency),
                            "gpu_mem_frac_initial": float(frac),
                            "gpu_mem_frac_effective": float(frac),
                            "success": 0,
                            "success_rate": 0.0,
                            "oom": 0,
                            "tokens_per_sec": 0.0,
                            "decode_step_p95_ms": 0.0,
                            "cuda_free_min_gb": 0.0,
                            "cuda_free_post_cleanup_min_gb": 0.0,
                            "cuda_free_post_cleanup_last_gb": 0.0,
                            "decode_cuda_free_post_cleanup_recent_min_gb": 0.0,
                            "cuda_total_gb": 0.0,
                            "p2_cuda_pressure_signal_steps": 0,
                            "kv_total_blocks": 0,
                            "kv_peak_used_blocks": 0,
                            "p2_active_steps": 0,
                            "p2_candidate_steps": 0,
                            "p2_recovery_fail_windows": 0,
                            "p2_no_candidate_steps": 0,
                            "p2_attempted_steps": 0,
                            "p2_success_steps": 0,
                            "first_p2_step": -1,
                            "error_reason": f"worker_failed: rc={proc.returncode}; stderr_tail={proc.stderr[-400:]}",
                        }
                    finally:
                        try:
                            Path(worker_out).unlink(missing_ok=True)
                        except Exception:
                            pass
                    rec["mode"] = "perf_frontier"
                    rec["gpu_mem_frac_nominal"] = float(frac_candidates[0]) if frac_candidates else float(frac)
                    conc_attempts.append(rec)
                    write_progress(progress_path, {"event": "perf_frontier_attempt", **rec})
                    if case_is_runnable(rec):
                        best_run = dict(rec)
                        best_conc = int(concurrency)
                        break
                attempts_all.extend(conc_attempts)
                if best_run is not None:
                    break

            row = {
                "group": group,
                "input_len": int(input_len),
                "runnable": best_run or {},
                "max_supported_concurrency_runnable": int(best_conc),
                "gpu_mem_frac_effective": float((best_run or {}).get("gpu_mem_frac_effective", 0.0)),
            }
            rows.append(row)
            write_progress(progress_path, {"event": "perf_frontier_selected", **row})

    payload = {
        "meta": {
            "task": "exp1_capacity_perf_frontier",
            "mode": "perf_frontier",
            "model_name": args.model_name,
            "groups": selected_groups,
            "input_lengths": input_lengths,
            "perf_frontier_concurrency_candidates": {
                str(int(k)): perf_frontier_candidates_for_input(args, int(k)) for k in input_lengths
            },
            "perf_frontier_gpu_mem_fracs": [float(x) for x in frac_candidates],
            "chunk_size": int(getattr(args, "chunk_size", 0) or 0),
            "prefill_batch_size": int(getattr(args, "prefill_batch_size", 0) or 0),
            "max_new_tokens": int(args.max_new_tokens),
            "repeats": int(args.repeats),
            "prompt_profile": PROMPT_PROFILE,
            "metric_profile": METRIC_PROFILE,
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        },
        "perf_frontier_rows": rows,
        "attempts": attempts_all,
    }
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    attempts_json.write_text(json.dumps({"attempts": attempts_all}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_perf_frontier_csv(frontier_csv, rows)
    print(f"Saved: {result_json}")

def write_main_csv(path: Path, rows: Sequence[Dict], input_lengths: Sequence[int], conc_list: Sequence[int], groups: Sequence[str]) -> None:
    header = ["concurrency", "group"]
    for input_len in input_lengths:
        header.extend([f"{input_len}_tps", f"{input_len}_p95_ms"])
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for conc in conc_list:
            for group in groups:
                line = [str(int(conc)), group]
                for input_len in input_lengths:
                    rec = next(r for r in rows if r["group"] == group and int(r["concurrency"]) == int(conc) and int(r["input_len"]) == int(input_len))
                    if int(rec["success"]) == 1:
                        line.extend([f"{float(rec['tokens_per_sec']):.4f}", f"{float(rec['decode_step_p95_ms']):.3f}"])
                    else:
                        line.extend(["OOM", "-"])
                f.write(",".join(line) + "\n")


def write_stability_csv(path: Path, rows: Sequence[Dict]) -> None:
    header = [
        "group","input_len","concurrency","gpu_mem_frac_initial","gpu_mem_frac_effective",
        "model_memory_gb","kv_available_gb","success","oom","tokens_per_sec","decode_step_p95_ms",
        "decode_min_n_free","prefill_min_n_free","global_min_n_free","wm_low","wm_high","p2_low_threshold","p2_target_free_blocks","kv_total_blocks","kv_peak_used_blocks","thrash_win16",
        "decode_append_fail_count","decode_backpressure_events","decode_retry_timeout_fail_count","decode_no_progress_steps","p2_active_steps","p2_candidate_steps","p2_recovery_fail_windows","p2_no_candidate_steps","p2_attempted_steps","p2_success_steps","p2_ready_candidate_steps","p2_decode_candidate_steps","p2_expected_reclaim_blocks","p2_gain_success_steps","p2_gain_fail_steps","p2_skipped_low_benefit_steps","first_p2_step","offloader_delta","avg_decode_microbatch_size","error_reason",
        "cuda_free_min_gb","cuda_free_post_cleanup_min_gb","cuda_free_post_cleanup_last_gb","decode_cuda_free_post_cleanup_recent_min_gb","p2_cuda_pressure_signal_steps","compression_profile","prompt_profile"
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rec in rows:
            vals = [
                rec["group"], str(rec["input_len"]), str(rec["concurrency"]),
                f"{float(rec['gpu_mem_frac_initial']):.2f}", f"{float(rec['gpu_mem_frac_effective']):.2f}",
                f"{float(rec['model_memory_gb']):.4f}", f"{float(rec['kv_available_gb']):.4f}",
                str(int(rec["success"])), str(int(rec["oom"])), f"{float(rec['tokens_per_sec']):.4f}",
                f"{float(rec['decode_step_p95_ms']):.3f}", str(int(rec['decode_min_n_free'])),
                str(int(rec['prefill_min_n_free'])), str(int(rec['global_min_n_free'])),
                str(int(rec.get('wm_low', 0))), str(int(rec.get('wm_high', 0))), str(int(rec.get('p2_low_threshold', 0))), str(int(rec.get('p2_target_free_blocks', 0))),
                str(int(rec['kv_total_blocks'])), str(int(rec['kv_peak_used_blocks'])),
                f"{float(rec['thrash_win16']):.6f}",
                str(int(rec['decode_append_fail_count'])), str(int(rec['decode_backpressure_events'])),
                str(int(rec['decode_retry_timeout_fail_count'])), str(int(rec['decode_no_progress_steps'])),
                str(int(rec.get('p2_active_steps', 0))), str(int(rec.get('p2_candidate_steps', 0))), str(int(rec.get('p2_recovery_fail_windows', 0))),
                str(int(rec.get('p2_no_candidate_steps', 0))), str(int(rec.get('p2_attempted_steps', 0))), str(int(rec.get('p2_success_steps', 0))),
                str(int(rec.get('p2_ready_candidate_steps', 0))), str(int(rec.get('p2_decode_candidate_steps', 0))), str(int(rec.get('p2_expected_reclaim_blocks', 0))),
                str(int(rec.get('p2_gain_success_steps', 0))), str(int(rec.get('p2_gain_fail_steps', 0))), str(int(rec.get('p2_skipped_low_benefit_steps', 0))),
                str(int(rec.get('first_p2_step', -1))),
                json.dumps(dict(rec['offloader_delta']), ensure_ascii=False),
                f"{float(rec['avg_decode_microbatch_size']):.4f}",
                json.dumps(str(rec['error_reason']), ensure_ascii=False),
                f"{float(rec.get('cuda_free_min_gb', 0.0)):.4f}",
                f"{float(rec.get('cuda_free_post_cleanup_min_gb', 0.0)):.4f}",
                f"{float(rec.get('cuda_free_post_cleanup_last_gb', 0.0)):.4f}",
                f"{float(rec.get('decode_cuda_free_post_cleanup_recent_min_gb', 0.0)):.4f}",
                str(int(rec.get('p2_cuda_pressure_signal_steps', 0))),
                json.dumps(str(rec['compression_profile']), ensure_ascii=False),
                json.dumps(str(rec['prompt_profile']), ensure_ascii=False),
            ]
            f.write(",".join(vals) + "\n")


def write_capacity_csv(path: Path, summary_rows: Sequence[Dict], input_lengths: Sequence[int]) -> None:
    header = ["group"] + [f"max_supported_concurrency_{x}" for x in input_lengths]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rec in summary_rows:
            vals = [rec["group"]] + [str(int(rec[f"max_supported_concurrency_{x}"])) for x in input_lengths]
            f.write(",".join(vals) + "\n")


def orchestrator_main(args: argparse.Namespace) -> None:
    input_lengths = parse_int_list(args.input_lengths)
    concurrency_list = parse_int_list(args.concurrency_list)
    selected_groups = parse_groups(args.groups)
    frac_map = parse_frac_map(args.gpu_mem_frac_map)
    missing = [x for x in input_lengths if x not in frac_map]
    if missing:
        raise ValueError(f"missing gpu_mem_frac mapping for lengths: {missing}")
    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")

    out_prefix = Path(args.out_prefix)
    progress_path = out_prefix.with_suffix(".progress.jsonl")
    result_json = out_prefix.with_name(out_prefix.name + "_results.json")
    main_csv = out_prefix.with_name(out_prefix.name + "_main.csv")
    stability_csv = out_prefix.with_name(out_prefix.name + "_stability.csv")
    capacity_csv = out_prefix.with_name(out_prefix.name + "_capacity.csv")
    progress_path.write_text("", encoding="utf-8")

    rows: List[Dict] = []
    script_path = Path(__file__).resolve()
    for input_len in input_lengths:
        initial_frac = float(frac_map[input_len])
        for concurrency in concurrency_list:
            for group in selected_groups:
                with tempfile.NamedTemporaryFile(prefix="exp1_case_", suffix=".json", delete=False) as tf:
                    worker_out = tf.name
                cmd = [
                    sys.executable,
                    str(script_path),
                    "--worker",
                    "--model-name", args.model_name,
                    "--max-new-tokens", str(int(args.max_new_tokens)),
                    "--repeats", str(int(args.repeats)),
                    "--gpu-mem-frac-fallback-step", str(float(args.gpu_mem_frac_fallback_step)),
                    "--gpu-mem-frac-min", str(float(args.gpu_mem_frac_min)),
                    "--chunk-size", str(int(getattr(args, "chunk_size", 0) or 0)),
                    "--prefill-batch-size", str(int(getattr(args, "prefill_batch_size", 0) or 0)),
                    "--decode-micro-batch-size", str(int(args.decode_micro_batch_size)),
                    "--decode-active-cap-initial", str(int(args.decode_active_cap_initial)),
                    "--max-decode-active-cap", str(int(args.max_decode_active_cap)),
                    "--worker-group", group,
                    "--worker-input-len", str(int(input_len)),
                    "--worker-concurrency", str(int(concurrency)),
                    "--worker-gpu-mem-frac-initial", str(initial_frac),
                    "--worker-out", worker_out,
                ]
                proc = subprocess.run(cmd, cwd=str(script_path.parent), capture_output=True, text=True)
                rec = None
                try:
                    rec = json.loads(Path(worker_out).read_text(encoding="utf-8"))
                except Exception:
                    rec = {
                        "group": group,
                        "input_len": int(input_len),
                        "concurrency": int(concurrency),
                        "gpu_mem_frac_initial": float(initial_frac),
                        "gpu_mem_frac_effective": float(initial_frac),
                        "model_memory_gb": 0.0,
                        "kv_available_gb": 0.0,
                        "success": 0,
                        "oom": 0,
                        "tokens_per_sec": 0.0,
                        "decode_step_p95_ms": 0.0,
                        "decode_min_n_free": 0,
                        "prefill_min_n_free": 0,
                        "global_min_n_free": 0,
                        "kv_total_blocks": 0,
                        "kv_peak_used_blocks": 0,
                        "thrash_win16": 0.0,
                        "decode_append_fail_count": 0,
                        "decode_backpressure_events": 0,
                        "decode_retry_timeout_fail_count": 0,
                        "decode_no_progress_steps": 0,
                        "p2_active_steps": 0,
                        "p2_candidate_steps": 0,
                        "p2_recovery_fail_windows": 0,
                        "p2_no_candidate_steps": 0,
                        "p2_attempted_steps": 0,
                        "p2_success_steps": 0,
                        "first_p2_step": -1,
                        "offloader_delta": {},
                        "avg_decode_microbatch_size": 0.0,
                        "decode_path_selected": "",
                        "decode_path_fallback_count": 0,
                        "decode_path_fallback_reason_topk": {},
                        "peak_gpu_mem_allocated_gb": 0.0,
                        "error_reason": f"worker_failed: rc={proc.returncode}; stderr_tail={proc.stderr[-400:]}",
                        "compression_profile": compression_profile_for_group(group),
                        "prompt_profile": PROMPT_PROFILE,
                        "metric_profile": METRIC_PROFILE,
                        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
                        "actual_prompt_tokens": [],
                        "wall_ms": 0.0,
                    }
                finally:
                    try:
                        Path(worker_out).unlink(missing_ok=True)
                    except Exception:
                        pass
                rows.append(rec)
                write_progress(progress_path, rec)

    capacity_rows = derive_capacity_summary(rows, input_lengths, selected_groups)
    payload = {
        "meta": {
            "task": "exp1_capacity_throughput_table",
            "model_name": args.model_name,
            "groups": selected_groups,
            "input_lengths": input_lengths,
            "concurrency_list": concurrency_list,
            "max_new_tokens": int(args.max_new_tokens),
            "chunk_size": int(getattr(args, "chunk_size", 0) or 0),
            "prefill_batch_size": int(getattr(args, "prefill_batch_size", 0) or 0),
            "gpu_mem_frac_map": {str(k): float(v) for k, v in frac_map.items()},
            "gpu_mem_frac_fallback_step": float(args.gpu_mem_frac_fallback_step),
            "gpu_mem_frac_min": float(args.gpu_mem_frac_min),
            "decode_micro_batch_size": int(args.decode_micro_batch_size),
            "decode_active_cap_initial": int(args.decode_active_cap_initial),
            "max_decode_active_cap": int(args.max_decode_active_cap),
            "gpu_mem_frac_semantics": "fraction of KV-available memory after model weights and 10% reserve",
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
            "alloc_conf_value": str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
            "prompt_profile": PROMPT_PROFILE,
            "metric_profile": METRIC_PROFILE,
            "compression_profiles": {
                group: compression_profile_for_group(group) for group in selected_groups
            },
        },
        "rows": rows,
        "capacity_summary": capacity_rows,
    }
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_main_csv(main_csv, rows, input_lengths, concurrency_list, selected_groups)
    write_stability_csv(stability_csv, rows)
    write_capacity_csv(capacity_csv, capacity_rows, input_lengths)
    print(f"Saved: {result_json}")
    print(f"Saved: {main_csv}")
    print(f"Saved: {stability_csv}")
    print(f"Saved: {capacity_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 1 concurrency capacity/throughput table")
    parser.add_argument("--model-name", type=str, default=LOCAL_MODEL_PATH)
    parser.add_argument("--mode", type=str, default="frontier", choices=["table", "frontier", "perf_frontier"])
    parser.add_argument(
        "--groups",
        type=str,
        default=",".join(DEFAULT_MAINLINE_GROUPS),
        help=f"comma-separated subset of: {','.join(ALL_GROUPS)}",
    )
    parser.add_argument("--input-lengths", type=str, default="8192,16384,32768")
    parser.add_argument("--concurrency-list", type=str, default="1,2,4,8,16")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--gpu-mem-frac-map", type=str, default="8192:0.80,16384:0.60,32768:0.40")
    parser.add_argument("--frontier-concurrency-candidates", type=str, default="1,2,4,8,16,24,32,48,64")
    parser.add_argument("--frontier-concurrency-map", type=str, default="8192:128,96,64,32,16,8,4,2,1;16384:64,32,16,8,4,2,1;32768:32,16,8,4,2,1")
    parser.add_argument("--perf-frontier-concurrency-map", type=str, default="8192:256,224,192,160,128,96,80,64,48,32,24,16,8,4,2,1;16384:128,112,96,80,64,56,48,40,32,24,16,8,4,2,1;32768:96,80,64,56,48,40,32,24,16,8,4,2,1")
    parser.add_argument("--perf-frontier-gpu-mem-fracs", type=str, default="0.92,0.90,0.88,0.86,0.84")
    parser.add_argument("--frontier-gpu-mem-frac-min", type=float, default=DEFAULT_GPU_MEM_FRAC_MIN)
    parser.add_argument("--frontier-gpu-mem-frac-max", type=float, default=DEFAULT_FRONTIER_GPU_MEM_FRAC_MAX)
    parser.add_argument("--frontier-gpu-mem-frac-resolution", type=float, default=DEFAULT_FRONTIER_GPU_MEM_FRAC_RESOLUTION)
    parser.add_argument("--frontier-refine-window", type=float, default=DEFAULT_FRONTIER_REFINE_WINDOW)
    parser.add_argument("--frontier-refine-step", type=float, default=DEFAULT_FRONTIER_REFINE_STEP)
    parser.add_argument("--frontier-search-max-new-tokens", type=int, default=DEFAULT_FRONTIER_SEARCH_MAX_NEW_TOKENS)
    parser.add_argument("--frontier-final-eval-max-new-tokens", type=int, default=DEFAULT_FRONTIER_FINAL_EVAL_MAX_NEW_TOKENS)
    parser.add_argument("--frontier-search-repeats", type=int, default=DEFAULT_FRONTIER_SEARCH_REPEATS)
    parser.add_argument("--safe-cuda-free-gb-min", type=float, default=DEFAULT_SAFE_CUDA_FREE_GB_MIN)
    parser.add_argument("--safe-cuda-free-frac-min", type=float, default=DEFAULT_SAFE_CUDA_FREE_FRAC_MIN)
    parser.add_argument("--gpu-mem-frac-fallback-step", type=float, default=DEFAULT_GPU_MEM_FRAC_FALLBACK_STEP)
    parser.add_argument("--gpu-mem-frac-min", type=float, default=DEFAULT_GPU_MEM_FRAC_MIN)
    parser.add_argument("--decode-micro-batch-size", type=int, default=0)
    parser.add_argument("--prefill-batch-size", type=int, default=0)
    parser.add_argument("--decode-active-cap-initial", type=int, default=0)
    parser.add_argument("--max-decode-active-cap", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--out-prefix", type=str, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-group", type=str, default="")
    parser.add_argument("--worker-input-len", type=int, default=0)
    parser.add_argument("--worker-concurrency", type=int, default=0)
    parser.add_argument("--worker-gpu-mem-frac-initial", type=float, default=0.0)
    parser.add_argument("--worker-out", type=str, default="")
    parser.add_argument("--worker-step-jsonl", type=str, default="")
    parser.add_argument("--worker-step-jsonl-every", type=int, default=1)
    parser.add_argument("--worker-fixed-gpu-mem-frac", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.worker:
        if not args.worker_group or not args.worker_out:
            raise ValueError("worker mode requires group and worker_out")
        worker_main(args)
    else:
        mode = str(args.mode).strip().lower()
        if mode == "frontier":
            orchestrator_frontier_main(args)
        elif mode == "perf_frontier":
            orchestrator_perf_frontier_main(args)
        else:
            orchestrator_main(args)


if __name__ == "__main__":
    main()

