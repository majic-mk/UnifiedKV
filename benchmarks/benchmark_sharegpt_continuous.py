import argparse
import gc
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    import torch
except Exception:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
CONFIGS_DIR = BENCHMARKS_DIR / "configs"
CORE_DIR = PROJECT_ROOT / "core"
for path in (BENCHMARKS_DIR, CONFIGS_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_internal_common import (
    build_engine,
    cleanup_engine,
    compression_profile_for_method,
    is_hf_style_method,
    is_internal_method,
)
from hf_style_common import run_hf_style_continuous
from benchmark_sharegpt_serving import (
    count_tokens,
    eval_sharegpt_output,
    load_records,
    resolve_dataset_path,
    scan_sharegpt_candidates,
    select_sharegpt_samples,
    summarize_int_distribution,
    valid_sharegpt_output,
)
from benchmark_vllm_common import (
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_SERVER_HOST,
    OOM_KEYWORDS,
    VLLM_SERVED_MODEL_NAME,
    aggregate_streaming_metrics,
    classify_frontier_reason,
    classify_status_from_rates,
    load_tokenizer,
    percentile,
    start_vllm_server,
    stop_vllm_server,
    stream_chat_completion,
)

MIN_KEYS = {"global_min_n_free", "decode_min_n_free", "prefill_min_n_free", "n_free", "n_free_final", "cuda_free_min_gb"}
MAX_KEYS = [
    "kv_total_blocks", "kv_peak_used_blocks", "decode_backpressure_events", "decode_memory_cap_events",
    "decode_retry_timeout_fail_count", "decode_retry_cooldown_enabled", "decode_retry_cooldown_events",
    "decode_retry_cooldown_bypass_events", "decode_retry_cooldown_active",
    "guard_seen_count", "guard_effective_shrink_count", "decode_length_bucketed_steps",
    "decode_length_bucket_subbatch_count", "decode_active_cap", "decode_active_cap_boot",
    "decode_active_cap_min_seen", "ready_decode_resident_blocks", "decode_active_resident_blocks",
    "p2_active_steps", "p2_candidate_steps", "p2_no_candidate_steps", "p2_attempted_steps",
    "p2_success_steps", "p2_ready_candidate_steps", "p2_decode_candidate_steps", "p2_last_candidate_count",
    "p2_last_no_candidate", "p2_reject_deferred", "p2_reject_no_resident", "p2_reject_protected",
    "p2_ready_protected_ignored", "p2_reject_active_floor", "p2_reject_plan_empty", "p2_managed_active",
    "p2_expected_reclaim_blocks", "p2_ready_offload_blocks_total", "p2_ready_offload_blocks_last", "p2_ready_offload_sequence_steps", "p2_ready_offload_decode_steps",
    "p2_ready_sequences_selected_per_step", "p2_ready_offload_blocks_per_step", "p2_ready_target_reclaim_blocks", "p2_ready_actual_reclaim_blocks",
    "p2_ready_stop_target_reached_steps", "p2_ready_stop_sequence_cap_reached_steps", "p2_ready_stop_block_cap_reached_steps",
    "p2_ready_stop_low_benefit_skip_steps", "p2_ready_stop_not_needed_steps", "p2_ready_stop_no_ready_candidate_steps",
    "kv_admission_enabled", "kv_admission_blocked_steps", "kv_admission_blocked_requests", "kv_admission_last_free_blocks",
    "kv_admission_last_required_blocks", "kv_admission_last_prompt_blocks", "kv_admission_last_prompt_resident_blocks",
    "kv_admission_last_request_blocks", "kv_admission_last_pending_prompt_blocks", "kv_admission_last_pending_output_blocks", "kv_admission_last_output_reserve_tokens", "kv_admission_last_output_reserve_blocks", "kv_admission_last_allowed",
    "online_prefill_admission_enabled", "online_prefill_admission_blocked_steps",
    "online_prefill_admission_blocked_requests", "online_prefill_admission_last_prompt_len",
    "online_prefill_admission_last_cuda_free_gb", "online_prefill_admission_last_active_short",
    "online_prefill_admission_last_active_mid", "online_prefill_admission_last_active_long",
    "online_prefill_admission_last_cap", "online_prefill_active_token_budget",
    "online_prefill_admission_last_active_tokens", "online_prefill_admission_last_projected_tokens",
    "online_prefill_admission_last_token_budget",
    "online_prefill_admission_token_budget_blocked_steps", "online_prefill_admission_token_budget_blocked_requests",
    "online_prefill_admission_last_allowed",
    "online_prefill_chunk_floor_pause_steps", "online_prefill_chunk_floor_last_chunk_cap",
    "p2_gain_success_steps", "p2_gain_fail_steps", "p2_skipped_low_benefit_steps",
    "first_p2_step", "p2_low_threshold", "wm_low", "wm_high", "cuda_alloc_peak_gb", "cuda_reserved_peak_gb",
    "decode_paged_flash_strict", "decode_paged_direct_steps", "decode_rebuild_steps",
    "decode_materialize_kv_bytes", "decode_paged_direct_resident_miss_steps",
    "decode_page16_native_strict", "decode_page16_native_steps",
    "decode_page16_native_resident_miss_steps", "decode_page16_native_kernel_ms",
    "kv_logical_block_size", "retain_budget_tokens", "effective_retained_tokens",
    "retained_block_count", "selected_writeback_enabled",
    "score_full_attention_materialized",
]
ALL_METRIC_KEYS = list(MIN_KEYS) + MAX_KEYS


def normalize_method(name: str) -> str:
    value = str(name).strip()
    return {
        "p2": "p2_only_compress",
        "P2": "p2_only_compress",
        "p2online": "p2_page16_online",
        "p2_online": "p2_page16_online",
        "p2offline": "p2_page16_offline",
        "p2_offline": "p2_page16_offline",
        "offcompress": "off_compress",
    }.get(value, value)


def parse_methods(text: str) -> List[str]:
    methods = [normalize_method(x) for x in str(text or "").split(",") if str(x).strip()]
    if not methods:
        raise ValueError("empty methods")
    for method in methods:
        if method == "vllm":
            continue
        if not is_hf_style_method(method) and not is_internal_method(method):
            raise ValueError(f"unknown method: {method}")
    return methods


def parse_frac_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for chunk in str(text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, v = chunk.split(":", 1)
        out[normalize_method(k)] = float(v)
    return out


def summarize_samples(samples: Sequence[Dict[str, Any]], candidate_count: int) -> Dict[str, Any]:
    prompts = [int(x["prompt_tokens"]) for x in samples]
    targets = [int(x["target_tokens"]) for x in samples]
    return {
        "candidate_count": int(candidate_count),
        "sample_count": int(len(samples)),
        "prompt_tokens": summarize_int_distribution(prompts),
        "target_tokens": summarize_int_distribution(targets),
        "sample_signature": [
            {
                "record_idx": int(x["record_idx"]),
                "turn_idx": int(x["turn_idx"]),
                "prompt_tokens": int(x["prompt_tokens"]),
                "target_tokens": int(x["target_tokens"]),
            }
            for x in samples
        ],
    }


def add_step_metrics(summary: Dict[str, Any], step_rows: Sequence[Dict[str, Any]]) -> None:
    for key in ALL_METRIC_KEYS:
        vals = []
        for row in step_rows:
            val = row.get(key)
            if isinstance(val, bool):
                vals.append(float(int(val)))
            elif isinstance(val, (int, float)):
                vals.append(float(val))
        if not vals:
            continue
        agg = min(vals) if key in MIN_KEYS else max(vals)
        summary[key] = int(agg) if abs(agg - int(agg)) < 1e-9 else float(agg)
    summary["step_count"] = int(len(step_rows))
    for key in (
        "decode_backend",
        "decode_paged_direct_blocked_reason",
        "decode_page16_native_blocked_reason",
        "p2_ready_stop_reason",
        "compression_mode",
        "prefill_writeback_backend",
        "online_prefill_admission_last_reason",
        "online_prefill_admission_last_bucket",
    ):
        for row in reversed(list(step_rows)):
            val = str(row.get(key, "")).strip()
            if val:
                summary[key] = val
                break

    actual_active_vals = []
    for row in step_rows:
        scheduled = int(row.get("decode_scheduled", 0) or 0)
        decode_tokens = int(row.get("decode_tokens", 0) or 0)
        # Count only decode steps that made progress; exclude prefill-only, idle,
        # and no-progress steps so the average is not artificially deflated.
        if scheduled > 0 and decode_tokens > 0:
            actual_active_vals.append(scheduled)
    if actual_active_vals:
        summary["actual_active_sum"] = int(sum(actual_active_vals))
        summary["actual_active_count"] = int(len(actual_active_vals))
        summary["actual_active_avg"] = float(round(sum(actual_active_vals) / len(actual_active_vals), 3))
        summary["actual_active_p95"] = float(round(percentile(actual_active_vals, 0.95), 3))
        summary["actual_active_max"] = int(max(actual_active_vals))
    else:
        summary["actual_active_sum"] = 0
        summary["actual_active_count"] = 0
        summary["actual_active_avg"] = 0.0
        summary["actual_active_p95"] = 0.0
        summary["actual_active_max"] = 0

    resident_miss_steps = 0
    for key in (
        "decode_page16_native_resident_miss_steps",
        "resident_miss_steps",
        "decode_paged_direct_resident_miss_steps",
    ):
        if key in summary:
            resident_miss_steps = int(summary.get(key, 0) or 0)
            break
    summary["resident_miss_steps"] = int(resident_miss_steps)

    reason_counts: Dict[str, int] = {}
    for row in step_rows:
        reason = str(row.get("p2_ready_stop_reason", "")).strip()
        if reason and reason != "none":
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
    if reason_counts:
        summary["p2_ready_stop_reason_counts"] = reason_counts


def summarize_internal_rows(rows: Sequence[Dict[str, Any]], step_rows: Sequence[Dict[str, Any]], total_requested: int, total_wall_ms: float, method: str, gpu_mem_frac: float, use_target_tokens: bool = False, max_request_new_tokens: int = 0) -> Dict[str, Any]:
    requested = max(1, int(total_requested))
    completed = sum(int(r.get("completed", 0)) for r in rows)
    valid = sum(int(r.get("valid", 0)) for r in rows)
    total_tokens = sum(int(r.get("completion_tokens", 0)) for r in rows)
    errors = [str(r.get("error_reason", "")).strip() for r in rows if str(r.get("error_reason", "")).strip()]
    oom_count = sum(1 for err in errors if any(word in err.lower() for word in OOM_KEYWORDS) or "timeout" in err.lower())
    completion_rate = completed / requested
    valid_rate = valid / requested
    status = classify_status_from_rates(completion_rate, valid_rate, oom_count, "; ".join(errors[:3]))
    latencies = [float(r.get("request_latency_ms", 0.0)) for r in rows if float(r.get("request_latency_ms", 0.0)) > 0]
    ttfts = [float(r.get("ttft_ms", 0.0)) for r in rows if float(r.get("ttft_ms", 0.0)) > 0]
    itls = [float(r.get("avg_itl_ms", 0.0)) for r in rows if float(r.get("avg_itl_ms", 0.0)) > 0]
    total_blocks = max((int(r.get("kv_total_blocks", 0)) for r in step_rows if int(r.get("kv_total_blocks", 0)) > 0), default=0)
    min_free_candidates = []
    for row in step_rows:
        for key in ("global_min_n_free", "decode_min_n_free", "prefill_min_n_free", "n_free"):
            value = int(row.get(key, -1))
            if value >= 0:
                min_free_candidates.append(value)
    min_free_blocks = min(min_free_candidates) if min_free_candidates else -1
    summary: Dict[str, Any] = {
        "method": method,
        "serving_mode": "continuous_refill",
        "gpu_mem_frac": float(gpu_mem_frac),
        "requested_requests": int(total_requested),
        "completed_requests": int(completed),
        "completion_rate": float(completion_rate),
        "valid_completion_rate": float(valid_rate),
        "oom_failure_count": int(oom_count),
        "status": status,
        "frontier_reason": classify_frontier_reason(status, "; ".join(errors[:3]), completion_rate, valid_rate),
        "generated_tokens": int(total_tokens),
        "tokens_per_sec": float((1000.0 * total_tokens / total_wall_ms) if total_wall_ms > 0 else 0.0),
        "wall_clock_total_runtime_ms": float(round(total_wall_ms, 3)),
        "request_latency_p95_ms": float(round(percentile(latencies, 0.95), 3)),
        "request_latency_p99_ms": float(round(percentile(latencies, 0.99), 3)),
        "ttft_p95_ms": float(round(percentile(ttfts, 0.95), 3)),
        "ttft_p99_ms": float(round(percentile(ttfts, 0.99), 3)),
        "itl_p95_ms": float(round(percentile(itls, 0.95), 3)),
        "itl_p99_ms": float(round(percentile(itls, 0.99), 3)),
        "min_free_blocks": int(min_free_blocks),
        "min_free_block_ratio": float(round(min_free_blocks / max(1, total_blocks), 6)) if min_free_blocks >= 0 and total_blocks > 0 else -1.0,
        "kv_total_blocks": int(total_blocks),
        "error_reason": "; ".join(errors[:3]),
        "compression_profile": compression_profile_for_method(method),
        "generation_length_policy": "target_tokens" if bool(use_target_tokens) else "fixed_max_new_tokens",
        "max_request_new_tokens": int(max_request_new_tokens),
        "timing_note": "Internal continuous runner reports engine-side request latency, TTFT, and avg ITL from per-request timing fields.",
        "rows": list(rows),
    }
    add_step_metrics(summary, step_rows)
    return summary


def sample_generation_limit(item: Dict[str, Any], fallback_max_new_tokens: int, use_target_tokens: bool) -> int:
    if bool(use_target_tokens):
        try:
            target = int(item.get("target_tokens", 0) or 0)
        except Exception:
            target = 0
        if target > 0:
            return max(1, target)
    return max(1, int(fallback_max_new_tokens))


def max_generation_limit(samples: Sequence[Dict[str, Any]], fallback_max_new_tokens: int, use_target_tokens: bool) -> int:
    if not bool(use_target_tokens):
        return max(1, int(fallback_max_new_tokens))
    vals = [sample_generation_limit(x, fallback_max_new_tokens, True) for x in samples]
    return max(vals) if vals else max(1, int(fallback_max_new_tokens))


def run_internal_continuous(model_name: str, method: str, samples: Sequence[Dict[str, Any]], concurrency: int, max_new_tokens: int, repeats: int, gpu_mem_frac: float, use_target_tokens: bool = False) -> Dict[str, Any]:
    tokenizer = load_tokenizer(model_name)
    engine = None
    rows: List[Dict[str, Any]] = []
    step_rows: List[Dict[str, Any]] = []
    total_wall_ms = 0.0
    total_requested = int(len(samples) * max(1, repeats))
    try:
        engine_max_new_tokens = max_generation_limit(samples, max_new_tokens, use_target_tokens)
        engine = build_engine(model_name, method, gpu_mem_frac, engine_max_new_tokens)
        for repeat_idx in range(max(1, repeats)):
            engine._reset_online_runtime(clear_request_counter=True)
            engine._return_details_online = False
            if torch is not None and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            rid_to_item: Dict[int, Dict[str, Any]] = {}
            submit_times: Dict[int, float] = {}
            submitted = 0

            def admission_allows_next() -> bool:
                if submitted >= len(samples):
                    return False
                item = samples[submitted]
                if hasattr(engine, "can_admit_prompt_tokens"):
                    req_max_new_tokens = sample_generation_limit(item, max_new_tokens, use_target_tokens)
                    check = engine.can_admit_prompt_tokens(int(item.get("prompt_tokens", 0)), int(req_max_new_tokens))
                    return bool(int(check.get("allowed", 1)))
                return True

            def submit_next() -> None:
                nonlocal submitted
                item = dict(samples[submitted])
                req_max_new_tokens = sample_generation_limit(item, max_new_tokens, use_target_tokens)
                item["request_max_new_tokens"] = int(req_max_new_tokens)
                rid = int(engine.submit(str(item["prompt"]), max_new_tokens=req_max_new_tokens))
                rid_to_item[rid] = item
                submit_times[rid] = time.perf_counter()
                submitted += 1

            for _ in range(min(int(concurrency), len(samples))):
                if not admission_allows_next():
                    break
                submit_next()
            t0 = time.perf_counter()
            last_print = t0
            while engine.has_pending_requests():
                stat = dict(engine.step())
                stat["repeat_idx"] = repeat_idx
                step_rows.append(stat)
                now = time.perf_counter()
                for fin in engine.collect_finished():
                    rid = int(fin.get("request_id", -1))
                    item = rid_to_item.pop(rid, {})
                    submit_t = submit_times.pop(rid, now)
                    text = str(fin.get("output", "") or "")
                    token_ids = list(fin.get("token_ids", []) or [])
                    comp_tokens = int(len(token_ids) if token_ids else count_tokens(tokenizer, text))
                    engine_wall_ms = float(fin.get("wall_ms", 0.0) or 0.0)
                    latency_ms = engine_wall_ms if engine_wall_ms > 0.0 else float((now - submit_t) * 1000.0)
                    ttft_ms = float(fin.get("ttft_ms", 0.0) or 0.0)
                    avg_itl_ms = float(fin.get("avg_itl_ms", 0.0) or 0.0)
                    extra = dict(eval_sharegpt_output(dict(item), text) or {})
                    valid = int(extra.pop("valid", int(valid_sharegpt_output(text))))
                    err = str(fin.get("error", "") or "")
                    completed = int(str(fin.get("state", "")) == "DONE" and bool(text.strip()))
                    if avg_itl_ms <= 0.0 and completed:
                        avg_itl_ms = latency_ms / max(1, comp_tokens)
                    rows.append({
                        "repeat_idx": repeat_idx,
                        "request_id": rid,
                        "record_idx": int(item.get("record_idx", -1)),
                        "turn_idx": int(item.get("turn_idx", -1)),
                        "prompt_tokens": int(item.get("prompt_tokens", 0) or 0),
                        "target_tokens": int(item.get("target_tokens", 0) or 0),
                        "request_max_new_tokens": int(item.get("request_max_new_tokens", max_new_tokens) or max_new_tokens),
                        "completed": completed,
                        "valid": int(valid if completed else 0),
                        "completion_tokens": comp_tokens,
                        "request_latency_ms": round(latency_ms, 3),
                        "wall_ms": round(latency_ms, 3),
                        "ttft_ms": round(ttft_ms, 3),
                        "avg_itl_ms": round(avg_itl_ms, 3) if completed else 0.0,
                        "first_token_source": str(fin.get("first_token_source", "") or ""),
                        "min_free_blocks": int(stat.get("global_min_n_free", -1)),
                        "error_reason": err,
                        "output_text": text,
                        **extra,
                    })
                while submitted < len(samples) and int(engine._pending_request_count()) < int(concurrency):
                    if not admission_allows_next():
                        break
                    submit_next()
                if now - last_print >= 60.0:
                    print(json.dumps({
                        "method": method,
                        "submitted": submitted,
                        "completed": len(rows),
                        "pending": int(engine._pending_request_count()),
                        "step": int(stat.get("step", 0)),
                        "n_free": int(stat.get("n_free", -1)),
                        "decode_backpressure_events": int(stat.get("decode_backpressure_events", 0)),
                        "decode_memory_cap_events": int(stat.get("decode_memory_cap_events", 0)),
                        "decode_retry_timeout_fail_count": int(stat.get("decode_retry_timeout_fail_count", 0)),
                        "decode_retry_cooldown_events": int(stat.get("decode_retry_cooldown_events", 0)),
                        "p2_attempted_steps": int(stat.get("p2_attempted_steps", 0)),
                        "kv_admission_blocked_steps": int(stat.get("kv_admission_blocked_steps", 0)),
                        "p2_ready_actual_reclaim_blocks": int(stat.get("p2_ready_actual_reclaim_blocks", 0)),
                    }, ensure_ascii=False), flush=True)
                    last_print = now
            total_wall_ms += (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        err = str(exc)
        while len(rows) < total_requested:
            rows.append({"completed": 0, "valid": 0, "completion_tokens": 0, "request_latency_ms": 0.0, "wall_ms": 0.0, "ttft_ms": 0.0, "avg_itl_ms": 0.0, "min_free_blocks": -1, "error_reason": err, "output_text": ""})
        if total_wall_ms <= 0:
            total_wall_ms = 1.0
    finally:
        cleanup_engine(engine)
    return summarize_internal_rows(rows, step_rows, total_requested, total_wall_ms, method, gpu_mem_frac, use_target_tokens, max_generation_limit(samples, max_new_tokens, use_target_tokens))


def run_vllm_continuous(model_name: str, samples: Sequence[Dict[str, Any]], concurrency: int, max_new_tokens: int, repeats: int, gpu_memory_utilization: float, request_timeout_s: float, server_log: str, base_url: str, vllm_ignore_eos: bool = False, use_target_tokens: bool = False) -> Dict[str, Any]:
    tokenizer = load_tokenizer(model_name)
    endpoint = str(base_url).strip()
    proc = None
    launched = False
    if not endpoint:
        max_model_len = max(int(x["prompt_tokens"]) + sample_generation_limit(x, max_new_tokens, use_target_tokens) for x in samples) + 512
        proc, port = start_vllm_server(model_name=model_name, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, host=DEFAULT_SERVER_HOST, log_path=Path(server_log) if str(server_log).strip() else None)
        endpoint = f"http://{DEFAULT_SERVER_HOST}:{int(port)}"
        launched = True
    rows: List[Dict[str, Any]] = []
    total_wall_ms = 0.0
    try:
        for repeat_idx in range(max(1, repeats)):
            submitted = 0
            pending: Dict[Any, int] = {}
            t0 = time.perf_counter()
            last_print = t0

            def worker(idx: int) -> Dict[str, Any]:
                item = samples[idx]
                try:
                    req_max_new_tokens = sample_generation_limit(item, max_new_tokens, use_target_tokens)
                    rec = stream_chat_completion(endpoint, str(item["prompt"]), req_max_new_tokens, tokenizer, request_timeout_s=request_timeout_s, ignore_eos=vllm_ignore_eos)
                    rec["request_max_new_tokens"] = int(req_max_new_tokens)
                    rec["error_reason"] = ""
                    return rec
                except Exception as exc:
                    return {"output_text": "", "completion_tokens": 0, "ttft_ms": 0.0, "avg_itl_ms": 0.0, "finish_reason": "", "completed": 0, "wall_ms": 0.0, "error_reason": str(exc)}

            with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
                while submitted < len(samples) and len(pending) < int(concurrency):
                    fut = pool.submit(worker, submitted)
                    pending[fut] = submitted
                    submitted += 1
                while pending:
                    done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                    for fut in done:
                        idx = pending.pop(fut)
                        item = samples[idx]
                        rec = dict(fut.result())
                        rows.append({**rec, "repeat_idx": repeat_idx, "record_idx": int(item["record_idx"]), "turn_idx": int(item["turn_idx"]), "prompt_tokens": int(item["prompt_tokens"]), "target_tokens": int(item["target_tokens"])})
                    while submitted < len(samples) and len(pending) < int(concurrency):
                        fut = pool.submit(worker, submitted)
                        pending[fut] = submitted
                        submitted += 1
                    now = time.perf_counter()
                    if now - last_print >= 60.0:
                        print(json.dumps({"method": "vllm", "submitted": submitted, "completed": len(rows), "inflight": len(pending)}, ensure_ascii=False), flush=True)
                        last_print = now
            total_wall_ms += (time.perf_counter() - t0) * 1000.0
    finally:
        if launched:
            stop_vllm_server(proc)
    agg = aggregate_streaming_metrics(rows, requested_repeats=repeats, valid_fn=valid_sharegpt_output, batch_wall_ms_list=[total_wall_ms])
    return {
        "method": "vllm",
        "serving_mode": "continuous_refill",
        "served_model_name": VLLM_SERVED_MODEL_NAME,
        "gpu_memory_utilization": gpu_memory_utilization,
        "ignore_eos": bool(vllm_ignore_eos),
        "generation_length_policy": "target_tokens" if bool(use_target_tokens) else "fixed_max_new_tokens",
        "max_request_new_tokens": int(max((int(r.get("request_max_new_tokens", 0)) for r in rows), default=int(max_new_tokens))),
        "generated_tokens": int(sum(int(r.get("completion_tokens", 0)) for r in rows)),
        **agg,
        "rows": rows,
        "meta_note": "vLLM runner keeps at most concurrency in-flight streaming requests; KV block metrics are not exposed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", default="p2_only_compress")
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--prompt-len-min", type=int, default=3584)
    parser.add_argument("--prompt-len-max", type=int, default=32768)
    parser.add_argument("--target-len-min", type=int, default=128)
    parser.add_argument("--target-len-max", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Fallback/fixed generation cap. With --use-target-tokens, per-sample target_tokens is used instead.")
    parser.add_argument("--use-target-tokens", action="store_true", help="Use each ShareGPT sample target_tokens as its per-request max_new_tokens.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--gpu-mem-frac-map", default="p2_only_compress:0.35,off_compress:0.35,p2_page16:0.65,p2_page16_offline:0.65,p2_page16_online:0.65")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--vllm-ignore-eos", action="store_true", help="Ask vLLM to ignore EOS so generation reaches max_tokens when possible.")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--server-log", default="")
    parser.add_argument("--out", default="benchmarks/results/probes/sharegpt_continuous_longio_c64_r1/result.json")
    parser.add_argument("--samples-json", default="")
    parser.add_argument("--save-samples-json", default="")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    tokenizer = load_tokenizer(args.model_name)
    dataset_path = resolve_dataset_path(args.dataset)
    sample_source = "scan"
    if str(args.samples_json).strip():
        sample_path = Path(str(args.samples_json))
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            samples = [dict(x) for x in payload.get("samples", [])]
            candidate_count = int(payload.get("candidate_count", len(samples)))
        else:
            samples = [dict(x) for x in payload]
            candidate_count = len(samples)
        if not samples:
            raise RuntimeError(f"empty samples_json: {sample_path}")
        sample_source = str(sample_path)
    else:
        records = load_records(dataset_path)
        candidates = scan_sharegpt_candidates(tokenizer, records, args.prompt_len_min, args.prompt_len_max, args.target_len_min, args.target_len_max, progress_prefix="continuous_sharegpt", progress_every=5000)
        candidate_count = len(candidates)
        if len(candidates) < args.sample_count:
            raise RuntimeError(f"insufficient ShareGPT candidates: {len(candidates)} < requested sample_count={args.sample_count}")
        samples = select_sharegpt_samples(candidates, args.sample_count, args.seed)
        if str(args.save_samples_json).strip():
            sample_path = Path(str(args.save_samples_json))
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_payload = {
                "candidate_count": int(candidate_count),
                "sample_count": int(len(samples)),
                "seed": int(args.seed),
                "prompt_len_range": [int(args.prompt_len_min), int(args.prompt_len_max)],
                "target_len_range": [int(args.target_len_min), int(args.target_len_max)],
                "samples": samples,
            }
            sample_path.write_text(json.dumps(sample_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            sample_source = str(sample_path)
    sample_summary = summarize_samples(samples, candidate_count)
    public_sample_summary = {k: v for k, v in sample_summary.items() if k != "sample_signature"}
    print(json.dumps({"sample_summary": public_sample_summary, "sample_source": sample_source}, ensure_ascii=False), flush=True)

    frac_map = parse_frac_map(args.gpu_mem_frac_map)
    rows = []
    for method in methods:
        print(json.dumps({"starting_method": method, "serving_mode": "continuous_refill"}, ensure_ascii=False), flush=True)
        if method == "vllm":
            row = run_vllm_continuous(args.model_name, samples, args.concurrency, args.max_new_tokens, args.repeats, args.vllm_gpu_memory_utilization, args.request_timeout_s, args.server_log, args.base_url, args.vllm_ignore_eos, args.use_target_tokens)
        elif is_hf_style_method(method):
            row = run_hf_style_continuous(
                model_name=args.model_name,
                method=method,
                items=samples,
                concurrency=args.concurrency,
                max_new_tokens=args.max_new_tokens,
                repeats=args.repeats,
                evaluate_output_fn=eval_sharegpt_output,
                use_target_tokens=args.use_target_tokens,
            )
        else:
            if method not in frac_map:
                raise ValueError(f"missing gpu mem frac for method {method}")
            row = run_internal_continuous(args.model_name, method, samples, args.concurrency, args.max_new_tokens, args.repeats, frac_map[method], args.use_target_tokens)
        row["concurrency"] = int(args.concurrency)
        row["actual_prompt_tokens_mean"] = float(sum(int(x["prompt_tokens"]) for x in samples) / max(1, len(samples)))
        rows.append(row)
        print(json.dumps({
            "method": row.get("method"),
            "status": row.get("status"),
            "tokens_per_sec": row.get("tokens_per_sec"),
            "generated_tokens": row.get("generated_tokens"),
            "wall_clock_total_runtime_ms": row.get("wall_clock_total_runtime_ms"),
            "min_free_blocks": row.get("min_free_blocks"),
            "decode_backpressure_events": row.get("decode_backpressure_events"),
            "decode_memory_cap_events": row.get("decode_memory_cap_events"),
            "decode_retry_timeout_fail_count": row.get("decode_retry_timeout_fail_count"),
            "decode_retry_cooldown_events": row.get("decode_retry_cooldown_events"),
            "decode_retry_cooldown_bypass_events": row.get("decode_retry_cooldown_bypass_events"),
            "p2_attempted_steps": row.get("p2_attempted_steps"),
        }, ensure_ascii=False), flush=True)

    payload = {
        "meta": {
            "task": "sharegpt_continuous_serving",
            "serving_mode": "continuous_refill",
            "model_name": args.model_name,
            "dataset": str(dataset_path),
            "methods": methods,
            "sample_count": len(samples),
            "requested_sample_count": args.sample_count,
            "concurrency": args.concurrency,
            "prompt_len_range": [args.prompt_len_min, args.prompt_len_max],
            "target_len_range": [args.target_len_min, args.target_len_max],
            "max_new_tokens": args.max_new_tokens,
            "generation_length_policy": "target_tokens" if bool(args.use_target_tokens) else "fixed_max_new_tokens",
            "max_request_new_tokens": int(max_generation_limit(samples, args.max_new_tokens, args.use_target_tokens)),
            "repeats": args.repeats,
            "seed": args.seed,
            "env_flags": {
                "KV_MIDDLEWARE_DISABLE_DECODE_LENGTH_BUCKET": os.environ.get("KV_MIDDLEWARE_DISABLE_DECODE_LENGTH_BUCKET", ""),
                "KV_MIDDLEWARE_DISABLE_DECODE_MEMORY_GUARD": os.environ.get("KV_MIDDLEWARE_DISABLE_DECODE_MEMORY_GUARD", ""),
                "KV_MIDDLEWARE_P2_READY_IGNORE_RESIDENCY_PROTECTED": os.environ.get("KV_MIDDLEWARE_P2_READY_IGNORE_RESIDENCY_PROTECTED", ""),
                "KV_MIDDLEWARE_SINGLETON_RETRY_COOLDOWN": os.environ.get("KV_MIDDLEWARE_SINGLETON_RETRY_COOLDOWN", ""),
                "KV_MIDDLEWARE_SINGLETON_RETRY_COOLDOWN_MIN": os.environ.get("KV_MIDDLEWARE_SINGLETON_RETRY_COOLDOWN_MIN", ""),
                "KV_MIDDLEWARE_SINGLETON_RETRY_COOLDOWN_MAX": os.environ.get("KV_MIDDLEWARE_SINGLETON_RETRY_COOLDOWN_MAX", ""),
                "KV_MIDDLEWARE_KV_ADMISSION": os.environ.get("KV_MIDDLEWARE_KV_ADMISSION", ""),
                "KV_MIDDLEWARE_KV_ADMISSION_MARGIN_BLOCKS": os.environ.get("KV_MIDDLEWARE_KV_ADMISSION_MARGIN_BLOCKS", ""),
                "KV_MIDDLEWARE_P2_READY_RECLAIM_MARGIN_BLOCKS": os.environ.get("KV_MIDDLEWARE_P2_READY_RECLAIM_MARGIN_BLOCKS", ""),
                "KV_MIDDLEWARE_P2_MAX_READY_SEQUENCES_PER_STEP": os.environ.get("KV_MIDDLEWARE_P2_MAX_READY_SEQUENCES_PER_STEP", ""),
                "KV_MIDDLEWARE_P2_MAX_READY_OFFLOAD_BLOCKS_PER_STEP": os.environ.get("KV_MIDDLEWARE_P2_MAX_READY_OFFLOAD_BLOCKS_PER_STEP", ""),
            },
            "sample_summary": sample_summary,
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
