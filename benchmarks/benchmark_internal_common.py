import gc
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(CONFIGS_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIGS_DIR))

from engine import ManagedInferenceEngine
from strategy_groups import COMMON_BASE, GROUP_ARGS, HF_STYLE_METHODS
from benchmark_vllm_common import (
    OOM_KEYWORDS,
    classify_frontier_reason,
    classify_status_from_rates,
    percentile,
)


def parse_methods(text: str, allow_vllm: bool = True) -> List[str]:
    values: List[str] = []
    for chunk in str(text or "").split(","):
        method = str(chunk).strip()
        if not method:
            continue
        if method == "vllm":
            if not allow_vllm:
                raise ValueError("method 'vllm' is not allowed here")
            values.append(method)
            continue
        if method not in GROUP_ARGS and method not in HF_STYLE_METHODS:
            raise ValueError(f"unknown method: {method}")
        values.append(method)
    if not values:
        raise ValueError("empty methods list")
    return values


def parse_method_frac_map(text: str, defaults: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    if not str(text or "").strip():
        return dict(defaults or {})
    result: Dict[str, float] = {}
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        method, frac = chunk.split(":", 1)
        result[str(method).strip()] = float(frac.strip())
    return result


def chunked(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    width = max(1, int(size))
    for idx in range(0, len(items), width):
        yield items[idx : idx + width]


def is_hf_style_method(method: str) -> bool:
    return str(method or "").strip() in HF_STYLE_METHODS


def is_internal_method(method: str) -> bool:
    return str(method or "").strip() in GROUP_ARGS


def build_engine(model_name: str, method: str, gpu_mem_frac: float, max_new_tokens: int) -> ManagedInferenceEngine:
    if not is_internal_method(method):
        raise ValueError(f"build_engine only supports internal methods, got: {method}")
    args = dict(COMMON_BASE)
    args["model_name"] = str(model_name)
    args["gpu_mem_frac"] = float(gpu_mem_frac)
    args["max_new_tokens"] = int(max_new_tokens)
    args.update(GROUP_ARGS[str(method)])
    return ManagedInferenceEngine(**args)


def cleanup_engine(engine: Optional[ManagedInferenceEngine]) -> None:
    if engine is None:
        return
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


def compression_profile_for_method(method: str) -> str:
    if not is_internal_method(method):
        raise ValueError(f"compression_profile_for_method only supports internal methods, got: {method}")
    args = dict(COMMON_BASE)
    args.update(GROUP_ARGS[str(method)])
    sink = int(args.get("sink_len", 16))
    obs = int(args.get("snapkv_observation_len", args.get("obs_len", 16)))
    retain = float(args.get("retain_ratio", 1.0))
    budget = int(args.get("retain_budget_tokens", 0) or 0)
    mode = "fixed_budget" if budget > 0 else "ratio"
    p2_sink = int(args.get("p2_sink_tokens", 16))
    p2_recent = int(args.get("p2_recent_tokens", 16))
    return f"sink={sink};snapkv_obs={obs};retain={retain:.3f};budget={budget};mode={mode};p2_sink={p2_sink};p2_recent={p2_recent}"


def count_tokens(tokenizer, text: str) -> int:
    if not str(text or "").strip():
        return 0
    return int(len(tokenizer(str(text), add_special_tokens=False).input_ids))


def summarize_internal_request_rows(
    request_rows: Sequence[Dict[str, Any]],
    total_requested: int,
    requested_repeats: int,
    batch_wall_ms_list: Sequence[float],
    total_blocks: int,
) -> Dict[str, Any]:
    requested = max(1, int(total_requested))
    completed = sum(int(row.get("completed", 0)) for row in request_rows)
    valid = sum(int(row.get("valid", 0)) for row in request_rows)
    total_tokens = sum(int(row.get("completion_tokens", 0)) for row in request_rows)
    total_wall_ms = float(sum(float(x) for x in batch_wall_ms_list))
    ttfts = [float(row.get("ttft_ms", 0.0)) for row in request_rows if float(row.get("ttft_ms", 0.0)) > 0]
    avg_itls = [float(row.get("avg_itl_ms", 0.0)) for row in request_rows if float(row.get("avg_itl_ms", 0.0)) > 0]
    errors = [str(row.get("error_reason", "")).strip() for row in request_rows if str(row.get("error_reason", "")).strip()]
    oom_count = sum(
        1
        for err in errors
        if any(word in err.lower() for word in OOM_KEYWORDS) or "timeout" in err.lower()
    )
    completion_rate = float(completed / requested)
    valid_completion_rate = float(valid / requested)
    status = classify_status_from_rates(completion_rate, valid_completion_rate, oom_count, "; ".join(errors[:3]))
    min_free_candidates = [
        int(row.get("min_free_blocks", -1))
        for row in request_rows
        if int(row.get("min_free_blocks", -1)) >= 0
    ]
    min_free_blocks = min(min_free_candidates) if min_free_candidates else -1
    min_free_ratio = float(min_free_blocks / max(1, int(total_blocks))) if min_free_blocks >= 0 and total_blocks > 0 else -1.0
    return {
        "requested_repeats": int(requested_repeats),
        "requested_requests": int(total_requested),
        "completed_requests": int(completed),
        "completion_rate": completion_rate,
        "valid_completion_rate": valid_completion_rate,
        "oom_failure_count": int(oom_count),
        "status": status,
        "frontier_reason": classify_frontier_reason(status, "; ".join(errors[:3]), completion_rate, valid_completion_rate),
        "tokens_per_sec": float((1000.0 * total_tokens / total_wall_ms) if total_wall_ms > 0 else 0.0),
        "ttft_p95_ms": float(round(percentile(ttfts, 0.95), 3)),
        "ttft_p99_ms": float(round(percentile(ttfts, 0.99), 3)),
        "itl_p95_ms": float(round(percentile(avg_itls, 0.95), 3)),
        "itl_p99_ms": float(round(percentile(avg_itls, 0.99), 3)),
        "wall_clock_total_runtime_ms": float(round(total_wall_ms, 3)),
        "min_free_blocks": int(min_free_blocks),
        "min_free_block_ratio": float(round(min_free_ratio, 6)) if min_free_ratio >= 0 else -1.0,
        "error_reason": "; ".join(errors[:3]),
        "stack_type": "internal",
        "kv_backend": "managed_paged_pool",
        "fallback_policy": "internal_retry_split",
    }


def run_internal_prompt_batches(
    model_name: str,
    method: str,
    gpu_mem_frac: float,
    prompts: Sequence[str],
    items: Sequence[Dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
    tokenizer,
    repeats: int,
    evaluate_output_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
) -> Dict[str, Any]:
    engine: Optional[ManagedInferenceEngine] = None
    request_rows: List[Dict[str, Any]] = []
    batch_wall_ms_list: List[float] = []
    batch_metric_rows: List[Dict[str, Any]] = []
    step_metric_rows: List[Dict[str, Any]] = []
    total_blocks = 0
    total_requested = int(len(items) * max(1, int(repeats)))
    progress_enabled = str(os.environ.get("KV_MIDDLEWARE_BENCH_PROGRESS", "")).strip().lower() in {"1", "true", "yes", "on"}
    fail_fast_on_failure = str(os.environ.get("KV_MIDDLEWARE_BENCH_FAIL_FAST_ON_FAILURE", "")).strip().lower() in {"1", "true", "yes", "on"}
    batch_index = 0
    total_batches = max(1, ((len(items) + max(1, int(concurrency)) - 1) // max(1, int(concurrency))) * max(1, int(repeats)))
    fatal_error = ""

    try:
        engine = build_engine(model_name, method, gpu_mem_frac, max_new_tokens)
        for repeat_idx in range(int(max(1, repeats))):
            if fatal_error:
                break
            for batch_items in chunked(list(items), int(concurrency)):
                batch_index += 1
                batch_prompts = [str(item.get("prompt", "")) for item in batch_items]
                batch_input_ids = [
                    [int(token_id) for token_id in item["input_ids"]]
                    for item in batch_items
                ]
                batch_eos_token_ids = [
                    [int(token_id) for token_id in item.get("eos_token_ids", [])]
                    for item in batch_items
                ]
                step_rows: List[Dict[str, Any]] = []

                def _step_cb(stat: Dict[str, Any]) -> None:
                    step_rows.append(dict(stat))

                try:
                    t0 = time.perf_counter()
                    outputs, metrics = engine.generate(
                        batch_prompts,
                        prompt_token_ids=batch_input_ids,
                        eos_token_ids=batch_eos_token_ids,
                        return_metrics=True,
                        step_callback=_step_cb,
                    )
                    batch_errors = list(metrics.get("failed_request_errors", []) or [])
                    batch_metric_rows.append(dict(metrics))
                    step_metric_rows.extend(dict(s) for s in step_rows)
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    batch_wall_ms_list.append(float(wall_ms))
                    total_blocks = max(total_blocks, int(metrics.get("kv_total_blocks", 0)))
                    decode_only = [float(s.get("step_ms", 0.0)) for s in step_rows if int(s.get("decode_tokens", 0)) > 0]
                    ttft_proxy_ms = float(metrics.get("prefill_ms", 0.0)) + (float(decode_only[0]) if decode_only else 0.0)
                    avg_itl_proxy_ms = float(mean(decode_only[1:])) if len(decode_only) > 1 else 0.0
                    min_free_candidates = [
                        int(metrics.get("global_min_n_free", -1)),
                        int(metrics.get("decode_min_n_free", -1)),
                        int(metrics.get("prefill_min_n_free", -1)),
                    ]
                    min_free_candidates.extend(
                        int(s.get("global_min_n_free", -1)) for s in step_rows if int(s.get("global_min_n_free", -1)) >= 0
                    )
                    min_free_candidates.extend(
                        int(s.get("decode_min_n_free", -1)) for s in step_rows if int(s.get("decode_min_n_free", -1)) >= 0
                    )
                    min_free_blocks = min([x for x in min_free_candidates if x >= 0], default=-1)
                    batch_row_start = len(request_rows)
                    for row_idx, (item, output_text) in enumerate(zip(batch_items, outputs)):
                        text = str(output_text or "")
                        row_error = str(batch_errors[row_idx]).strip() if row_idx < len(batch_errors) and not text.strip() else ""
                        extra = dict(evaluate_output_fn(dict(item), text) or {})
                        valid = int(extra.pop("valid", int(bool(text.strip()))))
                        request_rows.append(
                            {
                                "repeat_idx": int(repeat_idx),
                                "completed": int(bool(text.strip())),
                                "valid": int(valid),
                                "completion_tokens": int(count_tokens(tokenizer, text)),
                                "ttft_ms": float(ttft_proxy_ms if text.strip() else 0.0),
                                "avg_itl_ms": float(avg_itl_proxy_ms if text.strip() else 0.0),
                                "wall_ms": float(round(wall_ms / max(1, len(batch_items)), 3)),
                                "min_free_blocks": int(min_free_blocks),
                                "error_reason": row_error,
                                "output_text": text,
                                **extra,
                            }
                        )
                    recent_rows = request_rows[batch_row_start:]
                    batch_completed = sum(int(row.get("completed", 0)) for row in recent_rows)
                    completed_so_far = sum(int(row.get("completed", 0)) for row in request_rows)
                    batch_errors_seen = [str(row.get("error_reason", "")).strip() for row in recent_rows if str(row.get("error_reason", "")).strip()]
                    batch_failed = (len(recent_rows) < len(batch_items)) or batch_completed < len(batch_items) or bool(batch_errors_seen)
                    if progress_enabled:
                        done_so_far = len(request_rows)
                        status_text = "failed" if batch_failed else "ok"
                        err_text = f" error={batch_errors_seen[0][:120]}" if batch_errors_seen else ""
                        print(
                            f"[internal-progress] method={method} batch={batch_index}/{total_batches} "
                            f"processed={done_so_far}/{total_requested} completed={completed_so_far}/{total_requested} "
                            f"batch_completed={batch_completed}/{len(batch_items)} wall_ms={wall_ms:.1f} status={status_text}{err_text}",
                            flush=True,
                        )
                    if fail_fast_on_failure and batch_failed:
                        fatal_error = batch_errors_seen[0] if batch_errors_seen else "fail_fast_first_batch_incomplete_or_empty_generation"
                        break
                except Exception as exc:
                    fatal_error = str(exc)
                    if progress_enabled:
                        print(f"[internal-progress] method={method} batch={batch_index}/{total_batches} status=exception error={fatal_error[:160]}", flush=True)
                    for item in batch_items:
                        extra = dict(evaluate_output_fn(dict(item), "") or {})
                        extra.pop("valid", None)
                        request_rows.append(
                            {
                                "repeat_idx": int(repeat_idx),
                                "completed": 0,
                                "valid": 0,
                                "completion_tokens": 0,
                                "ttft_ms": 0.0,
                                "avg_itl_ms": 0.0,
                                "wall_ms": 0.0,
                                "min_free_blocks": -1,
                                "error_reason": fatal_error,
                                "output_text": "",
                                **extra,
                            }
                        )
                    break
    finally:
        cleanup_engine(engine)

    summary = summarize_internal_request_rows(
        request_rows=request_rows,
        total_requested=total_requested,
        requested_repeats=int(repeats),
        batch_wall_ms_list=batch_wall_ms_list,
        total_blocks=int(total_blocks),
    )
    aggregate_rows = list(batch_metric_rows) + list(step_metric_rows)
    metric_min_keys = {
        "global_min_n_free",
        "decode_min_n_free",
        "prefill_min_n_free",
        "n_free",
        "n_free_final",
        "cuda_free_min_gb",
    }
    metric_keys = [
        "global_min_n_free",
        "decode_min_n_free",
        "prefill_min_n_free",
        "kv_total_blocks",
        "kv_peak_used_blocks",
        "decode_backpressure_events",
        "decode_memory_cap_events",
        "guard_seen_count",
        "guard_effective_shrink_count",
        "guard_strong_shrink_count",
        "guard_target_batch_min",
        "guard_source_batch_max",
        "decode_length_bucketed_steps",
        "decode_length_bucket_subbatch_count",
        "decode_length_bucket_singleton_count",
        "decode_length_bucket_max_trigger_ratio",
        "decode_memory_est_peak_max_gb",
        "decode_memory_est_peak_maxlen_gb",
        "decode_memory_est_peak_sumlen_gb",
        "prefill_backpressure_events",
        "prefill_batch_failed_steps",
        "prefill_chunk_failed_steps",
        "prefill_activate_failed_steps",
        "prefill_no_progress_steps",
        "prefill_no_progress_peak",
        "prefill_no_progress_fail_count",
        "decode_active_cap",
        "decode_active_cap_final",
        "decode_active_cap_boot",
        "decode_active_cap_min_seen",
        "ready_decode_resident_blocks",
        "decode_active_resident_blocks",
        "p2_active_steps",
        "p2_candidate_steps",
        "p2_no_candidate_steps",
        "p2_attempted_steps",
        "p2_success_steps",
        "p2_ready_candidate_steps",
        "p2_decode_candidate_steps",
        "p2_last_candidate_count",
        "p2_last_no_candidate",
        "p2_reject_deferred",
        "p2_reject_no_resident",
        "p2_reject_protected",
        "p2_ready_protected_ignored",
        "p2_reject_active_floor",
        "p2_reject_plan_empty",
        "p2_managed_active",
        "p2_expected_reclaim_blocks",
        "p2_ready_offload_blocks_total",
        "p2_ready_offload_blocks_last",
        "p2_ready_offload_sequence_steps",
        "p2_ready_offload_decode_steps",
        "p2_ready_sequences_selected_per_step",
        "p2_ready_offload_blocks_per_step",
        "p2_ready_target_reclaim_blocks",
        "p2_ready_actual_reclaim_blocks",
        "p2_ready_stop_target_reached_steps",
        "p2_ready_stop_sequence_cap_reached_steps",
        "p2_ready_stop_block_cap_reached_steps",
        "p2_ready_stop_low_benefit_skip_steps",
        "p2_ready_stop_not_needed_steps",
        "p2_ready_stop_no_ready_candidate_steps",
        "kv_admission_enabled",
        "kv_admission_blocked_steps",
        "kv_admission_blocked_requests",
        "kv_admission_last_free_blocks",
        "kv_admission_last_required_blocks",
        "kv_admission_last_prompt_blocks",
        "kv_admission_last_prompt_resident_blocks",
        "kv_admission_last_request_blocks",
        "kv_admission_last_pending_output_blocks",
        "kv_admission_last_allowed",
        "p2_gain_success_steps",
        "p2_gain_fail_steps",
        "p2_skipped_low_benefit_steps",
        "first_p2_step",
        "p2_low_threshold",
        "wm_low",
        "wm_high",
        "n_free",
        "n_free_final",
        "cuda_free_min_gb",
        "cuda_alloc_peak_gb",
        "cuda_reserved_peak_gb",
        "decode_paged_flash_strict",
        "retain_ratio",
        "retain_budget_tokens",
        "effective_retained_tokens",
        "retained_block_count",
        "decode_paged_direct_steps",
        "decode_page16_native_strict",
        "decode_page16_native_steps",
        "decode_rebuild_steps",
        "decode_materialize_kv_bytes",
        "decode_paged_direct_resident_miss_steps",
        "decode_page16_native_resident_miss_steps",
        "decode_page16_native_kernel_ms",
        "kv_logical_block_size",
        "flash_attn_enabled",
        "selected_writeback_enabled",
        "gpu_selected_writeback_steps",
        "cpu_selected_compaction_steps",
        "gpu_writeback_oom_fallbacks",
        "writeback_transaction_rollbacks",
        "raw_kv_cpu_stash_bytes",
        "selected_global_block_count",
        "writeback_est_required_gb",
        "writeback_free_gb",
        "writeback_block_selection_shared_layers",
        "score_full_attention_materialized",
    ]
    for key in metric_keys:
        vals = []
        for row in aggregate_rows:
            if key not in row:
                continue
            val = row.get(key)
            if isinstance(val, bool):
                vals.append(int(val))
            elif isinstance(val, (int, float)):
                vals.append(val)
        if not vals:
            continue
        if key in metric_min_keys:
            summary[key] = min(vals)
        else:
            summary[key] = max(vals)
    for key in ("decode_backend", "compression_mode", "decode_paged_direct_blocked_reason", "decode_page16_native_blocked_reason", "p2_ready_stop_reason", "prefill_writeback_backend"):
        for row in reversed(aggregate_rows):
            val = str(row.get(key, "")).strip()
            if val:
                summary[key] = val
                break
    reason_counts: Dict[str, int] = {}
    for row in aggregate_rows:
        reason = str(row.get("p2_ready_stop_reason", "")).strip()
        if reason and reason != "none":
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
    if reason_counts:
        summary["p2_ready_stop_reason_counts"] = reason_counts
    summary["compression_profile"] = compression_profile_for_method(method)
    summary["gpu_mem_frac"] = float(gpu_mem_frac)
    summary["rows"] = request_rows
    return summary
