import gc
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_vllm_common import (
    OOM_KEYWORDS,
    classify_frontier_reason,
    classify_status_from_rates,
    percentile,
)


HF_STYLE_METHODS = ("hf_vanilla", "off_raw")
STRICT_OFF_RAW_UNSUPPORTED_REASON = (
    "strict_block_only_direct_decode_not_implemented_in_current_stack; "
    "current paged_direct path still materializes contiguous KV via index_select+reshape"
)
HF_STYLE_CONTINUOUS_UNSUPPORTED_REASON = (
    "hf_style_continuous_runner_not_implemented; current continuous serving harness remains internal-only"
)
HF_STYLE_INTERNAL_COUNTER_KEYS = [
    "decode_backpressure_events",
    "decode_memory_cap_events",
    "decode_retry_timeout_fail_count",
    "decode_retry_cooldown_enabled",
    "decode_retry_cooldown_events",
    "decode_retry_cooldown_bypass_events",
    "decode_retry_cooldown_active",
    "guard_seen_count",
    "guard_effective_shrink_count",
    "guard_strong_shrink_count",
    "guard_target_batch_min",
    "guard_source_batch_max",
    "decode_length_bucketed_steps",
    "decode_length_bucket_subbatch_count",
    "decode_length_bucket_singleton_count",
    "decode_length_bucket_max_trigger_ratio",
    "decode_active_cap",
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
    "p2_gain_success_steps",
    "p2_gain_fail_steps",
    "p2_skipped_low_benefit_steps",
    "first_p2_step",
    "p2_low_threshold",
    "wm_low",
    "wm_high",
]


def is_hf_style_method(method: str) -> bool:
    return str(method or "").strip() in HF_STYLE_METHODS


def strict_off_raw_support() -> Dict[str, Any]:
    return {
        "supported": False,
        "reason": STRICT_OFF_RAW_UNSUPPORTED_REASON,
    }


def hf_style_stack_meta(method: str) -> Dict[str, str]:
    name = str(method or "").strip()
    if name == "hf_vanilla":
        return {
            "stack_type": "hf_style",
            "kv_backend": "contiguous",
            "fallback_policy": "none",
            "compression_profile": "hf_vanilla",
        }
    if name == "off_raw":
        return {
            "stack_type": "hf_style",
            "kv_backend": "block_only",
            "fallback_policy": "none",
            "compression_profile": "off_raw_block_only",
        }
    raise ValueError(f"unknown hf-style method: {method}")


def _hf_style_zero_internal_counters() -> Dict[str, int]:
    return {key: 0 for key in HF_STYLE_INTERNAL_COUNTER_KEYS}


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(str(model_name), trust_remote_code=True)
    return _ensure_padding_token(tokenizer)


def _ensure_padding_token(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def _load_model(model_name: str, tokenizer):
    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = None
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        load_kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            **load_kwargs,
        )
    except Exception:
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_name),
            **load_kwargs,
        )
    if getattr(model, "config", None) is not None and len(tokenizer) > int(getattr(model.config, "vocab_size", len(tokenizer))):
        model.resize_token_embeddings(len(tokenizer))
    model.eval()
    return model


def _model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _chunked(items: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    width = max(1, int(size))
    for idx in range(0, len(items), width):
        yield items[idx : idx + width]


def _count_tokens(tokenizer, text: str) -> int:
    if not str(text or "").strip():
        return 0
    return int(len(tokenizer(str(text), add_special_tokens=False).input_ids))


def _encode_manifest_batch(items, pad_token_id: int, device: torch.device):
    rows = [
        torch.as_tensor(item["input_ids"], dtype=torch.long)
        for item in items
    ]
    if not rows or any(row.numel() <= 0 for row in rows):
        raise ValueError("each manifest item must contain non-empty input_ids")
    width = max(int(row.numel()) for row in rows)
    input_ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        length = int(row.numel())
        input_ids[index, width - length :] = row
        attention_mask[index, width - length :] = 1
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def _batch_eos_token_ids(items):
    values = [
        tuple(int(token_id) for token_id in item.get("eos_token_ids", []))
        for item in items
    ]
    if not values or not values[0]:
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError("all requests in an HF batch must use the same EOS set")
    return list(values[0])


def _build_failure_rows(
    items: Sequence[Dict[str, Any]],
    repeats: int,
    evaluate_output_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
    error_reason: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for repeat_idx in range(int(max(1, repeats))):
        for item in items:
            extra = dict(evaluate_output_fn(dict(item), "") or {})
            extra.pop("valid", None)
            rows.append(
                {
                    "repeat_idx": int(repeat_idx),
                    "completed": 0,
                    "valid": 0,
                    "completion_tokens": 0,
                    "request_latency_ms": 0.0,
                    "wall_ms": 0.0,
                    "ttft_ms": 0.0,
                    "avg_itl_ms": 0.0,
                    "min_free_blocks": -1,
                    "error_reason": str(error_reason),
                    "output_text": "",
                    **extra,
                }
            )
    return rows


def make_hf_style_unsupported_result(
    method: str,
    serving_mode: str,
    total_requested: int,
    concurrency: int,
    max_new_tokens: int,
    reason: str,
    rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    meta = hf_style_stack_meta(method)
    result: Dict[str, Any] = {
        "method": str(method),
        "serving_mode": str(serving_mode),
        "concurrency": int(concurrency),
        "max_new_tokens": int(max_new_tokens),
        "requested_requests": int(total_requested),
        "completed_requests": 0,
        "completion_rate": 0.0,
        "valid_completion_rate": 0.0,
        "oom_failure_count": 0,
        "status": "Unsupported",
        "frontier_reason": "unsupported",
        "generated_tokens": 0,
        "tokens_per_sec": 0.0,
        "request_latency_p95_ms": 0.0,
        "request_latency_p99_ms": 0.0,
        "ttft_p95_ms": 0.0,
        "ttft_p99_ms": 0.0,
        "itl_p95_ms": 0.0,
        "itl_p99_ms": 0.0,
        "wall_clock_total_runtime_ms": 0.0,
        "min_free_blocks": -1,
        "min_free_block_ratio": -1.0,
        "kv_total_blocks": 0,
        "error_reason": str(reason),
        "unsupported_reason": str(reason),
        "cuda_alloc_peak_gb": 0.0,
        "cuda_reserved_peak_gb": 0.0,
        "timing_note": "HF-style path unsupported on current stack; no model execution was attempted.",
        "rows": list(rows or []),
        **meta,
    }
    result.update(_hf_style_zero_internal_counters())
    return result


def _summarize_hf_rows(
    method: str,
    serving_mode: str,
    rows: Sequence[Dict[str, Any]],
    total_requested: int,
    total_wall_ms: float,
    concurrency: int,
    max_new_tokens: int,
    max_alloc_gb: float,
    max_reserved_gb: float,
) -> Dict[str, Any]:
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
    itls = [float(r.get("avg_itl_ms", 0.0)) for r in rows if float(r.get("avg_itl_ms", 0.0)) > 0]
    meta = hf_style_stack_meta(method)
    summary: Dict[str, Any] = {
        "method": str(method),
        "serving_mode": str(serving_mode),
        "concurrency": int(concurrency),
        "max_new_tokens": int(max_new_tokens),
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
        "ttft_p95_ms": 0.0,
        "ttft_p99_ms": 0.0,
        "itl_p95_ms": float(round(percentile(itls, 0.95), 3)),
        "itl_p99_ms": float(round(percentile(itls, 0.99), 3)),
        "min_free_blocks": -1,
        "min_free_block_ratio": -1.0,
        "kv_total_blocks": 0,
        "error_reason": "; ".join(errors[:3]),
        "cuda_alloc_peak_gb": float(round(max_alloc_gb, 3)),
        "cuda_reserved_peak_gb": float(round(max_reserved_gb, 3)),
        "timing_note": "HF-style batch runner does not expose TTFT; avg_itl is wall_ms/completion_tokens proxy.",
        "rows": list(rows),
        **meta,
    }
    summary.update(_hf_style_zero_internal_counters())
    return summary


def run_hf_style_prompt_batches(
    model_name: str,
    method: str,
    items: Sequence[Dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
    repeats: int,
    evaluate_output_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
    tokenizer=None,
) -> Dict[str, Any]:
    method = str(method)
    total_requested = int(len(items) * max(1, repeats))
    if method == "off_raw":
        return make_hf_style_unsupported_result(
            method=method,
            serving_mode="chunked_batch",
            total_requested=total_requested,
            concurrency=int(concurrency),
            max_new_tokens=int(max_new_tokens),
            reason=STRICT_OFF_RAW_UNSUPPORTED_REASON,
            rows=_build_failure_rows(items, repeats, evaluate_output_fn, STRICT_OFF_RAW_UNSUPPORTED_REASON),
        )
    if method != "hf_vanilla":
        raise ValueError(f"unsupported hf-style batch method: {method}")

    tokenizer = _ensure_padding_token(tokenizer or _load_tokenizer(model_name))
    model = None
    rows: List[Dict[str, Any]] = []
    total_wall_ms = 0.0
    fatal_error = ""
    max_alloc_gb = 0.0
    max_reserved_gb = 0.0
    try:
        model = _load_model(model_name, tokenizer)
        input_device = _model_input_device(model)
        for repeat_idx in range(max(1, int(repeats))):
            if fatal_error:
                break
            for batch in _chunked(list(items), int(concurrency)):
                try:
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    encoded = _encode_manifest_batch(
                        batch,
                        int(tokenizer.pad_token_id),
                        input_device,
                    )
                    prompt_width = int(encoded["input_ids"].shape[1])
                    eos_token_ids = _batch_eos_token_ids(batch)
                    with torch.inference_mode():
                        outputs = model.generate(
                            **encoded,
                            max_new_tokens=int(max_new_tokens),
                            min_new_tokens=1,
                            num_beams=1,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=int(tokenizer.pad_token_id),
                            eos_token_id=eos_token_ids,
                        )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    wall_ms = (time.perf_counter() - started) * 1000.0
                    total_wall_ms += wall_ms
                    if torch.cuda.is_available():
                        max_alloc_gb = max(max_alloc_gb, float(torch.cuda.max_memory_allocated() / (1024 ** 3)))
                        max_reserved_gb = max(max_reserved_gb, float(torch.cuda.max_memory_reserved() / (1024 ** 3)))
                    generated = outputs[:, prompt_width:]
                    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
                    for item, text in zip(batch, texts):
                        text = str(text or "")
                        extra = dict(evaluate_output_fn(dict(item), text) or {})
                        valid = int(extra.pop("valid", int(bool(text.strip()))))
                        comp_tokens = int(_count_tokens(tokenizer, text))
                        rows.append(
                            {
                                "repeat_idx": int(repeat_idx),
                                "completed": int(bool(text.strip())),
                                "valid": int(valid if text.strip() else 0),
                                "completion_tokens": int(comp_tokens),
                                "request_latency_ms": float(round(wall_ms, 3)),
                                "wall_ms": float(round(wall_ms, 3)),
                                "ttft_ms": 0.0,
                                "avg_itl_ms": float(round((wall_ms / max(1, comp_tokens)) if comp_tokens > 0 else 0.0, 3)),
                                "min_free_blocks": -1,
                                "error_reason": "",
                                "output_text": text,
                                **extra,
                            }
                        )
                    del encoded, outputs, generated
                except Exception as exc:
                    fatal_error = str(exc)
                    _cleanup_cuda()
                    for item in batch:
                        extra = dict(evaluate_output_fn(dict(item), "") or {})
                        extra.pop("valid", None)
                        rows.append(
                            {
                                "repeat_idx": int(repeat_idx),
                                "completed": 0,
                                "valid": 0,
                                "completion_tokens": 0,
                                "request_latency_ms": 0.0,
                                "wall_ms": 0.0,
                                "ttft_ms": 0.0,
                                "avg_itl_ms": 0.0,
                                "min_free_blocks": -1,
                                "error_reason": fatal_error,
                                "output_text": "",
                                **extra,
                            }
                        )
                    break
    finally:
        del model
        _cleanup_cuda()

    if fatal_error and len(rows) < total_requested:
        for _ in range(total_requested - len(rows)):
            rows.append(
                {
                    "completed": 0,
                    "valid": 0,
                    "completion_tokens": 0,
                    "request_latency_ms": 0.0,
                    "wall_ms": 0.0,
                    "ttft_ms": 0.0,
                    "avg_itl_ms": 0.0,
                    "min_free_blocks": -1,
                    "error_reason": fatal_error or "not_executed_after_failure",
                    "output_text": "",
                }
            )
    return _summarize_hf_rows(
        method=method,
        serving_mode="chunked_batch",
        rows=rows,
        total_requested=total_requested,
        total_wall_ms=total_wall_ms if total_wall_ms > 0 else 1.0,
        concurrency=int(concurrency),
        max_new_tokens=int(max_new_tokens),
        max_alloc_gb=max_alloc_gb,
        max_reserved_gb=max_reserved_gb,
    )


def run_hf_style_continuous(
    model_name: str,
    method: str,
    items: Sequence[Dict[str, Any]],
    concurrency: int,
    max_new_tokens: int,
    repeats: int,
    evaluate_output_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
    use_target_tokens: bool = False,
) -> Dict[str, Any]:
    method = str(method)
    total_requested = int(len(items) * max(1, repeats))
    if method == "off_raw":
        reason = STRICT_OFF_RAW_UNSUPPORTED_REASON
        return make_hf_style_unsupported_result(
            method=method,
            serving_mode="continuous_refill",
            total_requested=total_requested,
            concurrency=int(concurrency),
            max_new_tokens=int(max_new_tokens),
            reason=reason,
            rows=_build_failure_rows(items, repeats, evaluate_output_fn, reason),
        )
    if method != "hf_vanilla":
        raise ValueError(f"unsupported hf-style continuous method: {method}")

    tokenizer = _load_tokenizer(model_name)
    model = None
    rows: List[Dict[str, Any]] = []
    total_wall_ms = 0.0
    fatal_error = ""
    max_alloc_gb = 0.0
    max_reserved_gb = 0.0

    def request_limit(item: Dict[str, Any]) -> int:
        if bool(use_target_tokens):
            try:
                target = int(item.get("target_tokens", 0) or 0)
            except Exception:
                target = 0
            if target > 0:
                return max(1, target)
        return max(1, int(max_new_tokens))

    max_request_new_tokens = max((request_limit(dict(x)) for x in items), default=max(1, int(max_new_tokens)))

    try:
        model = _load_model(model_name, tokenizer)
        input_device = _model_input_device(model)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        def make_failed_row(repeat_idx: int, item: Dict[str, Any], error: str) -> Dict[str, Any]:
            extra = dict(evaluate_output_fn(dict(item), "") or {})
            extra.pop("valid", None)
            return {
                "repeat_idx": int(repeat_idx),
                "record_idx": int(item.get("record_idx", -1)),
                "turn_idx": int(item.get("turn_idx", -1)),
                "prompt_tokens": int(item.get("prompt_tokens", 0) or 0),
                "target_tokens": int(item.get("target_tokens", 0) or 0),
                "request_max_new_tokens": int(request_limit(item)),
                "completed": 0,
                "valid": 0,
                "completion_tokens": 0,
                "request_latency_ms": 0.0,
                "wall_ms": 0.0,
                "ttft_ms": 0.0,
                "avg_itl_ms": 0.0,
                "min_free_blocks": -1,
                "error_reason": str(error),
                "output_text": "",
                **extra,
            }

        def run_one(repeat_idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
            req_max_new_tokens = request_limit(item)
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                encoded = tokenizer(str(item["prompt"]), return_tensors="pt", truncation=False)
                encoded = {k: v.to(input_device) for k, v in encoded.items()}
                prompt_width = int(encoded["input_ids"].shape[1])
                with torch.inference_mode():
                    outputs = model.generate(
                        **encoded,
                        max_new_tokens=int(req_max_new_tokens),
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=int(tokenizer.pad_token_id),
                        eos_token_id=int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - started) * 1000.0
                generated = outputs[:, prompt_width:]
                text = str(tokenizer.decode(generated[0], skip_special_tokens=True) or "")
                comp_tokens = int(_count_tokens(tokenizer, text))
                extra = dict(evaluate_output_fn(dict(item), text) or {})
                valid = int(extra.pop("valid", int(bool(text.strip()))))
                del encoded, outputs, generated
                return {
                    "repeat_idx": int(repeat_idx),
                    "record_idx": int(item.get("record_idx", -1)),
                    "turn_idx": int(item.get("turn_idx", -1)),
                    "prompt_tokens": int(item.get("prompt_tokens", 0) or 0),
                    "target_tokens": int(item.get("target_tokens", 0) or 0),
                    "request_max_new_tokens": int(req_max_new_tokens),
                    "completed": int(bool(text.strip())),
                    "valid": int(valid if text.strip() else 0),
                    "completion_tokens": int(comp_tokens),
                    "request_latency_ms": float(round(wall_ms, 3)),
                    "wall_ms": float(round(wall_ms, 3)),
                    "ttft_ms": 0.0,
                    "avg_itl_ms": float(round((wall_ms / max(1, comp_tokens)) if comp_tokens > 0 else 0.0, 3)),
                    "min_free_blocks": -1,
                    "error_reason": "",
                    "output_text": text,
                    **extra,
                }
            except Exception as exc:
                _cleanup_cuda()
                return make_failed_row(repeat_idx, item, str(exc))

        items_list = [dict(x) for x in items]
        for repeat_idx in range(max(1, int(repeats))):
            if fatal_error:
                break
            submitted = 0
            pending: Dict[Any, int] = {}
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
                while submitted < len(items_list) and len(pending) < int(concurrency):
                    fut = pool.submit(run_one, int(repeat_idx), dict(items_list[submitted]))
                    pending[fut] = submitted
                    submitted += 1
                while pending:
                    done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                    for fut in done:
                        idx = pending.pop(fut)
                        try:
                            row = dict(fut.result())
                        except Exception as exc:
                            row = make_failed_row(int(repeat_idx), dict(items_list[idx]), str(exc))
                        rows.append(row)
                        err = str(row.get("error_reason", "")).strip()
                        if err:
                            fatal_error = err
                    if fatal_error:
                        for fut in list(pending.keys()):
                            fut.cancel()
                        break
                    while submitted < len(items_list) and len(pending) < int(concurrency):
                        fut = pool.submit(run_one, int(repeat_idx), dict(items_list[submitted]))
                        pending[fut] = submitted
                        submitted += 1
            total_wall_ms += (time.perf_counter() - t0) * 1000.0
            if fatal_error:
                for idx in range(submitted, len(items_list)):
                    rows.append(make_failed_row(int(repeat_idx), dict(items_list[idx]), fatal_error or "not_executed_after_failure"))
                break
        if torch.cuda.is_available():
            max_alloc_gb = max(max_alloc_gb, float(torch.cuda.max_memory_allocated() / (1024 ** 3)))
            max_reserved_gb = max(max_reserved_gb, float(torch.cuda.max_memory_reserved() / (1024 ** 3)))
    finally:
        del model
        _cleanup_cuda()

    if fatal_error and len(rows) < total_requested:
        blank = dict(items[0]) if items else {}
        for _ in range(total_requested - len(rows)):
            rows.append(make_failed_row(0, blank, fatal_error or "not_executed_after_failure"))

    summary = _summarize_hf_rows(
        method=method,
        serving_mode="continuous_refill",
        rows=rows,
        total_requested=total_requested,
        total_wall_ms=total_wall_ms if total_wall_ms > 0 else 1.0,
        concurrency=int(concurrency),
        max_new_tokens=int(max_new_tokens),
        max_alloc_gb=max_alloc_gb,
        max_reserved_gb=max_reserved_gb,
    )
    summary["generation_length_policy"] = "target_tokens" if bool(use_target_tokens) else "fixed_max_new_tokens"
    summary["max_request_new_tokens"] = int(max_request_new_tokens)
    summary["timing_note"] = (
        "HF continuous runner uses native Transformers generate() per request with request-level refill; "
        "it does not implement a token-level serving scheduler. TTFT is not exposed."
    )
    return summary
