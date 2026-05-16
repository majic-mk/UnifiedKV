
import argparse
import gc
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ALLOC_CONF_ENABLED = (
    str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")).strip() == "expandable_segments:True"
)

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from engine import ManagedInferenceEngine

LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
PROMPT_PROFILE = "online_mixed_synthetic_v1"
METRIC_PROFILE = "p2_online_mixed_flow"
GROUP_ORDER = ["off_compress", "main_auto_compress"]
OOM_KEYWORDS = (
    "out of memory",
    "cuda out of memory",
    "cannot evict enough blocks",
    "cublas_status_alloc_failed",
    "cuda error",
    "no free blocks",
    "allocation failed",
)

COMMON_BASE = {
    "model_name": LOCAL_MODEL_PATH,
    "cpu_mem_gb": 32.0,
    "chunk_size": 1024,
    "max_new_tokens": 512,
    "prefill_batch_size": 8,
    "decode_micro_batch_size": 16,
    "decode_active_cap_initial": 16,
    "max_decode_active_cap": 16,
    "sink_len": 16,
    "obs_len": 16,
    "decode_window_sink_len": 16,
    "decode_path_mode": "rebuild",
    "decode_paged_flash_enabled": False,
    "max_waiting_requests": 512,
    "max_prefill_active": 16,
    "prefill_token_budget_per_step": 16384,
}

OFF_COMPRESS_ARGS = {
    "retain_ratio": 0.10,
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": False,
}

MAIN_AUTO_COMPRESS_ARGS = {
    "retain_ratio": 0.10,
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": True,
    "decode_window_tiered": True,
    "decode_window_recent_len": 256,
    "decode_window_check_interval": 8,
    "decode_window_min_trigger_tokens": 64,
    "decode_window_min_drop_tokens": 64,
    "decode_window_cooldown_steps": 16,
    "decode_window_aggressive_recent_len": 128,
    "decode_window_aggressive_min_trigger_tokens": 32,
    "decode_window_aggressive_min_drop_tokens": 64,
    "decode_window_aggressive_cooldown_steps": 8,
    "decode_window_thrash_window_steps": 16,
    "decode_window_thrash_low": 0.30,
    "decode_window_thrash_high": 0.80,
    "decode_window_emergency_windows": 2,
    "decode_window_min_compress_guard_tokens": 64,
    "decode_window_anchor_tokens": 16,
    "decode_window_recent_score_weight": 0.60,
    "decode_window_anchor_decay_threshold": 128,
    "decode_window_anchor_decay_span": 384,
    "decode_window_anchor_min_weight": 0.10,
    "decode_window_conservative_keep_ratio": 0.50,
    "decode_window_conservative_delete_cap_ratio": 0.50,
    "decode_window_conservative_min_keep_ratio": 0.50,
    "decode_window_aggressive_keep_ratio": 0.35,
    "decode_window_aggressive_delete_cap_ratio": 0.65,
    "decode_window_aggressive_min_keep_ratio": 0.35,
    "decode_window_pressure_steps": 2,
    "decode_window_recover_steps": 8,
    "decode_window_pressure_margin_blocks": 96,
    "decode_window_emergency_check_interval": 4,
    "decode_window_emergency_margin_blocks": 48,
}

GROUP_ARGS = {
    "off_compress": OFF_COMPRESS_ARGS,
    "main_auto_compress": MAIN_AUTO_COMPRESS_ARGS,
}


def parse_length_spec(spec: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":", 1)
        out.append((int(k.strip()), int(v.strip())))
    if not out:
        raise ValueError("empty length spec")
    return out


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(x) for x in values)
    if not vals:
        return 0.0
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def align_prompt_tolerance(target_tokens: int) -> int:
    return max(1, min(int(round(target_tokens * 0.02)), 256))


def is_oom_text(text: str) -> bool:
    t = str(text).lower()
    return any(k in t for k in OOM_KEYWORDS)


def cleanup_engine(engine: Optional[ManagedInferenceEngine], reason: str = "") -> None:
    if engine is None:
        return
    try:
        if hasattr(engine, "has_pending_requests") and engine.has_pending_requests():
            req_map = dict(getattr(engine, "_requests", {}) or {})
            for req in req_map.values():
                try:
                    engine._mark_request_failed(req, RuntimeError(reason or "p2_cleanup"))
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


def build_engine(args: argparse.Namespace, group: str) -> ManagedInferenceEngine:
    kwargs = dict(COMMON_BASE)
    kwargs.update({
        "model_name": args.model_name,
        "gpu_mem_frac": float(args.gpu_mem_frac),
        "max_new_tokens": int(args.max_new_tokens),
        "decode_micro_batch_size": int(args.decode_micro_batch_size),
        "decode_active_cap_initial": int(args.decode_active_cap_initial),
        "max_decode_active_cap": int(args.max_decode_active_cap),
        "prefill_batch_size": int(args.prefill_batch_size),
        "max_prefill_active": int(args.max_prefill_active),
        "prefill_token_budget_per_step": int(args.prefill_token_budget_per_step),
        "cpu_mem_gb": float(args.cpu_mem_gb),
    })
    kwargs.update(GROUP_ARGS[group])
    return ManagedInferenceEngine(**kwargs)


def compute_memory_stats(engine: ManagedInferenceEngine, gpu_mem_frac_effective: float) -> Tuple[float, float]:
    total = float(torch.cuda.get_device_properties(0).total_memory)
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
    obs = int(args.get("obs_len", 16))
    retain = float(args.get("retain_ratio", 1.0))
    decode_sink = int(args.get("decode_window_sink_len", 16))
    decode_recent = int(args.get("decode_window_recent_len", 0))
    return f"sink={sink};obs={obs};retain={retain:.3f};decode_sink={decode_sink};decode_recent={decode_recent}"


def build_prompt_for_target(tokenizer, target_tokens: int, seq_tag: str) -> Tuple[str, int]:
    tol = align_prompt_tolerance(target_tokens)
    user_text = (
        f"[{seq_tag}] You are analyzing an online long-context inference service. "
        "Explain scheduler fairness, decode batching, asynchronous CPU offload, KV prefetch, pressure-aware pruning, "
        "and latency-throughput tradeoffs using coherent technical prose."
    )
    filler = (
        " Context fragment discusses cache locality, active-cap throttling, pinned-memory copies, "
        "prefill/decode overlap, cold-sequence eviction, retryable decode append failure, and watermark hysteresis."
    )
    prompt = user_text
    actual = 0
    for _ in range(14):
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
    return prompt, int(actual)


def build_prompt_from_seed(tokenizer, target_tokens: int, seq_tag: str, seed_text: str) -> Tuple[str, int]:
    tol = align_prompt_tolerance(target_tokens)
    base_seed = str(seed_text or "").strip()
    if not base_seed:
        return build_prompt_for_target(tokenizer, target_tokens, seq_tag)
    user_text = f"[{seq_tag}] Read the following real benchmark sample and answer according to the task.\n\n{base_seed}"
    filler = (
        "\n\nAdditional context discusses memory pressure, prefill/decode overlap, asynchronous offload, retryable append failure, and scheduler fairness."
    )
    prompt = user_text
    actual = 0
    for _ in range(16):
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
            user_text += filler * max(1, diff // 32)
        else:
            trim = max(64, min(len(user_text) // 10, abs(diff) * 3))
            user_text = user_text[:-trim] if trim < len(user_text) else user_text
    return prompt, int(actual)


def _extract_record_prompt(record: Dict[str, Any]) -> str:
    if isinstance(record.get("prompt"), str) and record.get("prompt", "").strip():
        return str(record["prompt"])
    context = str(record.get("context", "") or record.get("document", "") or record.get("text", "")).strip()
    query = str(record.get("input", "") or record.get("question", "") or record.get("query", "")).strip()
    instruction = str(record.get("instruction", "") or record.get("task", "")).strip()
    parts = [x for x in [instruction, context, query] if x]
    if parts:
        return "\n\n".join(parts)
    return json.dumps(record, ensure_ascii=False)


def _extract_answers(record: Dict[str, Any]) -> List[str]:
    raw = record.get("answers", record.get("answer", record.get("target", [])))
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if raw is None:
        return []
    return [str(raw)]


def _iter_json_records(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
        return
    obj = json.loads(text)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        for key in ["data", "records", "examples"]:
            val = obj.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        yield item
                return
        yield obj


def build_prompt_records(tokenizer, length_spec: List[Tuple[int, int]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    source = str(args.content_source).strip().lower()
    if source == "synthetic":
        buckets: Dict[int, Deque[Dict[str, Any]]] = {}
        for target_len, count in length_spec:
            q: Deque[Dict[str, Any]] = deque()
            for idx in range(int(count)):
                prompt, actual = build_prompt_for_target(tokenizer, int(target_len), f"len={target_len};idx={idx}")
                q.append({
                    "target_len": int(target_len),
                    "actual_prompt_tokens": int(actual),
                    "prompt": prompt,
                    "answers": [],
                    "source": "synthetic",
                })
            buckets[int(target_len)] = q
        ordered: List[Dict[str, Any]] = []
        ordered_lengths = sorted(buckets.keys(), reverse=True)
        while any(buckets[k] for k in ordered_lengths):
            for k in ordered_lengths:
                if buckets[k]:
                    ordered.append(buckets[k].popleft())
        for i, rec in enumerate(ordered):
            rec["request_index"] = int(i)
        return ordered

    def load_records_from_file(path_str: str, label: str) -> List[Dict[str, Any]]:
        if not path_str:
            return []
        p = Path(path_str)
        if not p.exists():
            raise FileNotFoundError(f"missing dataset file for {label}: {p}")
        out: List[Dict[str, Any]] = []
        for item in _iter_json_records(p):
            out.append({
                "prompt": _extract_record_prompt(item),
                "answers": _extract_answers(item),
                "source": label,
            })
        return out

    longbench_records = load_records_from_file(args.longbench_path, "longbench") if source in ("longbench", "mixed") else []
    ruler_records = load_records_from_file(args.ruler_path, "ruler") if source in ("ruler", "mixed") else []
    if source == "longbench":
        pool = longbench_records
    elif source == "ruler":
        pool = ruler_records
    elif source == "mixed":
        pool = []
        li = ri = 0
        while li < len(longbench_records) or ri < len(ruler_records):
            if li < len(longbench_records):
                pool.append(longbench_records[li]); li += 1
            if ri < len(ruler_records):
                pool.append(ruler_records[ri]); ri += 1
    else:
        raise ValueError(f"unsupported content source: {source}")
    if not pool:
        raise ValueError(f"no dataset records for content source: {source}")

    ordered: List[Dict[str, Any]] = []
    total_needed = sum(int(c) for _, c in length_spec)
    for idx in range(total_needed):
        target_len = int(length_spec[idx % len(length_spec)][0])
        record = dict(pool[idx % len(pool)])
        prompt, actual = build_prompt_from_seed(tokenizer, target_len, f"src={record['source']};idx={idx}", record.get("prompt", ""))
        record.update({
            "target_len": target_len,
            "actual_prompt_tokens": int(actual),
            "prompt": prompt,
            "request_index": int(idx),
        })
        ordered.append(record)
    return ordered


def maybe_apply_frontier_hint(args: argparse.Namespace) -> None:
    if args.gpu_mem_frac > 0 or not args.frontier_json:
        return
    payload = json.loads(Path(args.frontier_json).read_text(encoding="utf-8"))
    rows = list(payload.get("frontier_rows", []))
    wanted_lengths = [int(k) for k, _ in parse_length_spec(args.length_spec)]
    wanted_group = str(args.frontier_group).strip() or "main_auto_compress"
    matched = [r for r in rows if str(r.get("group")) == wanted_group and int(r.get("input_len", 0)) in wanted_lengths and float(r.get("gpu_mem_frac_best", 0.0)) > 0.0]
    if matched:
        args.gpu_mem_frac = min(float(r.get("gpu_mem_frac_best", 0.0)) for r in matched)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def summarize_engine_run(group: str, args: argparse.Namespace, step_rows: List[Dict[str, Any]], finished: List[Dict[str, Any]], model_memory_gb: float, kv_available_gb: float, wall_ms: float, prompt_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    decode_step_ms = [float(r.get("step_ms", 0.0)) for r in step_rows if int(r.get("decode_tokens", 0)) > 0]
    total_generated_tokens = sum(len(list(x.get("token_ids") or [])) for x in finished)
    success_count = sum(1 for x in finished if str(x.get("state")) == "DONE")
    fail_count = len(finished) - success_count
    oom_count = sum(1 for x in finished if is_oom_text(x.get("error", "")))
    last = step_rows[-1] if step_rows else {}
    off_cum = dict(last.get("offloader_delta_cum") or {})
    ready_decode_peak = max((int(r.get("ready_decode", 0)) for r in step_rows), default=0)
    decode_active_peak = max((int(r.get("decode_active", 0)) for r in step_rows), default=0)
    ready_decode_blocks_peak = max((int(r.get("ready_decode_logical_blocks", 0)) for r in step_rows), default=0)
    ready_decode_resident_blocks_peak = max((int(r.get("ready_decode_resident_blocks", 0)) for r in step_rows), default=0)
    decode_active_blocks_peak = max((int(r.get("decode_active_logical_blocks", 0)) for r in step_rows), default=0)
    decode_active_resident_blocks_peak = max((int(r.get("decode_active_resident_blocks", 0)) for r in step_rows), default=0)
    ready_decode_on_cpu_peak = max((int(r.get("ready_decode_on_cpu", 0)) for r in step_rows), default=0)
    ready_decode_on_gpu_peak = max((int(r.get("ready_decode_on_gpu", 0)) for r in step_rows), default=0)
    decode_active_on_cpu_peak = max((int(r.get("decode_active_on_cpu", 0)) for r in step_rows), default=0)
    decode_active_on_gpu_peak = max((int(r.get("decode_active_on_gpu", 0)) for r in step_rows), default=0)
    match_count = 0
    valid_count = 0
    finished_ordered = sorted(finished, key=lambda x: int(x.get("request_id", -1)))
    for idx, item in enumerate(finished_ordered):
        answers = list(prompt_records[idx].get("answers") or []) if idx < len(prompt_records) else []
        output = str(item.get("output", ""))
        if output.strip():
            valid_count += 1
        if answers and any(ans and ans in output for ans in answers):
            match_count += 1
    return {
        "group": group,
        "model_name": args.model_name,
        "prompt_profile": PROMPT_PROFILE,
        "metric_profile": METRIC_PROFILE,
        "content_source": args.content_source,
        "length_spec": args.length_spec,
        "initial_submit": int(args.initial_submit),
        "arrival_batch": int(args.arrival_batch),
        "arrival_interval_decode_steps": int(args.arrival_interval_decode_steps),
        "gpu_mem_frac": float(args.gpu_mem_frac),
        "max_new_tokens": int(args.max_new_tokens),
        "decode_micro_batch_size": int(args.decode_micro_batch_size),
        "decode_active_cap_initial": int(args.decode_active_cap_initial),
        "max_decode_active_cap": int(args.max_decode_active_cap),
        "request_count": int(len(prompt_records)),
        "actual_prompt_tokens": [int(x["actual_prompt_tokens"]) for x in prompt_records],
        "model_memory_gb": float(round(model_memory_gb, 4)),
        "kv_available_gb": float(round(kv_available_gb, 4)),
        "tokens_per_sec": float(total_generated_tokens / max(1e-9, wall_ms / 1000.0)),
        "decode_step_p95_ms": float(percentile(decode_step_ms, 0.95)),
        "success_rate": float(success_count / max(1, len(prompt_records))),
        "oom_rate": float(oom_count / max(1, len(prompt_records))),
        "match_rate": float(match_count / max(1, len(prompt_records))),
        "valid_rate": float(valid_count / max(1, len(prompt_records))),
        "success_count": int(success_count),
        "failed_count": int(fail_count),
        "decode_min_n_free": int(last.get("decode_min_n_free", 0)),
        "prefill_min_n_free": int(last.get("prefill_min_n_free", 0)),
        "global_min_n_free": int(last.get("global_min_n_free", 0)),
        "kv_total_blocks": int(last.get("kv_total_blocks", 0)),
        "kv_peak_used_blocks": int(last.get("kv_peak_used_blocks", 0)),
        "thrash_win16": float(last.get("thrash_win16", 0.0)),
        "decode_append_fail_count": int(last.get("decode_append_fail_count", 0)),
        "decode_backpressure_events": int(last.get("decode_backpressure_events", 0)),
        "prefill_backpressure_events": int(last.get("prefill_backpressure_events", 0)),
        "decode_retry_timeout_fail_count": int(last.get("decode_retry_timeout_fail_count", 0)),
        "decode_no_progress_steps": int(last.get("decode_no_progress_steps", 0)),
        "offloader_delta": off_cum,
        "offload_calls": int(off_cum.get("offload_calls", 0)),
        "prefetch_calls": int(off_cum.get("prefetch_calls", 0)),
        "offload_success": int(off_cum.get("offload_success", 0)),
        "prefetch_success": int(off_cum.get("prefetch_success", 0)),
        "ensure_success": int(off_cum.get("ensure_success", 0)),
        "auto_active_steps": int(last.get("decode_window_auto_active_steps", 0)),
        "p2_attempts": int(last.get("p2_attempts", 0)),
        "p2_successes": int(last.get("p2_successes", 0)),
        "p2_fail_streak": int(last.get("p2_fail_streak", 0)),
        "p2_blocks_p3": int(last.get("p2_blocks_p3", 0)),
        "decode_window_prunes": int(last.get("decode_window_prunes_cum", last.get("decode_window_prunes", 0))),
        "decode_window_tokens_dropped": int(last.get("decode_window_tokens_dropped_cum", last.get("decode_window_tokens_dropped", 0))),
        "ready_decode_peak": int(ready_decode_peak),
        "decode_active_peak": int(decode_active_peak),
        "ready_decode_blocks_peak": int(ready_decode_blocks_peak),
        "ready_decode_resident_blocks_peak": int(ready_decode_resident_blocks_peak),
        "decode_active_blocks_peak": int(decode_active_blocks_peak),
        "decode_active_resident_blocks_peak": int(decode_active_resident_blocks_peak),
        "ready_decode_on_cpu_peak": int(ready_decode_on_cpu_peak),
        "ready_decode_on_gpu_peak": int(ready_decode_on_gpu_peak),
        "decode_active_on_cpu_peak": int(decode_active_on_cpu_peak),
        "decode_active_on_gpu_peak": int(decode_active_on_gpu_peak),
        "cuda_free_min_gb": float(last.get("cuda_free_min_gb", 0.0)),
        "cuda_alloc_peak_gb": float(last.get("cuda_alloc_peak_gb", 0.0)),
        "cuda_reserved_peak_gb": float(last.get("cuda_reserved_peak_gb", 0.0)),
        "cuda_total_gb": float(last.get("cuda_total_gb", 0.0)),
        "avg_decode_microbatch_size": float(mean([float(r.get("decode_microbatches", 0)) for r in step_rows if int(r.get("decode_microbatches", 0)) > 0])) if any(int(r.get("decode_microbatches", 0)) > 0 for r in step_rows) else 0.0,
        "wall_ms": float(round(wall_ms, 3)),
        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        "compression_profile": compression_profile_for_group(group),
        "finished": finished_ordered,
    }


def run_engine_group(args: argparse.Namespace, group: str, prompt_records: List[Dict[str, Any]], progress_path: Path) -> Dict[str, Any]:
    engine: Optional[ManagedInferenceEngine] = None
    step_rows: List[Dict[str, Any]] = []
    finished: List[Dict[str, Any]] = []
    submitted = 0
    next_trigger = int(args.arrival_interval_decode_steps)
    decode_step_counter = 0
    model_memory_gb = 0.0
    kv_available_gb = 0.0
    t0 = time.perf_counter()
    try:
        engine = build_engine(args, group)
        model_memory_gb, kv_available_gb = compute_memory_stats(engine, float(args.gpu_mem_frac))
        initial = min(int(args.initial_submit), len(prompt_records))
        for rec in prompt_records[:initial]:
            engine.submit(rec["prompt"], request_id=int(rec["request_index"]), max_new_tokens=int(args.max_new_tokens))
            submitted += 1
        while engine.has_pending_requests() or submitted < len(prompt_records):
            stats = engine.step()
            just_finished = engine.collect_finished()
            if just_finished:
                finished.extend(just_finished)
            if int(stats.get("decode_tokens", 0)) > 0:
                decode_step_counter += 1
            if submitted < len(prompt_records):
                if (decode_step_counter >= next_trigger) or (not engine.has_pending_requests()):
                    end = min(len(prompt_records), submitted + int(args.arrival_batch))
                    for rec in prompt_records[submitted:end]:
                        engine.submit(rec["prompt"], request_id=int(rec["request_index"]), max_new_tokens=int(args.max_new_tokens))
                    submitted = end
                    next_trigger += int(args.arrival_interval_decode_steps)
            step_rec = {
                "group": group,
                "submitted_count": int(submitted),
                "finished_count": int(len(finished)),
                "decode_step_counter": int(decode_step_counter),
                **stats,
            }
            step_rows.append(step_rec)
            append_jsonl(progress_path, step_rec)
            if int(stats.get("step", 0)) >= int(args.max_engine_steps):
                raise RuntimeError(f"max_engine_steps_exceeded:{args.max_engine_steps}")
        wall_ms = (time.perf_counter() - t0) * 1000.0
        summary = summarize_engine_run(group, args, step_rows, finished, model_memory_gb, kv_available_gb, wall_ms, prompt_records)
        append_jsonl(progress_path, {"group": group, "event": "summary", **{k: v for k, v in summary.items() if k != "finished"}})
        return summary
    finally:
        if engine is not None:
            cleanup_engine(engine, f"p2_online_flow_{group}")


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    header = [
        "group","tokens_per_sec","decode_step_p95_ms","success_rate","oom_rate","match_rate","valid_rate",
        "offload_calls","prefetch_calls","ensure_success","decode_backpressure_events","prefill_backpressure_events","decode_append_fail_count",
        "ready_decode_peak","decode_active_peak","ready_decode_blocks_peak","ready_decode_resident_blocks_peak","decode_active_blocks_peak","decode_active_resident_blocks_peak",
        "ready_decode_on_cpu_peak","ready_decode_on_gpu_peak","decode_active_on_cpu_peak","decode_active_on_gpu_peak","p2_attempts","p2_successes","p2_fail_streak","p2_blocks_p3",
        "auto_active_steps","decode_window_prunes","decode_window_tokens_dropped","global_min_n_free","kv_total_blocks","kv_peak_used_blocks",
        "cuda_free_min_gb","cuda_alloc_peak_gb","cuda_reserved_peak_gb"
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rec in rows:
            vals = [
                str(rec.get("group", "")),
                f"{float(rec.get('tokens_per_sec', 0.0)):.4f}",
                f"{float(rec.get('decode_step_p95_ms', 0.0)):.3f}",
                f"{float(rec.get('success_rate', 0.0)):.4f}",
                f"{float(rec.get('oom_rate', 0.0)):.4f}",
                f"{float(rec.get('match_rate', 0.0)):.4f}",
                f"{float(rec.get('valid_rate', 0.0)):.4f}",
                str(int(rec.get("offload_calls", 0))),
                str(int(rec.get("prefetch_calls", 0))),
                str(int(rec.get("ensure_success", 0))),
                str(int(rec.get("decode_backpressure_events", 0))),
                str(int(rec.get("prefill_backpressure_events", 0))),
                str(int(rec.get("decode_append_fail_count", 0))),
                str(int(rec.get("ready_decode_peak", 0))),
                str(int(rec.get("decode_active_peak", 0))),
                str(int(rec.get("ready_decode_blocks_peak", 0))),
                str(int(rec.get("ready_decode_resident_blocks_peak", 0))),
                str(int(rec.get("decode_active_blocks_peak", 0))),
                str(int(rec.get("decode_active_resident_blocks_peak", 0))),
                str(int(rec.get("ready_decode_on_cpu_peak", 0))),
                str(int(rec.get("ready_decode_on_gpu_peak", 0))),
                str(int(rec.get("decode_active_on_cpu_peak", 0))),
                str(int(rec.get("decode_active_on_gpu_peak", 0))),
                str(int(rec.get("p2_attempts", 0))),
                str(int(rec.get("p2_successes", 0))),
                str(int(rec.get("p2_fail_streak", 0))),
                str(int(rec.get("p2_blocks_p3", 0))),
                str(int(rec.get("auto_active_steps", 0))),
                str(int(rec.get("decode_window_prunes", 0))),
                str(int(rec.get("decode_window_tokens_dropped", 0))),
                str(int(rec.get("global_min_n_free", 0))),
                str(int(rec.get("kv_total_blocks", 0))),
                str(int(rec.get("kv_peak_used_blocks", 0))),
                f"{float(rec.get('cuda_free_min_gb', 0.0)):.4f}",
                f"{float(rec.get('cuda_alloc_peak_gb', 0.0)):.4f}",
                f"{float(rec.get('cuda_reserved_peak_gb', 0.0)):.4f}",
            ]
            f.write(",".join(vals) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P2 online mixed-flow pressure benchmark")
    parser.add_argument("--model-name", type=str, default=LOCAL_MODEL_PATH)
    parser.add_argument("--groups", type=str, default=",".join(GROUP_ORDER))
    parser.add_argument("--content-source", type=str, default="synthetic", choices=["synthetic", "longbench", "ruler", "mixed"])
    parser.add_argument("--longbench-path", type=str, default="")
    parser.add_argument("--ruler-path", type=str, default="")
    parser.add_argument("--frontier-json", type=str, default="")
    parser.add_argument("--frontier-group", type=str, default="main_auto_compress")
    parser.add_argument("--length-spec", type=str, default="8192:16,16384:16,32768:16")
    parser.add_argument("--initial-submit", type=int, default=16)
    parser.add_argument("--arrival-batch", type=int, default=4)
    parser.add_argument("--arrival-interval-decode-steps", type=int, default=8)
    parser.add_argument("--gpu-mem-frac", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--decode-micro-batch-size", type=int, default=16)
    parser.add_argument("--decode-active-cap-initial", type=int, default=16)
    parser.add_argument("--max-decode-active-cap", type=int, default=16)
    parser.add_argument("--prefill-batch-size", type=int, default=8)
    parser.add_argument("--max-prefill-active", type=int, default=16)
    parser.add_argument("--prefill-token-budget-per-step", type=int, default=16384)
    parser.add_argument("--cpu-mem-gb", type=float, default=32.0)
    parser.add_argument("--max-engine-steps", type=int, default=200000)
    parser.add_argument("--out-prefix", type=str, default="benchmark_p2_online_mixed_flow")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    maybe_apply_frontier_hint(args)
    groups = [g.strip() for g in str(args.groups).split(",") if g.strip()]
    for g in groups:
        if g not in GROUP_ARGS:
            raise ValueError(f"unsupported group: {g}")

    out_prefix = Path(args.out_prefix)
    progress_path = out_prefix.with_suffix(".progress.jsonl")
    result_json = out_prefix.with_name(out_prefix.name + "_results.json")
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    progress_path.write_text("", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    prompt_records = build_prompt_records(tokenizer, parse_length_spec(args.length_spec), args)
    append_jsonl(progress_path, {
        "event": "meta",
        "prompt_profile": PROMPT_PROFILE,
        "metric_profile": METRIC_PROFILE,
        "request_count": int(len(prompt_records)),
        "groups": groups,
        "gpu_mem_frac": float(args.gpu_mem_frac),
        "length_spec": args.length_spec,
        "initial_submit": int(args.initial_submit),
        "arrival_batch": int(args.arrival_batch),
        "arrival_interval_decode_steps": int(args.arrival_interval_decode_steps),
        "decode_micro_batch_size": int(args.decode_micro_batch_size),
        "decode_active_cap_initial": int(args.decode_active_cap_initial),
        "max_decode_active_cap": int(args.max_decode_active_cap),
    })

    rows: List[Dict[str, Any]] = []
    for group in groups:
        rows.append(run_engine_group(args, group, prompt_records, progress_path))

    payload = {
        "meta": {
            "task": "p2_online_mixed_flow",
            "model_name": args.model_name,
            "groups": groups,
            "content_source": args.content_source,
            "length_spec": args.length_spec,
            "initial_submit": int(args.initial_submit),
            "arrival_batch": int(args.arrival_batch),
            "arrival_interval_decode_steps": int(args.arrival_interval_decode_steps),
            "gpu_mem_frac": float(args.gpu_mem_frac),
            "max_new_tokens": int(args.max_new_tokens),
            "decode_micro_batch_size": int(args.decode_micro_batch_size),
            "decode_active_cap_initial": int(args.decode_active_cap_initial),
            "max_decode_active_cap": int(args.max_decode_active_cap),
            "prompt_profile": PROMPT_PROFILE,
            "metric_profile": METRIC_PROFILE,
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        },
        "rows": rows,
    }
    write_json(result_json, payload)
    write_summary_csv(summary_csv, rows)
    print(f"Saved: {result_json}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()

