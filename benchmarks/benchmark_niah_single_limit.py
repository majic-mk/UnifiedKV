import argparse
import gc
import inspect
import json
import os
import random
import re
import signal
import subprocess
import tempfile
import time
import traceback
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ALLOC_CONF_ENABLED = (
    str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")).strip() == "expandable_segments:True"
)
import torch

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from engine import ManagedInferenceEngine

# Mirror copy for remote execution.
# Keep runtime semantics in sync with:
#   C:\Users\mamengkui\OneDrive\文档\Playground\autodl_mirror\benchmarks\benchmark_niah_single_limit.py


LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 32
DEPTHS = [0.10, 0.50, 0.90]
TARGET_EM_THRESHOLD = 0.75
TARGET_VALID_THRESHOLD = 0.75
GPU_MEM_FRAC_MAX = 0.78
GPU_MEM_FRAC_FALLBACK_STEP = 0.02
GPU_MEM_FRAC_FALLBACK_TRIES = 4
COMPRESS_MEM_FRAC_SAFETY_MARGIN = 0.0
MIN_TOKENS = 2000
MAX_TOKENS = 256000
PRECISION_TOKENS = 1000
SMOKE_LENGTHS = [16000]
ANCHOR_LEN_1 = 128000
ANCHOR_LEN_2 = 256000
SEED = 20260322
CASE_TIMEOUT_S = 600
SINGLE_CASE_SUBPROCESS_GRACE_S = 60
QUALITY_EVAL_SUBPROCESS_GRACE_S = 120


ENGINE_INIT_PARAMS = {
    name
    for name in inspect.signature(ManagedInferenceEngine.__init__).parameters.keys()
    if name != "self"
}

COMMON_BASE = {
    "model_name": LOCAL_MODEL_PATH,
    "cpu_mem_gb": 32.0,
    "chunk_size": 1024,
    "max_new_tokens": MAX_NEW_TOKENS,
    "prefill_batch_size": 1,
    "decode_micro_batch_size": 0,
    "sink_len": 16,
    "obs_len": 16,
    "decode_window_sink_len": 64,
    "decode_path_mode": "auto",
    "decode_paged_flash_enabled": True,
}

MODE_OFF_ARGS = {
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": False,
    "p2_enabled": False,
}

MODE_MAIN_AUTO_ARGS = {
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": True,
    "offload_budget_blocks": 256,
    "prefetch_budget_blocks": 256,
    "decode_window_cuda_pressure_min_gb": 1.0,
    "decode_window_cuda_emergency_min_gb": 0.5,
    "kv_min_resident_ratio": 0.20,
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

MODE_P2_ONLY_ARGS = {
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": False,
    "offload_budget_blocks": 256,
    "prefetch_budget_blocks": 256,
    "offload_budget_blocks_max": 320,
    "prefetch_budget_blocks_max": 320,
    "decode_reserve_blocks": 96,
    "ready_decode_eviction_threshold": 48,
    "p2_retry_windows_before_p3": 3,
    "p2_target_free_blocks": 0,
    "p3_pressure_threshold": 0,
    "p3_emergency_threshold": 0,
    "p3_disable_until_p2_active": True,
    "p3_exit_before_p2_exit": True,
    "decode_window_cuda_pressure_min_gb": 1.0,
    "decode_window_cuda_emergency_min_gb": 0.5,
    "kv_min_resident_ratio": 0.20,
}

COMPRESS_PROFILES: Dict[str, Dict[str, float]] = {
    "quality": {
        "retain_ratio": 0.75,
    },
    "serving": {
        "retain_ratio": 0.10,
    },
}
DEFAULT_COMPRESS_PROFILE = "quality"
PREFILL_COMPRESS_ARGS = dict(COMPRESS_PROFILES[DEFAULT_COMPRESS_PROFILE])

PREFILL_RAW_ARGS = {
    "retain_ratio": 1.0,
}

TRACK_ALIAS = {
    "off_compress": ("off", "compress"),
    "off_raw": ("off", "raw"),
    "p2_only_compress": ("p2_only", "compress"),
    "main_auto_compress": ("main_auto", "compress"),
    "main_auto_raw": ("main_auto", "raw"),
}

EXTRA_ENGINE_ARGS: Dict[str, object] = {}
NEEDLE_DEBUG_DIR = ""

METRIC_PROFILE = "strict+compat"
PROMPT_PROFILE = "chat_32"
STRICT_DEFINITION = "first non-empty line must be exactly 6 digits (optional 'answer:' prefix)"


def _first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _first_6digit_compat(text: str) -> str:
    m = re.search(r"\b(\d{6})\b", str(text))
    return m.group(1) if m else ""


def _first_6digit_strict(text: str) -> str:
    first = _first_nonempty_line(text).strip("`").strip()
    m = re.fullmatch(r"(?:answer\s*:\s*)?(\d{6})", first or "", flags=re.IGNORECASE)
    return m.group(1) if m else ""


def _eval_passkey_single(output: str, answer: str) -> Dict[str, float]:
    pred_strict = _first_6digit_strict(output)
    pred_compat = _first_6digit_compat(output)
    valid = 1.0 if pred_strict else 0.0
    em = 1.0 if pred_strict == answer and bool(pred_strict) else 0.0
    valid_compat = 1.0 if pred_compat else 0.0
    em_compat = 1.0 if pred_compat == answer and bool(pred_compat) else 0.0
    contains = 1.0 if answer in str(output) else 0.0
    em_given_valid = (em / valid) if valid > 1e-12 else 0.0
    return {
        "passkey_first_em": em,
        "valid_rate": valid,
        "em_given_valid": em_given_valid,
        "passkey_first_em_compat": em_compat,
        "valid_rate_compat": valid_compat,
        "passkey_contains_em": contains,
        "pred_strict": pred_strict,
        "pred_compat": pred_compat,
    }


def _skip_reason_counts(metrics: Dict) -> Dict[str, int]:
    delta = metrics.get("offloader_delta", {}) or {}
    return {
        "no_candidate_blocks": int(delta.get("window_prune_skip_no_candidate_blocks", 0)),
        "trigger_not_met": int(delta.get("window_prune_skip_trigger_not_met", 0)),
        "min_drop_not_met": int(delta.get("window_prune_skip_min_drop_not_met", 0)),
    }


def _non_trigger_reason(metrics: Dict) -> str:
    prunes = int(metrics.get("decode_window_prunes", 0))
    auto_steps = int(metrics.get("decode_window_auto_active_steps", 0))
    if prunes > 0:
        return ""
    if auto_steps <= 0:
        return "auto_not_activated"
    ranked = sorted(_skip_reason_counts(metrics).items(), key=lambda kv: kv[1], reverse=True)
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]
    return "no_prune_events"


def _make_prompt_for_target_tokens(tokenizer, target_tokens: int, depth_ratio: float, key: str) -> Tuple[str, int]:
    depth = float(max(0.0, min(1.0, depth_ratio)))
    intro = (
        "You are a retrieval system.\n"
        "Find PASSKEY_RECORD in the context and return ONLY the 6-digit number.\n"
    )
    needle = f"\nPASSKEY_RECORD: {key}\n"
    question = (
        "\nQuestion: What is PASSKEY_RECORD?\n"
        "Instruction: Output exactly one line with only the 6-digit number.\n"
    )
    filler = "Irrelevant context text about systems and caching.\n"

    intro_ids = tokenizer(intro, add_special_tokens=False).input_ids
    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids
    if not filler_ids:
        filler_ids = [32]

    fixed = len(intro_ids) + len(needle_ids) + len(question_ids)
    remaining = max(0, int(target_tokens) - fixed)
    left_tokens = int(round(remaining * depth))
    right_tokens = max(0, remaining - left_tokens)

    def _fill_to(n: int) -> List[int]:
        if n <= 0:
            return []
        reps = (n + len(filler_ids) - 1) // len(filler_ids)
        return (filler_ids * reps)[:n]

    ids = intro_ids + _fill_to(left_tokens) + needle_ids + _fill_to(right_tokens) + question_ids
    prompt_user = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    prompt = prompt_user
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = prompt_user
    # Avoid an extra full re-tokenization pass (expensive at 128k+/256k prompts).
    # For this benchmark, id-length is a sufficient prompt-length proxy.
    actual_tokens = int(len(ids))
    return prompt, actual_tokens


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    last = len(haystack) - len(needle) + 1
    first = needle[0]
    for idx in range(last):
        if haystack[idx] != first:
            continue
        if list(haystack[idx : idx + len(needle)]) == list(needle):
            return idx
    return -1


def _locate_needle_token_span(tokenizer, prompt: str, answer: str) -> Dict[str, object]:
    marker = f"PASSKEY_RECORD: {answer}"
    diag: Dict[str, object] = {
        "needle_marker": marker,
        "needle_char_start": int(str(prompt).find(marker)),
        "needle_char_end": -1,
        "full_prompt_tokens": 0,
        "needle_token_start": -1,
        "needle_token_end": -1,
        "needle_locate_error": "",
    }
    if int(diag["needle_char_start"]) >= 0:
        diag["needle_char_end"] = int(diag["needle_char_start"]) + len(marker)

    try:
        enc = tokenizer(prompt, truncation=False, return_offsets_mapping=True)
        input_ids = enc.get("input_ids", [])
        offsets = enc.get("offset_mapping", [])
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if offsets and isinstance(offsets[0], tuple):
            offsets = list(offsets)
        elif offsets and isinstance(offsets[0], list) and offsets and isinstance(offsets[0][0], tuple):
            offsets = offsets[0]
        diag["full_prompt_tokens"] = int(len(input_ids))
        char_start = int(diag["needle_char_start"])
        char_end = int(diag["needle_char_end"])
        if char_start >= 0 and char_end > char_start and offsets:
            token_idxs = [
                idx
                for idx, (start, end) in enumerate(offsets)
                if int(end) > char_start and int(start) < char_end
            ]
            if token_idxs:
                diag["needle_token_start"] = int(token_idxs[0])
                diag["needle_token_end"] = int(token_idxs[-1] + 1)
                return diag
    except Exception as exc:
        diag["needle_locate_error"] = f"offset_mapping_failed: {exc}"

    try:
        full_ids = tokenizer(prompt, truncation=False).input_ids
        if full_ids and isinstance(full_ids[0], list):
            full_ids = full_ids[0]
        marker_ids = tokenizer(marker, add_special_tokens=False).input_ids
        if marker_ids and isinstance(marker_ids[0], list):
            marker_ids = marker_ids[0]
        diag["full_prompt_tokens"] = int(len(full_ids))
        pos = _find_subsequence(full_ids, marker_ids)
        if pos >= 0:
            diag["needle_token_start"] = int(pos)
            diag["needle_token_end"] = int(pos + len(marker_ids))
            return diag
        if not diag["needle_locate_error"]:
            diag["needle_locate_error"] = "needle_subsequence_not_found"
    except Exception as exc:
        diag["needle_locate_error"] = f"token_search_failed: {exc}"
    return diag


def _build_needle_block_diagnostic(engine: ManagedInferenceEngine, prompt: str, answer: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    summary = _locate_needle_token_span(engine.tokenizer, prompt, answer)
    sched = getattr(engine, "scheduler", None)
    debug_map = dict(getattr(sched, "last_compress_debug", {}) or {}) if sched is not None else {}
    seq_debug = {}
    if debug_map:
        latest_sid = sorted(debug_map.keys())[-1]
        seq_debug = dict(debug_map.get(latest_sid) or {})
    layers = list(seq_debug.get("layers") or [])
    block_size = int(seq_debug.get("block_size") or getattr(getattr(sched, "pool", None), "B", 0) or 0)
    summary.update(
        {
            "needle_block_size": int(block_size),
            "needle_block_start": -1,
            "needle_block_end": -1,
            "needle_seq_id": int(seq_debug.get("seq_id", -1) or -1),
            "needle_seq_logical_len": int(seq_debug.get("logical_seq_len", 0) or 0),
            "needle_seq_compressed_len": int(seq_debug.get("compressed_seq_len", 0) or 0),
            "needle_allocated_block_count": int(seq_debug.get("allocated_block_count", 0) or 0),
            "needle_layers_total": int(len(layers)),
            "needle_layers_mid_overlap": 0,
            "needle_layers_any_preserved": 0,
            "needle_layers_all_preserved": 0,
            "needle_layers_dropped": 0,
            "needle_layers_sink_only": 0,
            "needle_layers_obs_only": 0,
        }
    )
    token_start = int(summary.get("needle_token_start", -1) or -1)
    token_end = int(summary.get("needle_token_end", -1) or -1)
    if block_size > 0 and token_start >= 0 and token_end > token_start:
        summary["needle_block_start"] = int(token_start // block_size)
        summary["needle_block_end"] = int((token_end - 1) // block_size)

    layer_details: List[Dict[str, object]] = []
    for layer in layers:
        logical_len = int(layer.get("logical_seq_len", 0) or 0)
        sink_len = int(layer.get("sink_len", 0) or 0)
        obs_len = int(layer.get("obs_len", 0) or 0)
        retained = {int(x) for x in list(layer.get("retained_block_idx") or [])}
        mid_start = sink_len
        mid_end = max(mid_start, logical_len - obs_len)
        detail = {
            "layer_id": int(layer.get("layer_id", -1) or -1),
            "logical_seq_len": logical_len,
            "compressed_seq_len": int(layer.get("compressed_seq_len", 0) or 0),
            "mid_block_count": int(layer.get("mid_block_count", 0) or 0),
            "retained_block_count": int(layer.get("retained_block_count", 0) or 0),
            "needle_mid_blocks": [],
            "needle_any_preserved": False,
            "needle_all_preserved": False,
            "needle_in_sink_only": False,
            "needle_in_obs_only": False,
            "reason": str(layer.get("reason", "")),
        }
        if token_start >= 0 and token_end > token_start:
            if token_end <= sink_len:
                detail["needle_in_sink_only"] = True
                summary["needle_layers_sink_only"] = int(summary["needle_layers_sink_only"]) + 1
            elif token_start >= mid_end and logical_len > 0:
                detail["needle_in_obs_only"] = True
                summary["needle_layers_obs_only"] = int(summary["needle_layers_obs_only"]) + 1
            else:
                overlap_start = max(mid_start, token_start)
                overlap_end = min(mid_end, token_end)
                if overlap_start < overlap_end and block_size > 0:
                    rel_start = overlap_start - mid_start
                    rel_end = overlap_end - mid_start
                    block_lo = int(rel_start // block_size)
                    block_hi = int((rel_end - 1) // block_size)
                    needle_mid_blocks = list(range(block_lo, block_hi + 1))
                    detail["needle_mid_blocks"] = [int(x) for x in needle_mid_blocks]
                    any_preserved = any(int(b) in retained for b in needle_mid_blocks)
                    all_preserved = all(int(b) in retained for b in needle_mid_blocks)
                    detail["needle_any_preserved"] = bool(any_preserved)
                    detail["needle_all_preserved"] = bool(all_preserved)
                    summary["needle_layers_mid_overlap"] = int(summary["needle_layers_mid_overlap"]) + 1
                    if any_preserved:
                        summary["needle_layers_any_preserved"] = int(summary["needle_layers_any_preserved"]) + 1
                    if all_preserved:
                        summary["needle_layers_all_preserved"] = int(summary["needle_layers_all_preserved"]) + 1
                    if not any_preserved:
                        summary["needle_layers_dropped"] = int(summary["needle_layers_dropped"]) + 1
        layer_details.append(detail)

    detail_payload = {
        "summary": summary,
        "seq_debug": seq_debug,
        "layer_details": layer_details,
    }
    return summary, detail_payload


def _write_needle_debug_payload(
    mode: str,
    prefill_track: str,
    stage: str,
    target_len: int,
    depth: float,
    payload: Dict[str, object],
) -> str:
    if not str(NEEDLE_DEBUG_DIR).strip():
        return ""
    os.makedirs(NEEDLE_DEBUG_DIR, exist_ok=True)
    safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage))
    fname = (
        f"{mode}_{prefill_track}_{safe_stage}_"
        f"len{int(target_len)}_depth{int(round(float(depth) * 100)):03d}.json"
    )
    out_path = os.path.join(NEEDLE_DEBUG_DIR, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def _align_down(value: int, step: int) -> int:
    if step <= 0:
        return int(value)
    return int(value // step * step)


def _align_up(value: int, step: int) -> int:
    if step <= 0:
        return int(value)
    return int(((value + step - 1) // step) * step)


def _cleanup_engine_pending(engine: ManagedInferenceEngine, reason: str):
    try:
        if hasattr(engine, "has_pending_requests") and engine.has_pending_requests():
            req_map = dict(getattr(engine, "_requests", {}) or {})
            for _, req in req_map.items():
                try:
                    engine._mark_request_failed(req, RuntimeError(reason))
                except Exception:
                    pass
    except Exception:
        pass
    try:
        if hasattr(engine, "_reset_online_runtime"):
            engine._reset_online_runtime(clear_request_counter=False)
    except Exception:
        pass


def _build_engine(gpu_mem_frac: float, mode: str, prefill_track: str) -> ManagedInferenceEngine:
    args = dict(COMMON_BASE)
    args["gpu_mem_frac"] = float(gpu_mem_frac)
    if mode == "off":
        args.update(MODE_OFF_ARGS)
    elif mode == "p2_only":
        args.update(MODE_P2_ONLY_ARGS)
    elif mode == "main_auto":
        args.update(MODE_MAIN_AUTO_ARGS)
    else:
        raise ValueError(f"unknown mode: {mode}")
    if prefill_track == "compress":
        args.update(PREFILL_COMPRESS_ARGS)
    elif prefill_track == "raw":
        args.update(PREFILL_RAW_ARGS)
    else:
        raise ValueError(f"unknown prefill_track: {prefill_track}")
    args.update(EXTRA_ENGINE_ARGS)
    if "obs_len" in args and "snapkv_observation_len" in ENGINE_INIT_PARAMS:
        args["snapkv_observation_len"] = args.pop("obs_len")
    filtered_args = {k: v for k, v in args.items() if k in ENGINE_INIT_PARAMS}
    return ManagedInferenceEngine(**filtered_args)


def _run_single_case(engine: ManagedInferenceEngine, mode: str, prefill_track: str, target_len: int, depth: float, seed: int, stage: str, case_timeout_s: int) -> Dict:
    rng = random.Random((seed * 1000003) ^ (target_len * 1009) ^ int(depth * 1000))
    answer = f"{rng.randint(100000, 999999)}"
    prompt, actual_len = _make_prompt_for_target_tokens(engine.tokenizer, target_len, depth, answer)
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "prefill_track": prefill_track,
        "stage": stage,
        "target_len": int(target_len),
        "actual_prompt_tokens": int(actual_len),
        "depth": float(depth),
        "success": 0,
        "passkey_first_em": 0.0,
        "valid_rate": 0.0,
        "em_given_valid": 0.0,
        "tokens_s": 0.0,
        "p95_ms": 0.0,
        "decode_min_n_free": 0,
        "thrash_win16": 0.0,
        "prunes": 0,
        "dropped": 0,
        "auto_active_steps": 0,
        "decode_path_selected": "",
        "decode_path_fallback_count": 0,
        "decode_path_fallback_reason_topk": {},
        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        "metric_profile": METRIC_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "compression_profile": "",
        "auto_trigger_observed": 0,
        "non_trigger_reason": "run_failed",
        "error": "",
        "pred_strict": "",
        "pred_compat": "",
        "wall_ms": 0.0,
        "needle_token_start": -1,
        "needle_token_end": -1,
        "needle_block_start": -1,
        "needle_block_end": -1,
        "needle_seq_logical_len": 0,
        "needle_seq_compressed_len": 0,
        "needle_allocated_block_count": 0,
        "needle_layers_total": 0,
        "needle_layers_mid_overlap": 0,
        "needle_layers_any_preserved": 0,
        "needle_layers_all_preserved": 0,
        "needle_layers_dropped": 0,
        "needle_layers_sink_only": 0,
        "needle_layers_obs_only": 0,
        "needle_debug_path": "",
        "needle_locate_error": "",
        "output_preview": "",
        "output_len_chars": 0,
        "has_nonempty_output": 0,
        "generated_tokens_metric": 0,
        "decode_steps_metric": 0,
        "terminal_state": "",
        "terminal_error": "",
        "terminal_decode_steps": 0,
        "terminal_arrival_step": -1,
        "terminal_finished_step": -1,
        "terminal_last_retryable_reason": "",
        "terminal_last_retryable_error": "",
    }
    t0 = time.perf_counter()
    print(
        json.dumps(
            {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "prefill_track": prefill_track,
                "stage": f"{stage}_start",
                "target_len": int(target_len),
                "depth": float(depth),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        def _do_generate():
            return engine.generate([prompt], return_metrics=True, return_details=True)

        if int(case_timeout_s) > 0 and hasattr(signal, "SIGALRM"):
            old_handler = signal.getsignal(signal.SIGALRM)

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"single_case_timeout>{int(case_timeout_s)}s")

            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(case_timeout_s))
            try:
                outputs, metrics, details = _do_generate()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            outputs, metrics, details = _do_generate()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        output_text = outputs[0] if (isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], str)) else ""
        detail0 = {}
        if isinstance(details, dict):
            for key in ("states", "errors", "last_retryable_reasons", "last_retryable_errors", "request_decode_steps", "arrival_steps", "finished_steps"):
                val = details.get(key)
                if isinstance(val, list) and len(val) > 0:
                    detail0[key] = val[0]
        quality = _eval_passkey_single(output_text, answer)
        needle_summary, needle_payload = _build_needle_block_diagnostic(engine, prompt, answer)
        needle_debug_path = _write_needle_debug_payload(
            mode,
            prefill_track,
            stage,
            int(target_len),
            float(depth),
            needle_payload,
        )
        has_nonempty_output = bool(str(output_text).strip())
        timeout_like = int(case_timeout_s) > 0 and wall_ms >= (int(case_timeout_s) * 1000.0 - 50.0)
        terminal_state = str(detail0.get("states", "") or "")
        terminal_error = str(detail0.get("errors", "") or "")
        success_flag = 1 if terminal_state == "DONE" else 0
        err_msg = ""
        if success_flag != 1:
            err_msg = terminal_error
            if not err_msg:
                if not has_nonempty_output:
                    err_msg = "empty_output"
                elif timeout_like:
                    err_msg = f"single_case_timeout>{int(case_timeout_s)}s_or_stalled"
                else:
                    err_msg = f"terminal_state_{terminal_state or 'UNKNOWN'}"
            _cleanup_engine_pending(engine, err_msg)
        rec.update({
            "success": int(success_flag),
            "passkey_first_em": float(quality["passkey_first_em"]),
            "valid_rate": float(quality["valid_rate"]),
            "em_given_valid": float(quality["em_given_valid"]),
            "tokens_s": float(metrics.get("tokens_per_sec", 0.0)),
            "p95_ms": float(metrics.get("decode_step_p95_ms", 0.0)),
            "decode_min_n_free": int(metrics.get("decode_min_n_free", 0)),
            "thrash_win16": float(metrics.get("thrash_win16", 0.0)),
            "prunes": int(metrics.get("decode_window_prunes", 0)),
            "dropped": int(metrics.get("decode_window_tokens_dropped", 0)),
            "auto_active_steps": int(metrics.get("decode_window_auto_active_steps", 0)),
            "decode_path_selected": str(metrics.get("decode_path_selected", "")),
            "decode_path_fallback_count": int(metrics.get("decode_path_fallback_count", 0)),
            "decode_path_fallback_reason_topk": dict(metrics.get("decode_path_fallback_reason_topk", {}) or {}),
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
            "metric_profile": METRIC_PROFILE,
            "prompt_profile": PROMPT_PROFILE,
            "compression_profile": str(
                f"sink={int(getattr(engine.scheduler.snapkv, 'Ls', 0))};"
                f"obs={int(getattr(engine.scheduler.snapkv, 'Lo', 0))};"
                f"retain={float(getattr(engine.scheduler.snapkv, 'retain_ratio', 0.0)):.3f};"
                f"decode_sink={int(getattr(engine.scheduler, 'decode_window_sink_len', 0))};"
                f"decode_recent={int(getattr(engine.scheduler, 'decode_window_recent_len', 0))}"
            ),
            "auto_trigger_observed": int(
                int(metrics.get("decode_window_auto_active_steps", 0)) > 0
                and int(metrics.get("decode_window_prunes", 0)) > 0
            ),
            "non_trigger_reason": _non_trigger_reason(metrics),
            "pred_strict": str(quality["pred_strict"]),
            "pred_compat": str(quality["pred_compat"]),
            "error": err_msg,
            "wall_ms": float(round(wall_ms, 3)),
            "needle_token_start": int(needle_summary.get("needle_token_start", -1) or -1),
            "needle_token_end": int(needle_summary.get("needle_token_end", -1) or -1),
            "needle_block_start": int(needle_summary.get("needle_block_start", -1) or -1),
            "needle_block_end": int(needle_summary.get("needle_block_end", -1) or -1),
            "needle_seq_logical_len": int(needle_summary.get("needle_seq_logical_len", 0) or 0),
            "needle_seq_compressed_len": int(needle_summary.get("needle_seq_compressed_len", 0) or 0),
            "needle_allocated_block_count": int(needle_summary.get("needle_allocated_block_count", 0) or 0),
            "needle_layers_total": int(needle_summary.get("needle_layers_total", 0) or 0),
            "needle_layers_mid_overlap": int(needle_summary.get("needle_layers_mid_overlap", 0) or 0),
            "needle_layers_any_preserved": int(needle_summary.get("needle_layers_any_preserved", 0) or 0),
            "needle_layers_all_preserved": int(needle_summary.get("needle_layers_all_preserved", 0) or 0),
            "needle_layers_dropped": int(needle_summary.get("needle_layers_dropped", 0) or 0),
            "needle_layers_sink_only": int(needle_summary.get("needle_layers_sink_only", 0) or 0),
            "needle_layers_obs_only": int(needle_summary.get("needle_layers_obs_only", 0) or 0),
            "needle_debug_path": str(needle_debug_path),
            "needle_locate_error": str(needle_summary.get("needle_locate_error", "") or ""),
            "output_preview": str(output_text[:200]),
            "output_len_chars": int(len(output_text)),
            "has_nonempty_output": int(has_nonempty_output),
            "generated_tokens_metric": int(metrics.get("generated_tokens", 0) or 0),
            "decode_steps_metric": int(metrics.get("decode_steps", 0) or 0),
            "terminal_state": terminal_state,
            "terminal_error": terminal_error,
            "terminal_decode_steps": int(detail0.get("request_decode_steps", 0) or 0),
            "terminal_arrival_step": int(detail0.get("arrival_steps", -1) or -1),
            "terminal_finished_step": int(detail0.get("finished_steps", -1) or -1),
            "terminal_last_retryable_reason": str(detail0.get("last_retryable_reasons", "") or ""),
            "terminal_last_retryable_error": str(detail0.get("last_retryable_errors", "") or ""),
        })
    except Exception as exc:
        rec["error"] = str(exc)
        rec["wall_ms"] = float(round((time.perf_counter() - t0) * 1000.0, 3))
        _cleanup_engine_pending(engine, rec["error"] or "single_case_exception")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rec


def _single_case_subprocess_timeout_s(case_timeout_s: int) -> int:
    timeout_s = max(0, int(case_timeout_s))
    if timeout_s <= 0:
        return 600
    return max(120, timeout_s + SINGLE_CASE_SUBPROCESS_GRACE_S)


def _run_single_case_subprocess(
    mode: str,
    prefill_track: str,
    target_len: int,
    depth: float,
    seed: int,
    stage: str,
    gpu_mem_frac: float,
    case_timeout_s: int,
) -> Dict:
    fd, json_path = tempfile.mkstemp(prefix="niah_single_case_", suffix=".json")
    os.close(fd)
    try:
        os.unlink(json_path)
    except FileNotFoundError:
        pass
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-single-case-json-out",
        json_path,
        "--child-mode",
        str(mode),
        "--child-prefill-track",
        str(prefill_track),
        "--child-target-len",
        str(int(target_len)),
        "--child-depth",
        str(float(depth)),
        "--child-stage",
        str(stage),
        "--model-name",
        str(COMMON_BASE["model_name"]),
        "--max-new-tokens",
        str(int(MAX_NEW_TOKENS)),
        "--target-em-threshold",
        str(float(TARGET_EM_THRESHOLD)),
        "--target-valid-threshold",
        str(float(TARGET_VALID_THRESHOLD)),
        "--gpu-mem-frac-max",
        str(float(gpu_mem_frac)),
        "--gpu-mem-frac-fallback-step",
        "0",
        "--gpu-mem-frac-fallback-tries",
        "1",
        "--compress-retain-ratio",
        str(float(PREFILL_COMPRESS_ARGS.get("retain_ratio", 1.0))),
        "--seed",
        str(int(seed)),
        "--case-timeout-s",
        str(int(case_timeout_s)),
        "--sink-len",
        str(int(COMMON_BASE.get("sink_len", 0))),
        "--obs-len",
        str(int(COMMON_BASE.get("obs_len", 0))),
        "--decode-window-sink-len",
        str(int(COMMON_BASE.get("decode_window_sink_len", 0))),
        "--decode-window-recent-len",
        str(int(MODE_MAIN_AUTO_ARGS.get("decode_window_recent_len", 0))),
    ]
    if EXTRA_ENGINE_ARGS:
        cmd.extend(
            [
                "--engine-overrides-json",
                json.dumps(EXTRA_ENGINE_ARGS, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    if str(NEEDLE_DEBUG_DIR).strip():
        cmd.extend(["--needle-debug-dir", str(NEEDLE_DEBUG_DIR)])
    try:
        child_timeout_s = _single_case_subprocess_timeout_s(case_timeout_s)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=child_timeout_s,
            check=False,
        )
        payload = None
        payload_error = ""
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                payload_error = f"child_payload_read_failed: {exc}"
        if (proc.returncode != 0) or (payload is None):
            return {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "prefill_track": prefill_track,
                "stage": stage,
                "target_len": int(target_len),
                "actual_prompt_tokens": 0,
                "depth": float(depth),
                "success": 0,
                "passkey_first_em": 0.0,
                "valid_rate": 0.0,
                "em_given_valid": 0.0,
                "tokens_s": 0.0,
                "p95_ms": 0.0,
                "decode_min_n_free": 0,
                "thrash_win16": 0.0,
                "prunes": 0,
                "dropped": 0,
                "auto_active_steps": 0,
                "decode_path_selected": "",
                "decode_path_fallback_count": 0,
                "decode_path_fallback_reason_topk": {},
                "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
                "metric_profile": METRIC_PROFILE,
                "prompt_profile": PROMPT_PROFILE,
                "compression_profile": "",
                "auto_trigger_observed": 0,
                "non_trigger_reason": "child_single_case_failed",
                "error": (payload_error or proc.stderr or proc.stdout or f"child_returncode={proc.returncode}")[:800],
                "pred_strict": "",
                "pred_compat": "",
                "wall_ms": 0.0,
            }
        return dict(payload.get("rec") or {})
    except subprocess.TimeoutExpired as exc:
        return {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "prefill_track": prefill_track,
            "stage": stage,
            "target_len": int(target_len),
            "actual_prompt_tokens": 0,
            "depth": float(depth),
            "success": 0,
            "passkey_first_em": 0.0,
            "valid_rate": 0.0,
            "em_given_valid": 0.0,
            "tokens_s": 0.0,
            "p95_ms": 0.0,
            "decode_min_n_free": 0,
            "thrash_win16": 0.0,
            "prunes": 0,
            "dropped": 0,
            "auto_active_steps": 0,
            "decode_path_selected": "",
            "decode_path_fallback_count": 0,
            "decode_path_fallback_reason_topk": {},
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
            "metric_profile": METRIC_PROFILE,
            "prompt_profile": PROMPT_PROFILE,
            "compression_profile": "",
            "auto_trigger_observed": 0,
            "non_trigger_reason": "child_single_case_timeout",
            "error": f"child_single_case_timeout>{child_timeout_s}s: {exc}",
            "pred_strict": "",
            "pred_compat": "",
            "wall_ms": 0.0,
        }
    finally:
        try:
            if os.path.exists(json_path):
                os.unlink(json_path)
        except Exception:
            pass


def _track_summary_rec(mode: str, prefill_track: str, stage: str, target_len: int, depth_runs: List[Dict]) -> Dict:
    success_all = all(r.get("success", 0) == 1 for r in depth_runs)
    em_mean = float(mean(float(r.get("passkey_first_em", 0.0)) for r in depth_runs)) if depth_runs else 0.0
    valid_mean = float(mean(float(r.get("valid_rate", 0.0)) for r in depth_runs)) if depth_runs else 0.0
    em_given_valid = (em_mean / valid_mean) if valid_mean > 1e-12 else 0.0
    tokens_s_mean = float(mean(float(r.get("tokens_s", 0.0)) for r in depth_runs)) if depth_runs else 0.0
    p95_mean = float(mean(float(r.get("p95_ms", 0.0)) for r in depth_runs)) if depth_runs else 0.0
    n_free_min = int(min(int(r.get("decode_min_n_free", 0)) for r in depth_runs)) if depth_runs else 0
    thrash_mean = float(mean(float(r.get("thrash_win16", 0.0)) for r in depth_runs)) if depth_runs else 0.0
    prunes_sum = int(sum(int(r.get("prunes", 0)) for r in depth_runs))
    dropped_sum = int(sum(int(r.get("dropped", 0)) for r in depth_runs))
    auto_steps_sum = int(sum(int(r.get("auto_active_steps", 0)) for r in depth_runs))
    path_counts: Dict[str, int] = {}
    fallback_sum = 0
    fallback_reason_counts: Dict[str, int] = {}
    for r in depth_runs:
        p = str(r.get("decode_path_selected", "") or "")
        if p:
            path_counts[p] = int(path_counts.get(p, 0)) + 1
        fallback_sum += int(r.get("decode_path_fallback_count", 0))
        for k, v in dict(r.get("decode_path_fallback_reason_topk", {}) or {}).items():
            fallback_reason_counts[str(k)] = int(fallback_reason_counts.get(str(k), 0)) + int(v)
    path_selected = ""
    if path_counts:
        path_selected = sorted(path_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[0][0]
    topk = {
        k: int(v)
        for k, v in sorted(fallback_reason_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:3]
    }
    wall_sum = float(sum(float(r.get("wall_ms", 0.0)) for r in depth_runs))
    errors = [str(r.get("error", "")) for r in depth_runs if str(r.get("error", ""))]
    non_trig = ""
    if prunes_sum <= 0:
        reasons = [str(r.get("non_trigger_reason", "")) for r in depth_runs if str(r.get("non_trigger_reason", ""))]
        non_trig = reasons[0] if reasons else ""
    compression_profile = ""
    for r in depth_runs:
        cp = str(r.get("compression_profile", "") or "")
        if cp:
            compression_profile = cp
            break
    return {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "prefill_track": prefill_track,
        "stage": stage,
        "target_len": int(target_len),
        "actual_prompt_tokens": int(mean(float(r.get("actual_prompt_tokens", 0)) for r in depth_runs)) if depth_runs else 0,
        "depth": "mean(0.10,0.50,0.90)",
        "success": 1 if success_all else 0,
        "passkey_first_em": em_mean,
        "valid_rate": valid_mean,
        "em_given_valid": em_given_valid,
        "tokens_s": tokens_s_mean,
        "p95_ms": p95_mean,
        "decode_min_n_free": n_free_min,
        "thrash_win16": thrash_mean,
        "prunes": prunes_sum,
        "dropped": dropped_sum,
        "auto_active_steps": auto_steps_sum,
        "decode_path_selected": str(path_selected),
        "decode_path_fallback_count": int(fallback_sum),
        "decode_path_fallback_reason_topk": topk,
        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        "metric_profile": METRIC_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "compression_profile": compression_profile,
        "auto_trigger_observed": int(auto_steps_sum > 0 and prunes_sum > 0),
        "non_trigger_reason": non_trig,
        "error": "; ".join(errors),
        "pred_strict": "",
        "pred_compat": "",
        "wall_ms": round(wall_sum, 3),
        "depth_runs": [
            {
                "depth": r.get("depth"),
                "success": r.get("success"),
                "passkey_first_em": r.get("passkey_first_em"),
                "valid_rate": r.get("valid_rate"),
                "tokens_s": r.get("tokens_s"),
                "p95_ms": r.get("p95_ms"),
                "error": r.get("error", ""),
            }
            for r in depth_runs
        ],
    }


def _emit_progress(rec: Dict, progress_jsonl: str):
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    with open(progress_jsonl, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_partial(partial_out: str, state: Dict):
    with open(partial_out, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _binary_search_survival(
    engine: Optional[ManagedInferenceEngine],
    mode: str,
    prefill_track: str,
    low_success: int,
    high_fail: int,
    precision: int,
    seed: int,
    add_record: Callable[[Dict], None],
    case_timeout_s: int,
    case_eval_factory: Optional[Callable[[int, float, str], Dict]] = None,
) -> int:
    lo = int(low_success)
    hi = int(high_fail)
    while hi - lo > precision:
        mid = _align_down((lo + hi) // 2, precision)
        if mid <= lo:
            mid = lo + precision
        if mid >= hi:
            break
        if case_eval_factory is not None:
            rec = case_eval_factory(mid, 0.50, "survival_binary")
        else:
            rec = _run_single_case(engine, mode, prefill_track, mid, 0.50, seed, "survival_binary", case_timeout_s)
        add_record(rec)
        if rec["success"] == 1:
            lo = mid
        else:
            hi = mid
    return int(lo)


def _search_survival_limit_anchor_binary(
    engine: Optional[ManagedInferenceEngine],
    mode: str,
    prefill_track: str,
    smoke_floor: int,
    anchor1: int,
    anchor2: int,
    max_tokens: int,
    precision: int,
    seed: int,
    add_record: Callable[[Dict], None],
    case_timeout_s: int,
    case_eval_factory: Optional[Callable[[int, float, str], Dict]] = None,
) -> int:
    smoke_floor = int(min(smoke_floor, max_tokens))
    a1 = int(min(anchor1, max_tokens))
    a2 = int(min(anchor2, max_tokens))
    if a1 < smoke_floor:
        a1 = smoke_floor
    if a2 < a1:
        a2 = a1

    if case_eval_factory is not None:
        rec1 = case_eval_factory(a1, 0.50, "survival_anchor_128k")
    else:
        rec1 = _run_single_case(engine, mode, prefill_track, a1, 0.50, seed, "survival_anchor_128k", case_timeout_s)
    add_record(rec1)
    if rec1["success"] != 1:
        return _binary_search_survival(engine, mode, prefill_track, smoke_floor, a1, precision, seed, add_record, case_timeout_s, case_eval_factory)

    if a2 <= a1:
        return int(a1)

    if case_eval_factory is not None:
        rec2 = case_eval_factory(a2, 0.50, "survival_anchor_256k")
    else:
        rec2 = _run_single_case(engine, mode, prefill_track, a2, 0.50, seed, "survival_anchor_256k", case_timeout_s)
    add_record(rec2)
    if rec2["success"] == 1:
        return int(a2)
    return _binary_search_survival(engine, mode, prefill_track, a1, a2, precision, seed, add_record, case_timeout_s, case_eval_factory)


def _eval_quality_at_length(engine: ManagedInferenceEngine, mode: str, prefill_track: str, target_len: int, depths: Sequence[float], seed: int, stage: str, add_record: Callable[[Dict], None], case_timeout_s: int) -> Tuple[bool, Dict]:
    depth_runs: List[Dict] = []
    for depth in depths:
        depth_runs.append(_run_single_case(engine, mode, prefill_track, int(target_len), float(depth), seed, f"{stage}_depth", case_timeout_s))
    summary = _track_summary_rec(mode, prefill_track, stage, int(target_len), depth_runs)
    add_record(summary)
    pass_flag = bool(summary["success"] == 1 and float(summary["passkey_first_em"]) >= TARGET_EM_THRESHOLD and float(summary["valid_rate"]) >= TARGET_VALID_THRESHOLD)
    stat = {
        "success_all": int(summary["success"]),
        "passkey_first_em_mean": float(summary["passkey_first_em"]),
        "valid_rate_mean": float(summary["valid_rate"]),
        "em_given_valid": float(summary["em_given_valid"]),
        "tokens_s_mean": float(summary["tokens_s"]),
        "p95_ms_mean": float(summary["p95_ms"]),
        "pass": int(pass_flag),
    }
    return pass_flag, stat


def _binary_search_quality(engine: Optional[ManagedInferenceEngine], mode: str, prefill_track: str, low_pass: int, high_fail: int, precision: int, seed: int, add_record: Callable[[Dict], None], best_stat: Dict, case_timeout_s: int, quality_eval_factory: Optional[Callable[[int, str], Tuple[bool, Dict]]] = None) -> Tuple[int, Dict]:
    lo = int(low_pass)
    hi = int(high_fail)
    stat_best = dict(best_stat)
    while hi - lo > precision:
        mid = _align_down((lo + hi) // 2, precision)
        if mid <= lo:
            mid = lo + precision
        if mid >= hi:
            break
        if quality_eval_factory is not None:
            ok, stat = quality_eval_factory(mid, "quality_binary")
        else:
            ok, stat = _eval_quality_at_length(engine, mode, prefill_track, mid, DEPTHS, seed, "quality_binary", add_record, case_timeout_s)
        if ok:
            lo = mid
            stat_best = stat
        else:
            hi = mid
    return int(lo), stat_best


def _run_quality_eval_subprocess(
    mode: str,
    prefill_track: str,
    target_len: int,
    depths: Sequence[float],
    seed: int,
    stage: str,
    gpu_mem_frac: float,
    add_record: Callable[[Dict], None],
    case_timeout_s: int,
) -> Tuple[bool, Dict]:
    fd, json_path = tempfile.mkstemp(prefix="niah_quality_eval_", suffix=".json")
    os.close(fd)
    try:
        os.unlink(json_path)
    except FileNotFoundError:
        pass
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-quality-eval-json-out",
        json_path,
        "--child-mode",
        str(mode),
        "--child-prefill-track",
        str(prefill_track),
        "--child-target-len",
        str(int(target_len)),
        "--child-depths",
        ",".join(f"{float(x):.2f}" for x in depths),
        "--child-stage",
        str(stage),
        "--model-name",
        str(COMMON_BASE["model_name"]),
        "--max-new-tokens",
        str(int(MAX_NEW_TOKENS)),
        "--target-em-threshold",
        str(float(TARGET_EM_THRESHOLD)),
        "--target-valid-threshold",
        str(float(TARGET_VALID_THRESHOLD)),
        "--gpu-mem-frac-max",
        str(float(gpu_mem_frac)),
        "--gpu-mem-frac-fallback-step",
        "0",
        "--gpu-mem-frac-fallback-tries",
        "1",
        "--compress-retain-ratio",
        str(float(PREFILL_COMPRESS_ARGS.get("retain_ratio", 1.0))),
        "--seed",
        str(int(seed)),
        "--case-timeout-s",
        str(int(case_timeout_s)),
        "--sink-len",
        str(int(COMMON_BASE.get("sink_len", 0))),
        "--obs-len",
        str(int(COMMON_BASE.get("obs_len", 0))),
        "--decode-window-sink-len",
        str(int(COMMON_BASE.get("decode_window_sink_len", 0))),
        "--decode-window-recent-len",
        str(int(MODE_MAIN_AUTO_ARGS.get("decode_window_recent_len", 0))),
    ]
    if EXTRA_ENGINE_ARGS:
        cmd.extend(
            [
                "--engine-overrides-json",
                json.dumps(EXTRA_ENGINE_ARGS, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    if str(NEEDLE_DEBUG_DIR).strip():
        cmd.extend(["--needle-debug-dir", str(NEEDLE_DEBUG_DIR)])
    try:
        quality_timeout_s = max(
            300,
            int(case_timeout_s) * max(1, len(depths)) + QUALITY_EVAL_SUBPROCESS_GRACE_S,
        )
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=quality_timeout_s,
            check=False,
        )
        payload = None
        payload_error = ""
        if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                payload_error = f"child_payload_read_failed: {exc}"
        if (proc.returncode != 0) or (payload is None):
            rec = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "prefill_track": prefill_track,
                "stage": stage,
                "target_len": int(target_len),
                "actual_prompt_tokens": 0,
                "depth": f"mean({','.join(f'{float(x):.2f}' for x in depths)})",
                "success": 0,
                "passkey_first_em": 0.0,
                "valid_rate": 0.0,
                "em_given_valid": 0.0,
                "tokens_s": 0.0,
                "p95_ms": 0.0,
                "decode_min_n_free": 0,
                "thrash_win16": 0.0,
                "prunes": 0,
                "dropped": 0,
                "auto_active_steps": 0,
                "decode_path_selected": "",
                "decode_path_fallback_count": 0,
                "decode_path_fallback_reason_topk": {},
                "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
                "metric_profile": METRIC_PROFILE,
                "prompt_profile": PROMPT_PROFILE,
                "compression_profile": "",
                "auto_trigger_observed": 0,
                "non_trigger_reason": "child_quality_eval_failed",
                "error": (payload_error or proc.stderr or proc.stdout or f"child_returncode={proc.returncode}")[:800],
                "pred_strict": "",
                "pred_compat": "",
                "wall_ms": 0.0,
            }
            add_record(rec)
            return False, {
                "success_all": 0,
                "passkey_first_em_mean": 0.0,
                "valid_rate_mean": 0.0,
                "em_given_valid": 0.0,
                "tokens_s_mean": 0.0,
                "p95_ms_mean": 0.0,
                "pass": 0,
            }
        payload = payload or {}
        for rec in payload.get("records", []) or []:
            add_record(rec)
        stat = payload.get("stat") or {
            "success_all": 0,
            "passkey_first_em_mean": 0.0,
            "valid_rate_mean": 0.0,
            "em_given_valid": 0.0,
            "tokens_s_mean": 0.0,
            "p95_ms_mean": 0.0,
            "pass": 0,
        }
        return bool(payload.get("ok")), stat
    except subprocess.TimeoutExpired as exc:
        rec = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "prefill_track": prefill_track,
            "stage": stage,
            "target_len": int(target_len),
            "actual_prompt_tokens": 0,
            "depth": f"mean({','.join(f'{float(x):.2f}' for x in depths)})",
            "success": 0,
            "passkey_first_em": 0.0,
            "valid_rate": 0.0,
            "em_given_valid": 0.0,
            "tokens_s": 0.0,
            "p95_ms": 0.0,
            "decode_min_n_free": 0,
            "thrash_win16": 0.0,
            "prunes": 0,
            "dropped": 0,
            "auto_active_steps": 0,
            "decode_path_selected": "",
            "decode_path_fallback_count": 0,
            "decode_path_fallback_reason_topk": {},
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
            "metric_profile": METRIC_PROFILE,
            "prompt_profile": PROMPT_PROFILE,
            "compression_profile": "",
            "auto_trigger_observed": 0,
            "non_trigger_reason": "child_quality_eval_timeout",
            "error": f"child_quality_eval_timeout>{quality_timeout_s}s: {exc}",
            "pred_strict": "",
            "pred_compat": "",
            "wall_ms": 0.0,
        }
        add_record(rec)
        return False, {
            "success_all": 0,
            "passkey_first_em_mean": 0.0,
            "valid_rate_mean": 0.0,
            "em_given_valid": 0.0,
            "tokens_s_mean": 0.0,
            "p95_ms_mean": 0.0,
            "pass": 0,
        }
    finally:
        try:
            if os.path.exists(json_path):
                os.unlink(json_path)
        except Exception:
            pass
        try:
            if os.path.exists(json_path):
                os.remove(json_path)
        except Exception:
            pass


def _search_quality_limit_anchor_binary(engine: Optional[ManagedInferenceEngine], mode: str, prefill_track: str, survival_limit: int, smoke_floor: int, anchor1: int, anchor2: int, precision: int, seed: int, add_record: Callable[[Dict], None], case_timeout_s: int, quality_eval_factory: Optional[Callable[[int, str], Tuple[bool, Dict]]] = None, low_anchor_eval_factory: Optional[Callable[[int], Tuple[bool, Dict]]] = None) -> Tuple[int, Optional[Dict]]:
    if int(survival_limit) < int(smoke_floor):
        return 0, None

    low = int(max(smoke_floor, MIN_TOKENS))
    low = _align_up(low, precision)
    a1 = int(min(anchor1, survival_limit))
    a2 = int(min(anchor2, survival_limit))
    if a1 < low:
        a1 = low
    if a2 < a1:
        a2 = a1

    if quality_eval_factory is not None:
        ok1, stat1 = quality_eval_factory(a1, "quality_anchor_128k")
    else:
        ok1, stat1 = _eval_quality_at_length(engine, mode, prefill_track, a1, DEPTHS, seed, "quality_anchor_128k", add_record, case_timeout_s)
    if ok1:
        if a2 <= a1:
            return a1, stat1
        if quality_eval_factory is not None:
            ok2, stat2 = quality_eval_factory(a2, "quality_anchor_256k")
        else:
            ok2, stat2 = _eval_quality_at_length(engine, mode, prefill_track, a2, DEPTHS, seed, "quality_anchor_256k", add_record, case_timeout_s)
        if ok2:
            return a2, stat2
        return _binary_search_quality(engine, mode, prefill_track, a1, a2, precision, seed, add_record, stat1, case_timeout_s, quality_eval_factory)

    if low >= a1:
        return 0, stat1
    # Evaluate the fallback low anchor in a child process. This isolates the
    # retry from timeout-polluted runtime state and avoids in-process model
    # reload OOM when we need a fresh engine.
    if low_anchor_eval_factory is not None:
        ok_low, stat_low = low_anchor_eval_factory(low)
    else:
        if quality_eval_factory is not None:
            ok_low, stat_low = quality_eval_factory(low, "quality_anchor_low")
        else:
            ok_low, stat_low = _eval_quality_at_length(engine, mode, prefill_track, low, DEPTHS, seed, "quality_anchor_low", add_record, case_timeout_s)
    if not ok_low:
        return 0, stat_low
    return _binary_search_quality(engine, mode, prefill_track, low, a1, precision, seed, add_record, stat_low, case_timeout_s, quality_eval_factory)


def _run_track(mode: str, prefill_track: str, seed: int, gpu_mem_frac_max: float, smoke_lengths: Sequence[int], anchor1: int, anchor2: int, min_tokens: int, max_tokens: int, precision: int, add_record: Callable[[Dict], None], case_timeout_s: int, compress_mem_frac_safety_margin: float = COMPRESS_MEM_FRAC_SAFETY_MARGIN) -> Dict:
    records: List[Dict] = []
    fallback_trace: List[Dict] = []
    chosen_mem_frac: Optional[float] = None
    chosen_try_idx: Optional[int] = None
    engine = None
    smoke_floor = int(min(max_tokens, max(min_tokens, max(int(x) for x in smoke_lengths))))

    def _add(rec: Dict):
        records.append(rec)
        add_record(rec)

    for i in range(GPU_MEM_FRAC_FALLBACK_TRIES):
        mem_frac = max(0.10, float(gpu_mem_frac_max) - i * GPU_MEM_FRAC_FALLBACK_STEP)
        fallback_trace.append({"try_idx": i + 1, "gpu_mem_frac": mem_frac})
        try:
            engine = _build_engine(mem_frac, mode, prefill_track)
        except Exception as exc:
            fallback_trace[-1]["build_error"] = str(exc)
            engine = None
            continue

        smoke_ok = True
        for slen in smoke_lengths:
            rec = _run_single_case(engine, mode, prefill_track, int(slen), 0.50, seed, "smoke", case_timeout_s)
            _add(rec)
            if rec["success"] != 1:
                smoke_ok = False
                fallback_trace[-1]["smoke_fail_at"] = int(slen)
                break
        if smoke_ok:
            chosen_mem_frac = mem_frac
            chosen_try_idx = int(i)
            break

        del engine
        engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if engine is None or chosen_mem_frac is None:
        return {
            "mode": mode,
            "prefill_track": prefill_track,
            "gpu_mem_frac": None,
            "smoke_ok": 0,
            "l_survival_max": 0,
            "l_quality_max": 0,
            "quality_stat_at_limit": None,
            "records": records,
            "fallback_trace": fallback_trace,
            "error": "all_gpu_mem_frac_candidates_failed",
        }

    # Use the selected mem-frac directly. We do not silently apply an extra
    # downward margin after smoke success, because paper experiments should
    # report the actual tested mem-frac rather than a hidden effective value.

    cfg_limit = int(getattr(getattr(getattr(engine, "model", None), "config", None), "max_position_embeddings", 0) or 0)
    tok_limit = int(getattr(getattr(engine, "tokenizer", None), "model_max_length", 0) or 0)
    # Some tokenizers use huge sentinel values for "unbounded".
    if tok_limit <= 0 or tok_limit > 2_000_000:
        tok_limit = 0
    model_limit = max(cfg_limit, tok_limit)
    if model_limit > 0:
        model_limit = _align_down(model_limit, precision)
    max_tokens_eff = int(max_tokens)
    anchor1_eff = int(anchor1)
    anchor2_eff = int(anchor2)
    smoke_floor_eff = int(smoke_floor)
    if model_limit > 0:
        max_tokens_eff = int(min(max_tokens_eff, model_limit))
        anchor1_eff = int(min(anchor1_eff, max_tokens_eff))
        anchor2_eff = int(min(anchor2_eff, max_tokens_eff))
        smoke_floor_eff = int(min(smoke_floor_eff, max_tokens_eff))
    if fallback_trace:
        fallback_trace[-1]["model_context_limit"] = int(model_limit)
        fallback_trace[-1]["effective_max_tokens"] = int(max_tokens_eff)
        fallback_trace[-1]["effective_anchor_len_1"] = int(anchor1_eff)
        fallback_trace[-1]["effective_anchor_len_2"] = int(anchor2_eff)

    del engine
    engine = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    survival = _search_survival_limit_anchor_binary(
        None,
        mode,
        prefill_track,
        smoke_floor_eff,
        anchor1_eff,
        anchor2_eff,
        max_tokens_eff,
        precision,
        seed,
        _add,
        case_timeout_s,
        case_eval_factory=lambda target_len, depth, stage: _run_single_case_subprocess(
            mode,
            prefill_track,
            int(target_len),
            float(depth),
            seed,
            str(stage),
            float(chosen_mem_frac),
            case_timeout_s,
        ),
    )

    quality_eval_factory = lambda target_len, stage: _run_quality_eval_subprocess(
        mode,
        prefill_track,
        int(target_len),
        DEPTHS,
        seed,
        str(stage),
        float(chosen_mem_frac),
        _add,
        case_timeout_s,
    )

    quality, quality_stat = _search_quality_limit_anchor_binary(
        None,
        mode,
        prefill_track,
        survival,
        smoke_floor_eff,
        anchor1_eff,
        anchor2_eff,
        precision,
        seed,
        _add,
        case_timeout_s,
        quality_eval_factory=quality_eval_factory,
        low_anchor_eval_factory=(
            (lambda target_len: _run_quality_eval_subprocess(mode, prefill_track, target_len, "quality_anchor_low"))
            if quality_eval_factory is not None
            else None
        ),
    )

    return {
        "mode": mode,
        "prefill_track": prefill_track,
        "gpu_mem_frac": float(chosen_mem_frac),
        "smoke_ok": 1,
        "l_survival_max": int(survival),
        "l_quality_max": int(quality),
        "quality_stat_at_limit": quality_stat,
        "records": records,
        "fallback_trace": fallback_trace,
        "error": "",
    }


def _build_table_c_row(track_res: Dict) -> Dict:
    lq = int(track_res.get("l_quality_max", 0))
    recs = [r for r in track_res.get("records", []) if int(r.get("target_len", -1)) == lq and str(r.get("stage", "")).startswith("quality")]
    if not recs:
        return {
            "mode": track_res.get("mode"),
            "prefill_track": track_res.get("prefill_track"),
            "l_quality_max": lq,
            "tokens_s_at_l_quality": 0.0,
            "p95_ms_at_l_quality": 0.0,
            "em_mean_at_l_quality": 0.0,
            "valid_mean_at_l_quality": 0.0,
            "em_given_valid_at_l_quality": 0.0,
        }
    em_mean = float(mean(r.get("passkey_first_em", 0.0) for r in recs))
    valid_mean = float(mean(r.get("valid_rate", 0.0) for r in recs))
    return {
        "mode": track_res.get("mode"),
        "prefill_track": track_res.get("prefill_track"),
        "l_quality_max": lq,
        "tokens_s_at_l_quality": float(mean(r.get("tokens_s", 0.0) for r in recs)),
        "p95_ms_at_l_quality": float(mean(r.get("p95_ms", 0.0) for r in recs)),
        "em_mean_at_l_quality": em_mean,
        "valid_mean_at_l_quality": valid_mean,
        "em_given_valid_at_l_quality": (em_mean / valid_mean) if valid_mean > 1e-12 else 0.0,
    }


def _parse_int_list(s: str) -> List[int]:
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _parse_tracks(s: str) -> List[Tuple[str, str]]:
    out = []
    for x in str(s).split(","):
        k = x.strip()
        if not k:
            continue
        if k not in TRACK_ALIAS:
            raise ValueError(f"unknown track alias: {k}")
        out.append(TRACK_ALIAS[k])
    if not out:
        raise ValueError("no tracks selected")
    return out


def _build_payload(meta: Dict, track_results: List[Dict]) -> Dict:
    table_a = []
    table_b = []
    table_c = []
    for r in track_results:
        table_a.append(
            {
                "mode": r.get("mode"),
                "prefill_track": r.get("prefill_track"),
                "gpu_mem_frac": r.get("gpu_mem_frac"),
                "l_survival_max": r.get("l_survival_max", 0),
                "metric_profile": METRIC_PROFILE,
                "prompt_profile": PROMPT_PROFILE,
            }
        )
        table_b.append(
            {
                "mode": r.get("mode"),
                "prefill_track": r.get("prefill_track"),
                "gpu_mem_frac": r.get("gpu_mem_frac"),
                "l_quality_max_em075": r.get("l_quality_max", 0),
                "quality_stat_at_limit": r.get("quality_stat_at_limit"),
                "metric_profile": METRIC_PROFILE,
                "prompt_profile": PROMPT_PROFILE,
            }
        )
        table_c.append(_build_table_c_row(r))
    return {
        "meta": meta,
        "table_a_l_survival_max": table_a,
        "table_b_l_quality_max_em075": table_b,
        "table_c_perf_at_l_quality_max": table_c,
        "tracks": track_results,
    }


def _run_child_quality_eval(args) -> int:
    depths = [float(x) for x in str(args.child_depths or "").split(",") if str(x).strip()]
    if not depths:
        raise ValueError("child-depths must not be empty")
    records: List[Dict] = []

    def _add(rec: Dict):
        records.append(rec)

    engine = _build_engine(float(args.gpu_mem_frac_max), str(args.child_mode), str(args.child_prefill_track))
    try:
        ok, stat = _eval_quality_at_length(
            engine,
            str(args.child_mode),
            str(args.child_prefill_track),
            int(args.child_target_len),
            depths,
            int(args.seed),
            str(args.child_stage or "quality_anchor_low"),
            _add,
            int(args.case_timeout_s),
        )
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "ok": bool(ok),
        "stat": stat,
        "records": records,
    }
    with open(args.child_quality_eval_json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return 0


def _run_child_single_case(args) -> int:
    engine = _build_engine(float(args.gpu_mem_frac_max), str(args.child_mode), str(args.child_prefill_track))
    try:
        rec = _run_single_case(
            engine,
            str(args.child_mode),
            str(args.child_prefill_track),
            int(args.child_target_len),
            float(args.child_depth),
            int(args.seed),
            str(args.child_stage or "single_case"),
            int(args.case_timeout_s),
        )
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(args.child_single_case_json_out, "w", encoding="utf-8") as f:
        json.dump({"rec": rec}, f, ensure_ascii=False, indent=2)
    return 0


def main():
    global TARGET_EM_THRESHOLD, TARGET_VALID_THRESHOLD, MAX_NEW_TOKENS, DEPTHS, EXTRA_ENGINE_ARGS
    global GPU_MEM_FRAC_FALLBACK_STEP, GPU_MEM_FRAC_FALLBACK_TRIES
    global NEEDLE_DEBUG_DIR
    parser = argparse.ArgumentParser(description="NIAH single-sequence extreme-length benchmark")
    parser.add_argument("--model-name", type=str, default=str(COMMON_BASE.get("model_name", LOCAL_MODEL_PATH)))
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--depths", type=str, default="0.10,0.50,0.90")
    parser.add_argument("--target-em-threshold", type=float, default=TARGET_EM_THRESHOLD)
    parser.add_argument("--target-valid-threshold", type=float, default=TARGET_VALID_THRESHOLD)
    parser.add_argument("--gpu-mem-frac-fallback-step", type=float, default=GPU_MEM_FRAC_FALLBACK_STEP)
    parser.add_argument("--gpu-mem-frac-fallback-tries", type=int, default=GPU_MEM_FRAC_FALLBACK_TRIES)
    parser.add_argument(
        "--engine-overrides-json",
        type=str,
        default="",
        help="JSON object merged into engine args after mode/prefill defaults; used for paper ablations.",
    )
    parser.add_argument("--gpu-mem-frac-max", type=float, default=GPU_MEM_FRAC_MAX)
    parser.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--precision-tokens", type=int, default=PRECISION_TOKENS)
    parser.add_argument("--smoke-lengths", type=str, default="16000")
    parser.add_argument("--anchor-len-1", type=int, default=ANCHOR_LEN_1)
    parser.add_argument("--anchor-len-2", type=int, default=ANCHOR_LEN_2)
    parser.add_argument("--tracks", type=str, default="off_compress,off_raw,p2_only_compress,main_auto_compress", help="comma-separated aliases: off_compress,off_raw,p2_only_compress,main_auto_compress,main_auto_raw")
    parser.add_argument(
        "--compress-profile",
        type=str,
        choices=sorted(COMPRESS_PROFILES.keys()),
        default=DEFAULT_COMPRESS_PROFILE,
        help="explicit off_compress profile; avoids implicit retain_ratio defaults in paper runs",
    )
    parser.add_argument(
        "--compress-retain-ratio",
        type=float,
        default=None,
        help="optional retain_ratio override for prefill compress tracks; if set, it overrides --compress-profile",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--compress-mem-frac-safety-margin",
        type=float,
        default=COMPRESS_MEM_FRAC_SAFETY_MARGIN,
        help="extra mem-frac margin applied to compress tracks after smoke",
    )
    parser.add_argument("--case-timeout-s", type=int, default=CASE_TIMEOUT_S, help="per target_len timeout in seconds; <=0 disables timeout")
    parser.add_argument("--out", type=str, default="benchmark_niah_single_limit_results.json")
    parser.add_argument("--partial-out", type=str, default="benchmark_niah_single_limit_results.partial.json")
    parser.add_argument("--progress-jsonl", type=str, default="benchmark_niah_single_limit_progress.jsonl")
    parser.add_argument("--sink-len", type=int, default=int(COMMON_BASE.get("sink_len", 64)))
    parser.add_argument("--obs-len", type=int, default=int(COMMON_BASE.get("obs_len", 64)))
    parser.add_argument("--needle-debug-dir", type=str, default="")
    parser.add_argument(
        "--decode-window-sink-len",
        type=int,
        default=int(COMMON_BASE.get("decode_window_sink_len", 64)),
    )
    parser.add_argument(
        "--decode-window-recent-len",
        type=int,
        default=int(MODE_MAIN_AUTO_ARGS.get("decode_window_recent_len", 256)),
    )
    parser.add_argument("--child-quality-eval-json-out", type=str, default="")
    parser.add_argument("--child-single-case-json-out", type=str, default="")
    parser.add_argument("--child-mode", type=str, default="")
    parser.add_argument("--child-prefill-track", type=str, default="")
    parser.add_argument("--child-target-len", type=int, default=0)
    parser.add_argument("--child-depth", type=float, default=0.5)
    parser.add_argument("--child-depths", type=str, default="")
    parser.add_argument("--child-stage", type=str, default="quality_anchor_low")
    args = parser.parse_args()

    smoke_lengths = _parse_int_list(args.smoke_lengths)
    selected_tracks = _parse_tracks(args.tracks)
    DEPTHS = [float(x) for x in str(args.depths).split(",") if str(x).strip()]
    if not DEPTHS:
        raise ValueError("depths must not be empty")
    MAX_NEW_TOKENS = int(max(1, args.max_new_tokens))
    TARGET_EM_THRESHOLD = float(max(0.0, min(1.0, args.target_em_threshold)))
    TARGET_VALID_THRESHOLD = float(max(0.0, min(1.0, args.target_valid_threshold)))
    GPU_MEM_FRAC_FALLBACK_STEP = float(max(0.0, args.gpu_mem_frac_fallback_step))
    GPU_MEM_FRAC_FALLBACK_TRIES = int(max(1, args.gpu_mem_frac_fallback_tries))
    NEEDLE_DEBUG_DIR = str(args.needle_debug_dir or "")
    if str(args.engine_overrides_json).strip():
        parsed = json.loads(str(args.engine_overrides_json))
        if not isinstance(parsed, dict):
            raise ValueError("engine-overrides-json must decode to a JSON object")
        EXTRA_ENGINE_ARGS = dict(parsed)
    else:
        EXTRA_ENGINE_ARGS = {}
    COMMON_BASE["model_name"] = str(args.model_name)
    COMMON_BASE["max_new_tokens"] = int(MAX_NEW_TOKENS)
    selected_compress_profile = str(args.compress_profile)
    PREFILL_COMPRESS_ARGS.clear()
    PREFILL_COMPRESS_ARGS.update(dict(COMPRESS_PROFILES[selected_compress_profile]))
    if args.compress_retain_ratio is not None:
        PREFILL_COMPRESS_ARGS["retain_ratio"] = float(max(0.0, min(1.0, args.compress_retain_ratio)))
        selected_compress_profile = f"custom:{float(PREFILL_COMPRESS_ARGS['retain_ratio']):.3f}"
    anchor1 = _align_down(int(args.anchor_len_1), int(args.precision_tokens))
    anchor2 = _align_down(int(args.anchor_len_2), int(args.precision_tokens))
    if anchor1 <= 0 or anchor2 <= 0:
        raise ValueError("anchor lengths must be positive")
    if anchor2 < anchor1:
        anchor2 = anchor1
    COMMON_BASE["sink_len"] = int(max(0, args.sink_len))
    COMMON_BASE["obs_len"] = int(max(0, args.obs_len))
    COMMON_BASE["decode_window_sink_len"] = int(max(0, args.decode_window_sink_len))
    MODE_MAIN_AUTO_ARGS["decode_window_recent_len"] = int(max(0, args.decode_window_recent_len))

    if str(args.child_quality_eval_json_out).strip():
        raise SystemExit(_run_child_quality_eval(args))
    if str(args.child_single_case_json_out).strip():
        raise SystemExit(_run_child_single_case(args))

    with open(args.progress_jsonl, "w", encoding="utf-8") as f:
        f.write("")

    meta = {
        "task": "niah_single_sequence_extreme_length",
        "model_name": str(COMMON_BASE["model_name"]),
        "max_new_tokens": int(MAX_NEW_TOKENS),
        "depths": [float(x) for x in DEPTHS],
        "target_em_threshold": TARGET_EM_THRESHOLD,
        "target_valid_threshold": TARGET_VALID_THRESHOLD,
        "metric_profile": METRIC_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "strict_definition": STRICT_DEFINITION,
        "compat_definition": "first 6-digit match anywhere in output",
        "gpu_mem_frac_max": float(args.gpu_mem_frac_max),
        "gpu_mem_frac_fallback_step": GPU_MEM_FRAC_FALLBACK_STEP,
        "gpu_mem_frac_fallback_tries": GPU_MEM_FRAC_FALLBACK_TRIES,
        "memory_policy_note": "KV pool uses available_vram_after_model_and_10pct_reserve * gpu_mem_frac; gpu_mem_frac_max=0.78 keeps 22% of available_vram for transient attention/workspace.",
        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        "alloc_conf_value": str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
        "decode_path_mode": "auto",
        "min_tokens": int(args.min_tokens),
        "max_tokens": int(args.max_tokens),
        "precision_tokens": int(args.precision_tokens),
        "smoke_lengths": smoke_lengths,
        "anchor_len_1": int(anchor1),
        "anchor_len_2": int(anchor2),
        "seed": int(args.seed),
        "case_timeout_s": int(args.case_timeout_s),
        "tracks_selected": [{"mode": m, "prefill_track": p} for (m, p) in selected_tracks],
        "compress_profile_name": selected_compress_profile,
        "compress_retain_ratio": float(PREFILL_COMPRESS_ARGS["retain_ratio"]),
        "compress_mem_frac_safety_margin": float(max(0.0, args.compress_mem_frac_safety_margin)),
        "sink_len": int(COMMON_BASE["sink_len"]),
        "obs_len": int(COMMON_BASE["obs_len"]),
        "decode_window_sink_len": int(COMMON_BASE["decode_window_sink_len"]),
        "decode_window_recent_len": int(MODE_MAIN_AUTO_ARGS["decode_window_recent_len"]),
        "compression_profile": str(
            f"sink={int(COMMON_BASE['sink_len'])};"
            f"obs={int(COMMON_BASE['obs_len'])};"
            f"retain={float(PREFILL_COMPRESS_ARGS['retain_ratio']):.3f};"
            f"decode_sink={int(COMMON_BASE['decode_window_sink_len'])};"
            f"decode_recent={int(MODE_MAIN_AUTO_ARGS['decode_window_recent_len'])}"
        ),
    }

    track_results: List[Dict] = []

    def save_partial():
        payload = _build_payload(meta, track_results)
        _write_partial(args.partial_out, payload)

    for mode, prefill_track in selected_tracks:
        print(f"\\n===== TRACK: mode={mode}, prefill_track={prefill_track} =====", flush=True)
        t0 = time.perf_counter()
        track_obj: Dict = {
            "mode": mode,
            "prefill_track": prefill_track,
            "gpu_mem_frac": None,
            "smoke_ok": 0,
            "l_survival_max": 0,
            "l_quality_max": 0,
            "quality_stat_at_limit": None,
            "records": [],
            "fallback_trace": [],
            "error": "",
            "track_wall_s": 0.0,
        }
        track_results.append(track_obj)

        def _add_record(rec: Dict):
            track_obj["records"].append(rec)
            _emit_progress(rec, args.progress_jsonl)
            save_partial()

        try:
            res = _run_track(
                mode,
                prefill_track,
                args.seed,
                args.gpu_mem_frac_max,
                smoke_lengths,
                anchor1,
                anchor2,
                args.min_tokens,
                args.max_tokens,
                args.precision_tokens,
                _add_record,
                int(args.case_timeout_s),
                float(max(0.0, args.compress_mem_frac_safety_margin)),
            )
        except Exception as exc:
            track_obj["error"] = f"track_exception: {exc}"
            track_obj["traceback"] = traceback.format_exc()
            track_obj["track_wall_s"] = round(time.perf_counter() - t0, 3)
            save_partial()
            print(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode,
                "prefill_track": prefill_track,
                "stage": "track_exception",
                "error": track_obj["error"],
            }, ensure_ascii=False), flush=True)
            continue
        for k, v in res.items():
            if k == "records":
                continue
            track_obj[k] = v
        track_obj["track_wall_s"] = round(time.perf_counter() - t0, 3)
        save_partial()
        print(json.dumps({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "prefill_track": prefill_track,
            "stage": "track_done",
            "l_survival_max": track_obj.get("l_survival_max"),
            "l_quality_max": track_obj.get("l_quality_max"),
            "gpu_mem_frac": track_obj.get("gpu_mem_frac"),
            "error": track_obj.get("error", ""),
            "track_wall_s": track_obj.get("track_wall_s", 0.0),
        }, ensure_ascii=False), flush=True)

    final_payload = _build_payload(meta, track_results)

    print("\\n===== TABLE A: L_survival_max =====")
    print("mode,prefill_track,gpu_mem_frac,l_survival_max")
    for row in final_payload["table_a_l_survival_max"]:
        print(f"{row['mode']},{row['prefill_track']},{row['gpu_mem_frac']},{row['l_survival_max']}")

    print("\\n===== TABLE B: L_quality_max@EM>=0.75 =====")
    print("mode,prefill_track,gpu_mem_frac,l_quality_max_em075,em_mean,valid_mean,em_given_valid")
    for row in final_payload["table_b_l_quality_max_em075"]:
        qs = row.get("quality_stat_at_limit") or {}
        print(f"{row['mode']},{row['prefill_track']},{row['gpu_mem_frac']},{row['l_quality_max_em075']},{float(qs.get('passkey_first_em_mean', 0.0)):.4f},{float(qs.get('valid_rate_mean', 0.0)):.4f},{float(qs.get('em_given_valid', 0.0)):.4f}")

    print("\\n===== TABLE C: Perf at L_quality_max (optional) =====")
    print("mode,prefill_track,l_quality_max,tokens_s,p95_ms,em_mean,valid_mean,em_given_valid")
    for row in final_payload["table_c_perf_at_l_quality_max"]:
        print(f"{row['mode']},{row['prefill_track']},{row['l_quality_max']},{row['tokens_s_at_l_quality']:.4f},{row['p95_ms_at_l_quality']:.4f},{row['em_mean_at_l_quality']:.4f},{row['valid_mean_at_l_quality']:.4f},{row['em_given_valid_at_l_quality']:.4f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, ensure_ascii=False, indent=2)
    print(f"\\nSaved: {args.out}", flush=True)


if __name__ == "__main__":
    main()

