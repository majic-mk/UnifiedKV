import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ALLOC_CONF_ENABLED = (
    str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")).strip() == "expandable_segments:True"
)

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(CONFIGS_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIGS_DIR))

from engine import ManagedInferenceEngine
from strategy_groups import COMMON_BASE, GROUP_ARGS, LOCAL_MODEL_PATH


DEFAULT_METHODS = ["off_compress", "p2_only_compress"]
DEFAULT_DEPTHS = [0.10, 0.50, 0.90]
DEFAULT_CONCURRENCY = [1, 4, 8, 16]
DEFAULT_KEYS_PER_DEPTH = 20
DEFAULT_INPUT_LEN = 32000
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_GPU_MEM_FRACS = {
    "off_compress": 0.68,
    "p2_only_compress": 0.52,
}
DEFAULT_GPU_MEM_FRAC_FALLBACK_STEP = 0.04
DEFAULT_GPU_MEM_FRAC_FALLBACK_TRIES = 3
DEFAULT_SEED = 20260407


def parse_csv_ints(s: str) -> List[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return vals


def parse_csv_floats(s: str) -> List[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def parse_methods(s: str) -> List[str]:
    vals: List[str] = []
    for x in str(s).split(","):
        method = str(x).strip()
        if not method:
            continue
        if method == "vllm":
            raise ValueError("method 'vllm' is not implemented in this runner; use a dedicated vLLM runner.")
        if method not in GROUP_ARGS:
            raise ValueError(f"unknown method: {method}")
        vals.append(method)
    if not vals:
        raise ValueError("empty methods list")
    return vals


def parse_method_frac_map(s: str) -> Dict[str, float]:
    text = str(s or "").strip()
    if not text:
        return dict(DEFAULT_GPU_MEM_FRACS)
    out: Dict[str, float] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        method, frac = chunk.split(":", 1)
        out[str(method).strip()] = float(frac.strip())
    return out


def percentile(vals: Sequence[float], q: float) -> float:
    seq = sorted(float(v) for v in vals if v is not None)
    if not seq:
        return 0.0
    if len(seq) == 1:
        return float(seq[0])
    pos = max(0.0, min(1.0, float(q))) * (len(seq) - 1)
    lo = int(pos)
    hi = min(len(seq) - 1, lo + 1)
    frac = pos - lo
    return float(seq[lo] * (1.0 - frac) + seq[hi] * frac)


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


def build_engine(
    model_name: str,
    method: str,
    gpu_mem_frac: float,
    max_new_tokens: int,
    chunk_size: int = 0,
    prefill_batch_size: int = 0,
) -> ManagedInferenceEngine:
    args = dict(COMMON_BASE)
    args["model_name"] = str(model_name)
    args["gpu_mem_frac"] = float(gpu_mem_frac)
    args["max_new_tokens"] = int(max_new_tokens)
    if int(chunk_size) > 0:
        args["chunk_size"] = int(chunk_size)
    if int(prefill_batch_size) > 0:
        args["prefill_batch_size"] = int(prefill_batch_size)
    args.update(GROUP_ARGS[str(method)])
    return ManagedInferenceEngine(**args)


def _first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def first_6digit_compat(text: str) -> str:
    m = re.search(r"\b(\d{6})\b", str(text))
    return m.group(1) if m else ""


def first_6digit_strict(text: str) -> str:
    first = _first_nonempty_line(text).strip("`").strip()
    m = re.fullmatch(r"(?:answer\s*[:：]\s*)?(\d{6})", first or "", flags=re.IGNORECASE)
    return m.group(1) if m else ""


def build_passkey_prompt(tokenizer, target_tokens: int, depth: float, answer: str, seq_label: str) -> Tuple[str, int]:
    intro = (
        f"[{seq_label}] You are doing a long-context retrieval test.\n"
        "Find PASSKEY_RECORD in the context and return ONLY the 6-digit number.\n"
    )
    needle = f"\nPASSKEY_RECORD: passkey::{answer}::end\n"
    question = (
        "\nQuestion: What is the passkey in PASSKEY_RECORD?\n"
        "Instruction: Output exactly one line with only the 6-digit number.\n"
    )
    filler = "Irrelevant context about serving systems, KV blocks, scheduling, batching, and memory pressure.\n"

    intro_ids = tokenizer(intro, add_special_tokens=False).input_ids
    needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids or [32]

    fixed = len(intro_ids) + len(needle_ids) + len(question_ids)
    remaining = max(0, int(target_tokens) - fixed)
    left = int(round(remaining * float(depth)))
    right = max(0, remaining - left)

    def _fill_to(n: int) -> List[int]:
        reps = (n + len(filler_ids) - 1) // len(filler_ids)
        return (filler_ids * reps)[:n]

    ids = intro_ids + _fill_to(left) + needle_ids + _fill_to(right) + question_ids
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
    actual = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return prompt, int(actual)


def build_cases(
    tokenizer,
    input_len: int,
    depths: Sequence[float],
    keys_per_depth: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed) ^ (int(input_len) * 1009) ^ (len(depths) * 313))
    cases: List[Dict[str, Any]] = []
    for depth in depths:
        for sample_idx in range(int(keys_per_depth)):
            answer = f"{rng.randint(100000, 999999)}"
            seq_label = f"d{depth:.2f}_i{sample_idx}"
            prompt, actual_tokens = build_passkey_prompt(tokenizer, int(input_len), float(depth), answer, seq_label)
            cases.append(
                {
                    "depth": float(depth),
                    "sample_idx": int(sample_idx),
                    "answer": str(answer),
                    "prompt": prompt,
                    "actual_prompt_tokens": int(actual_tokens),
                    "seq_label": str(seq_label),
                }
            )
    rng.shuffle(cases)
    return cases


def iter_batches(cases: Sequence[Dict[str, Any]], concurrency: int) -> List[List[Dict[str, Any]]]:
    size = int(max(1, concurrency))
    return [list(cases[i : i + size]) for i in range(0, len(cases), size)]


def eval_outputs(outputs: Sequence[str], batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    strict_preds = [first_6digit_strict(o) for o in outputs]
    compat_preds = [first_6digit_compat(o) for o in outputs]
    answers = [str(item["answer"]) for item in batch]
    n = max(1, len(batch))
    completed = sum(1 for o in outputs if str(o).strip())
    valid = sum(1 for p in strict_preds if bool(p))
    valid_compat = sum(1 for p in compat_preds if bool(p))
    em = sum(1 for p, a in zip(strict_preds, answers) if p == a)
    em_compat = sum(1 for p, a in zip(compat_preds, answers) if p == a)
    by_depth: Dict[str, Dict[str, int]] = {}
    for item, pred_strict, pred_compat in zip(batch, strict_preds, compat_preds):
        depth_key = f"{float(item['depth']):.2f}"
        bucket = by_depth.setdefault(depth_key, {"n": 0, "strict_em": 0, "strict_valid": 0, "compat_em": 0, "compat_valid": 0})
        bucket["n"] += 1
        bucket["strict_valid"] += int(bool(pred_strict))
        bucket["compat_valid"] += int(bool(pred_compat))
        bucket["strict_em"] += int(pred_strict == str(item["answer"]) and bool(pred_strict))
        bucket["compat_em"] += int(pred_compat == str(item["answer"]) and bool(pred_compat))
    return {
        "completed": int(completed),
        "completion_rate": float(completed / n),
        "passkey_first_em": float(em / n),
        "valid_rate": float(valid / n),
        "passkey_first_em_compat": float(em_compat / n),
        "valid_rate_compat": float(valid_compat / n),
        "by_depth": {
            depth: {
                "n": int(stats["n"]),
                "passkey_first_em": float(stats["strict_em"] / max(1, stats["n"])),
                "valid_rate": float(stats["strict_valid"] / max(1, stats["n"])),
                "passkey_first_em_compat": float(stats["compat_em"] / max(1, stats["n"])),
                "valid_rate_compat": float(stats["compat_valid"] / max(1, stats["n"])),
            }
            for depth, stats in by_depth.items()
        },
    }


def compression_profile_for_method(method: str) -> str:
    args = dict(COMMON_BASE)
    args.update(GROUP_ARGS[str(method)])
    sink = int(args.get("sink_len", 16))
    obs = int(args.get("snapkv_observation_len", args.get("obs_len", 16)))
    retain = float(args.get("retain_ratio", 1.0))
    p2_sink = int(args.get("p2_sink_tokens", 16))
    p2_recent = int(args.get("p2_recent_tokens", 16))
    return f"sink={sink};snapkv_obs={obs};retain={retain:.3f};p2_sink={p2_sink};p2_recent={p2_recent}"


def aggregate_rows(
    method: str,
    input_len: int,
    concurrency: int,
    gpu_mem_frac_effective: float,
    batch_rows: Sequence[Dict[str, Any]],
    actual_prompt_tokens: Sequence[int],
    error_reason: str = "",
) -> Dict[str, Any]:
    total_cases = sum(int(r["n_cases"]) for r in batch_rows)
    total_completed = sum(int(r["completed"]) for r in batch_rows)
    total_em = sum(float(r["passkey_first_em"]) * int(r["n_cases"]) for r in batch_rows)
    total_valid = sum(float(r["valid_rate"]) * int(r["n_cases"]) for r in batch_rows)
    total_em_compat = sum(float(r["passkey_first_em_compat"]) * int(r["n_cases"]) for r in batch_rows)
    total_valid_compat = sum(float(r["valid_rate_compat"]) * int(r["n_cases"]) for r in batch_rows)
    total_generated_tokens = sum(int(r["generated_tokens"]) for r in batch_rows)
    total_wall_ms = sum(float(r["wall_ms"]) for r in batch_rows)
    total_prefill_ms = sum(float(r["prefill_ms"]) for r in batch_rows)
    total_decode_ms = sum(float(r["decode_ms"]) for r in batch_rows)
    total_batch_tps = [float(r["tokens_per_sec"]) for r in batch_rows]
    all_step_ms: List[float] = []
    all_decode_step_ms: List[float] = []
    ttft_proxy_ms: List[float] = []
    avg_itl_proxy_ms: List[float] = []
    ensure_fail = 0
    prefetch_fail = 0
    p2_attempted_steps = 0
    p2_success_steps = 0
    decode_path_fallback_count = 0
    decode_path_selected_counts: Dict[str, int] = {}
    depth_totals: Dict[str, Dict[str, float]] = {}

    for row in batch_rows:
        all_step_ms.extend(float(x) for x in row.get("step_ms", []))
        all_decode_step_ms.extend(float(x) for x in row.get("decode_step_ms_only", []))
        if row.get("ttft_proxy_ms", 0.0) > 0:
            ttft_proxy_ms.append(float(row["ttft_proxy_ms"]))
        if row.get("avg_itl_proxy_ms", 0.0) > 0:
            avg_itl_proxy_ms.append(float(row["avg_itl_proxy_ms"]))
        ensure_fail += int(row.get("ensure_fail", 0))
        prefetch_fail += int(row.get("prefetch_fail", 0))
        p2_attempted_steps += int(row.get("p2_attempted_steps", 0))
        p2_success_steps += int(row.get("p2_success_steps", 0))
        decode_path_fallback_count += int(row.get("decode_path_fallback_count", 0))
        label = str(row.get("decode_path_selected", "") or "")
        if label:
            decode_path_selected_counts[label] = decode_path_selected_counts.get(label, 0) + 1
        for depth, stats in dict(row.get("by_depth", {}) or {}).items():
            bucket = depth_totals.setdefault(depth, {"n": 0.0, "em": 0.0, "valid": 0.0, "em_compat": 0.0, "valid_compat": 0.0})
            n = float(stats.get("n", 0))
            bucket["n"] += n
            bucket["em"] += float(stats.get("passkey_first_em", 0.0)) * n
            bucket["valid"] += float(stats.get("valid_rate", 0.0)) * n
            bucket["em_compat"] += float(stats.get("passkey_first_em_compat", 0.0)) * n
            bucket["valid_compat"] += float(stats.get("valid_rate_compat", 0.0)) * n

    dominant_path = ""
    if decode_path_selected_counts:
        dominant_path = sorted(decode_path_selected_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[0][0]

    by_depth = {
        depth: {
            "n": int(bucket["n"]),
            "passkey_first_em": float(bucket["em"] / max(1.0, bucket["n"])),
            "valid_rate": float(bucket["valid"] / max(1.0, bucket["n"])),
            "passkey_first_em_compat": float(bucket["em_compat"] / max(1.0, bucket["n"])),
            "valid_rate_compat": float(bucket["valid_compat"] / max(1.0, bucket["n"])),
        }
        for depth, bucket in depth_totals.items()
    }

    return {
        "method": str(method),
        "input_len": int(input_len),
        "concurrency": int(concurrency),
        "gpu_mem_frac_effective": float(gpu_mem_frac_effective),
        "success": int(not error_reason),
        "n_cases": int(total_cases),
        "completed": int(total_completed),
        "completion_rate": float(total_completed / max(1, total_cases)),
        "passkey_first_em": float(total_em / max(1, total_cases)),
        "valid_rate": float(total_valid / max(1, total_cases)),
        "passkey_first_em_compat": float(total_em_compat / max(1, total_cases)),
        "valid_rate_compat": float(total_valid_compat / max(1, total_cases)),
        "actual_prompt_tokens_mean": float(mean(int(x) for x in actual_prompt_tokens)) if actual_prompt_tokens else 0.0,
        "tokens_per_sec_mean_batch": float(mean(total_batch_tps)) if total_batch_tps else 0.0,
        "tokens_per_sec_global": float((1000.0 * total_generated_tokens / total_wall_ms) if total_wall_ms > 0 else 0.0),
        "prefill_ms_total": round(float(total_prefill_ms), 3),
        "decode_ms_total": round(float(total_decode_ms), 3),
        "wall_ms_total": round(float(total_wall_ms), 3),
        "step_ms_p99": round(percentile(all_step_ms, 0.99), 3),
        "decode_step_ms_p99": round(percentile(all_decode_step_ms, 0.99), 3),
        "decode_step_ms_p95": round(percentile(all_decode_step_ms, 0.95), 3),
        "ttft_p95_ms": round(percentile(ttft_proxy_ms, 0.95), 3),
        "ttft_p99_ms": round(percentile(ttft_proxy_ms, 0.99), 3),
        "itl_p95_ms": round(percentile(avg_itl_proxy_ms, 0.95), 3),
        "itl_p99_ms": round(percentile(avg_itl_proxy_ms, 0.99), 3),
        "ttft_proxy_p99_ms": round(percentile(ttft_proxy_ms, 0.99), 3),
        "ensure_fail": int(ensure_fail),
        "prefetch_fail": int(prefetch_fail),
        "p2_attempted_steps": int(p2_attempted_steps),
        "p2_success_steps": int(p2_success_steps),
        "decode_path_selected": dominant_path,
        "decode_path_fallback_count": int(decode_path_fallback_count),
        "compression_profile": compression_profile_for_method(method),
        "chunk_size": int(chunk_size) if int(chunk_size) > 0 else int(COMMON_BASE.get("chunk_size", 0)),
        "prefill_batch_size": int(prefill_batch_size)
        if int(prefill_batch_size) > 0
        else int(COMMON_BASE.get("prefill_batch_size", 0)),
        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        "by_depth": by_depth,
        "error_reason": str(error_reason),
    }


def run_method_concurrency(
    model_name: str,
    method: str,
    input_len: int,
    max_new_tokens: int,
    depths: Sequence[float],
    keys_per_depth: int,
    concurrency: int,
    start_frac: float,
    fallback_step: float,
    fallback_tries: int,
    seed: int,
    chunk_size: int,
    prefill_batch_size: int,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    cases = build_cases(tokenizer, input_len, depths, keys_per_depth, seed)
    actual_prompt_tokens = [int(c["actual_prompt_tokens"]) for c in cases]
    batches = iter_batches(cases, int(concurrency))

    last_error = ""
    for try_idx in range(int(max(1, fallback_tries))):
        frac = max(0.10, float(start_frac) - try_idx * float(max(0.0, fallback_step)))
        engine = None
        batch_rows: List[Dict[str, Any]] = []
        try:
            engine = build_engine(
                model_name,
                method,
                frac,
                max_new_tokens,
                chunk_size=int(chunk_size),
                prefill_batch_size=int(prefill_batch_size),
            )
            for batch_idx, batch in enumerate(batches):
                prompts = [str(x["prompt"]) for x in batch]
                answers = [str(x["answer"]) for x in batch]
                step_rows: List[Dict[str, Any]] = []

                def _step_cb(stat: Dict[str, Any]) -> None:
                    step_rows.append(dict(stat))

                t0 = time.perf_counter()
                outputs, metrics = engine.generate(prompts, return_metrics=True, step_callback=_step_cb)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                quality = eval_outputs(outputs, batch)
                decode_only = [float(s.get("step_ms", 0.0)) for s in step_rows if int(s.get("decode_tokens", 0)) > 0]
                ttft_proxy_ms = float(metrics.get("prefill_ms", 0.0))
                if decode_only:
                    ttft_proxy_ms += float(decode_only[0])
                avg_itl_ms = float(mean(decode_only[1:])) if len(decode_only) > 1 else 0.0
                delta = dict(metrics.get("offloader_delta", {}) or {})
                batch_rows.append(
                    {
                        "batch_idx": int(batch_idx),
                        "n_cases": int(len(batch)),
                        "completed": int(round(float(quality["completion_rate"]) * len(batch))),
                        "passkey_first_em": float(quality["passkey_first_em"]),
                        "valid_rate": float(quality["valid_rate"]),
                        "passkey_first_em_compat": float(quality["passkey_first_em_compat"]),
                        "valid_rate_compat": float(quality["valid_rate_compat"]),
                        "generated_tokens": int(metrics.get("generated_tokens", 0)),
                        "tokens_per_sec": float(metrics.get("tokens_per_sec", 0.0)),
                        "prefill_ms": float(metrics.get("prefill_ms", 0.0)),
                        "decode_ms": float(metrics.get("decode_ms", 0.0)),
                        "wall_ms": float(round(wall_ms, 3)),
                        "step_ms": [float(s.get("step_ms", 0.0)) for s in step_rows],
                        "decode_step_ms_only": decode_only,
                        "ttft_proxy_ms": float(ttft_proxy_ms),
                        "avg_itl_proxy_ms": float(avg_itl_ms),
                        "ensure_fail": int(delta.get("ensure_fail", 0)),
                        "prefetch_fail": int(delta.get("prefetch_fail", 0)),
                        "p2_attempted_steps": int(metrics.get("p2_attempted_steps", 0)),
                        "p2_success_steps": int(metrics.get("p2_success_steps", 0)),
                        "decode_path_selected": str(metrics.get("decode_path_selected", "")),
                        "decode_path_fallback_count": int(metrics.get("decode_path_fallback_count", 0)),
                        "by_depth": dict(quality.get("by_depth", {})),
                    }
                )
            return aggregate_rows(method, input_len, concurrency, frac, batch_rows, actual_prompt_tokens)
        except Exception as exc:
            last_error = str(exc)
        finally:
            cleanup_engine(engine)

    return {
        "method": str(method),
        "input_len": int(input_len),
        "concurrency": int(concurrency),
        "gpu_mem_frac_effective": 0.0,
        "success": 0,
        "n_cases": int(len(cases)),
        "completed": 0,
        "completion_rate": 0.0,
        "passkey_first_em": 0.0,
        "valid_rate": 0.0,
        "passkey_first_em_compat": 0.0,
        "valid_rate_compat": 0.0,
        "actual_prompt_tokens_mean": float(mean(actual_prompt_tokens)) if actual_prompt_tokens else 0.0,
        "tokens_per_sec_mean_batch": 0.0,
        "tokens_per_sec_global": 0.0,
        "prefill_ms_total": 0.0,
        "decode_ms_total": 0.0,
        "wall_ms_total": 0.0,
        "step_ms_p99": 0.0,
        "decode_step_ms_p99": 0.0,
        "decode_step_ms_p95": 0.0,
        "ttft_p95_ms": 0.0,
        "ttft_p99_ms": 0.0,
        "itl_p95_ms": 0.0,
        "itl_p99_ms": 0.0,
        "ttft_proxy_p99_ms": 0.0,
        "ensure_fail": 0,
        "prefetch_fail": 0,
        "p2_attempted_steps": 0,
        "p2_success_steps": 0,
        "decode_path_selected": "",
        "decode_path_fallback_count": 0,
        "compression_profile": compression_profile_for_method(method),
        "chunk_size": int(chunk_size) if int(chunk_size) > 0 else int(COMMON_BASE.get("chunk_size", 0)),
        "prefill_batch_size": int(prefill_batch_size)
        if int(prefill_batch_size) > 0
        else int(COMMON_BASE.get("prefill_batch_size", 0)),
        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        "by_depth": {},
        "error_reason": str(last_error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="32K passkey under load for internal methods.")
    parser.add_argument("--model-name", type=str, default=LOCAL_MODEL_PATH)
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--input-len", type=int, default=DEFAULT_INPUT_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--depths", type=str, default="0.10,0.50,0.90")
    parser.add_argument("--keys-per-depth", type=int, default=DEFAULT_KEYS_PER_DEPTH)
    parser.add_argument("--concurrency-list", type=str, default="1,4,8,16")
    parser.add_argument(
        "--gpu-mem-frac-map",
        type=str,
        default=",".join(f"{k}:{v}" for k, v in DEFAULT_GPU_MEM_FRACS.items()),
        help="Comma-separated method:frac pairs, e.g. off_compress:0.68,p2_only_compress:0.52",
    )
    parser.add_argument("--gpu-mem-frac-fallback-step", type=float, default=DEFAULT_GPU_MEM_FRAC_FALLBACK_STEP)
    parser.add_argument("--gpu-mem-frac-fallback-tries", type=int, default=DEFAULT_GPU_MEM_FRAC_FALLBACK_TRIES)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--prefill-batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=str, default="benchmark_passkey_under_load_results.json")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    depths = parse_csv_floats(args.depths)
    concurrencies = parse_csv_ints(args.concurrency_list)
    frac_map = parse_method_frac_map(args.gpu_mem_frac_map)

    rows: List[Dict[str, Any]] = []
    for method in methods:
        if method not in frac_map:
            raise ValueError(f"missing gpu_mem_frac for method: {method}")
        for concurrency in concurrencies:
            print(f"\n===== method={method} concurrency={concurrency} =====", flush=True)
            row = run_method_concurrency(
                model_name=str(args.model_name),
                method=method,
                input_len=int(args.input_len),
                max_new_tokens=int(args.max_new_tokens),
                depths=depths,
                keys_per_depth=int(args.keys_per_depth),
                concurrency=int(concurrency),
                start_frac=float(frac_map[method]),
                fallback_step=float(args.gpu_mem_frac_fallback_step),
                fallback_tries=int(args.gpu_mem_frac_fallback_tries),
                seed=int(args.seed),
                chunk_size=int(args.chunk_size),
                prefill_batch_size=int(args.prefill_batch_size),
            )
            print(json.dumps({
                "method": row["method"],
                "concurrency": row["concurrency"],
                "gpu_mem_frac_effective": row["gpu_mem_frac_effective"],
                "success": row["success"],
                "completion_rate": row["completion_rate"],
                "passkey_first_em": row["passkey_first_em"],
                "valid_rate": row["valid_rate"],
                "passkey_first_em_compat": row["passkey_first_em_compat"],
                "valid_rate_compat": row["valid_rate_compat"],
                "tokens_per_sec_global": row["tokens_per_sec_global"],
                "decode_step_ms_p95": row["decode_step_ms_p95"],
                "decode_step_ms_p99": row["decode_step_ms_p99"],
                "ttft_p95_ms": row["ttft_p95_ms"],
                "ttft_p99_ms": row["ttft_p99_ms"],
                "itl_p95_ms": row["itl_p95_ms"],
                "itl_p99_ms": row["itl_p99_ms"],
                "ttft_proxy_p99_ms": row["ttft_proxy_p99_ms"],
                "ensure_fail": row["ensure_fail"],
                "prefetch_fail": row["prefetch_fail"],
                "p2_attempted_steps": row["p2_attempted_steps"],
                "p2_success_steps": row["p2_success_steps"],
                "error_reason": row["error_reason"],
            }, ensure_ascii=False, indent=2), flush=True)
            rows.append(row)

    payload = {
        "meta": {
            "task": "passkey_under_load",
            "model_name": str(args.model_name),
            "input_len": int(args.input_len),
            "max_new_tokens": int(args.max_new_tokens),
            "depths": [float(x) for x in depths],
            "keys_per_depth": int(args.keys_per_depth),
            "concurrency_list": [int(x) for x in concurrencies],
            "methods": list(methods),
            "gpu_mem_frac_map": {str(k): float(v) for k, v in frac_map.items()},
            "gpu_mem_frac_fallback_step": float(args.gpu_mem_frac_fallback_step),
            "gpu_mem_frac_fallback_tries": int(args.gpu_mem_frac_fallback_tries),
            "chunk_size": int(args.chunk_size),
            "prefill_batch_size": int(args.prefill_batch_size),
            "seed": int(args.seed),
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
            "report_order": ["completion rate", "valid rate", "EM"],
            "valid_definition": "request completed normally and the target answer field is parseable",
            "em_definition": "request completed normally and the first parsed answer exactly matches the target passkey",
            "ttft_definition": "engine-side proxy equal to prefill_ms + first decode step_ms for each batch",
            "itl_definition": "engine-side proxy equal to the per-batch average of decode steps after the first token, aggregated with p95/p99 across batches",
            "ttft_note": "ttft_p95_ms/ttft_p99_ms are batch-level proxies, not client-observed per-request TTFT.",
            "itl_note": "itl_p95_ms/itl_p99_ms are batch-level proxies from decode step timings, not client-observed per-request ITL.",
        },
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
