import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(CONFIGS_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIGS_DIR))

from engine import ManagedInferenceEngine, OnlineRequest, REQ_WAITING_PREFILL, REQ_DONE  # noqa: E402
from strategy_groups import COMMON_BASE, GROUP_ARGS  # noqa: E402

METHOD_LABELS = {
    "hf_vanilla": "HuggingFace",
    "vllm": "vLLM",
    "legacy_off_raw_page16": "UnifiedKV-Raw",
    "off_compress_page16": "UnifiedKV-Compress",
    "off_compress_page16_b2048": "UnifiedKV-Compress-b2048",
    "p2_page16_offline": "UnifiedKV-Offline",
    "p2_page16_offline_b2048": "UnifiedKV-Offline-b2048",
}
OOM_KEYWORDS = (
    "out of memory",
    "cuda out of memory",
    "cublas_status_alloc_failed",
    "allocation failed",
    "no free blocks",
    "cannot evict",
    "direct_decode_expand_evict_failed",
)


def parse_csv(text: str) -> List[str]:
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty csv")
    return vals


def parse_int_csv(text: str) -> List[int]:
    return [int(x) for x in parse_csv(text)]


def parse_frac_map(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not str(text or "").strip():
        return out
    for chunk in str(text).split(","):
        if not chunk.strip():
            continue
        k, v = chunk.split(":", 1)
        out[k.strip()] = float(v.strip())
    return out


def is_oom_error(text: str) -> bool:
    s = str(text or "").lower()
    return any(k in s for k in OOM_KEYWORDS)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(str(model_name), trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def choose_filler_ids(tokenizer, count: int = 512) -> List[int]:
    special = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if x is not None)
    candidates: List[int] = []
    for text in (" the", " a", " to", " of", " and", " in", " for", " with", "x", "0", "1", "2"):
        ids = tokenizer(text, add_special_tokens=False).input_ids
        for tid in ids:
            tid = int(tid)
            if tid not in special and tid >= 0 and tid not in candidates:
                candidates.append(tid)
    vocab = int(getattr(tokenizer, "vocab_size", 32000) or 32000)
    for tid in range(10, min(vocab, 10000)):
        if len(candidates) >= int(count):
            break
        if tid not in special and tid not in candidates:
            candidates.append(int(tid))
    if not candidates:
        return [10]
    return candidates


def build_fixed_input_ids(tokenizer, input_len: int, batch_size: int) -> List[List[int]]:
    """Build exact-length synthetic prompts with deterministic per-request variation.

    The variation avoids accidentally giving vLLM/HF/UnifiedKV identical long prefixes.
    Input length is the actual input_ids length, including BOS when available.
    """
    if input_len <= 0:
        raise ValueError("input_len must be positive")
    bos = tokenizer.bos_token_id
    fillers = choose_filler_ids(tokenizer)
    base_filler = int(fillers[0])
    has_bos = bos is not None and int(bos) >= 0 and input_len >= 1
    seqs: List[List[int]] = []
    start = 1 if has_bos else 0
    for req_i in range(int(batch_size)):
        if has_bos:
            seq = [int(bos)] + [base_filler] * (int(input_len) - 1)
        else:
            seq = [base_filler] * int(input_len)
        # Break the shared prefix immediately after BOS and scatter a few markers.
        for j, pos in enumerate(range(start, min(int(input_len), start + 32))):
            seq[pos] = int(fillers[(req_i * 37 + j) % len(fillers)])
        for pos in range(start + 128, int(input_len), 1024):
            seq[pos] = int(fillers[(req_i * 53 + pos // 1024) % len(fillers)])
        if len(seq) != int(input_len):
            raise RuntimeError(f"bad synthetic input length: {len(seq)} != {input_len}")
        seqs.append(seq)
    return seqs


def hf_forward_last_logits(model, input_ids: torch.Tensor, past_key_values: Any = None) -> Tuple[Any, torch.Tensor, str]:
    """Forward while materializing only the final-position logits when possible."""
    kwargs: Dict[str, Any] = {"input_ids": input_ids, "use_cache": True}
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    try:
        out = model(**kwargs, logits_to_keep=1)
        return out, out.logits[:, -1, :], "logits_to_keep=1"
    except TypeError:
        pass

    base_model = getattr(model, "model", None) or getattr(model, "transformer", None)
    lm_head = getattr(model, "lm_head", None)
    if base_model is not None and lm_head is not None:
        out = base_model(**kwargs)
        hidden = out.last_hidden_state[:, -1:, :]
        logits = lm_head(hidden)[:, -1, :]
        return out, logits, "base_model_last_hidden_lm_head"

    out = model(**kwargs)
    return out, out.logits[:, -1, :], "full_logits_fallback"


def summarize_step_rows(step_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not step_rows:
        return {}
    last = dict(step_rows[-1])
    out: Dict[str, Any] = {}
    keys_max = [
        "decode_page16_native_steps",
        "decode_rebuild_steps",
        "decode_materialize_kv_bytes",
        "decode_memory_cap_events",
        "decode_backpressure_events",
        "decode_retry_timeout_fail_count",
        "p2_attempted_steps",
        "p2_success_steps",
        "p2_ready_actual_reclaim_blocks",
        "kv_total_blocks",
        "kv_peak_used_blocks",
        "decode_active_cap",
        "decode_active_cap_boot",
        "decode_active_cap_min_seen",
    ]
    for key in keys_max:
        vals = []
        for row in step_rows:
            try:
                vals.append(int(row.get(key, 0)))
            except Exception:
                pass
        out[key] = max(vals) if vals else int(last.get(key, 0) or 0)
    free_vals = []
    for row in step_rows:
        for key in ("n_free", "global_min_n_free", "decode_min_n_free"):
            try:
                v = int(row.get(key, -1))
            except Exception:
                v = -1
            if v >= 0:
                free_vals.append(v)
    out["min_free_blocks"] = min(free_vals) if free_vals else -1
    for key in ("decode_backend", "decode_path_mode", "decode_path_selected", "decode_page16_native_blocked_reason"):
        out[key] = str(last.get(key, ""))
    return out


def run_hf_worker(model_name: str, input_ids_list: List[List[int]], max_new_tokens: int) -> Dict[str, Any]:
    tokenizer = load_tokenizer(model_name)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    generated = 0
    logits_policy = ""
    with torch.inference_mode():
        out, logits, logits_policy = hf_forward_last_logits(model, input_ids)
        past = out.past_key_values
        next_tok = torch.argmax(logits, dim=-1)
        generated += int(next_tok.numel())
        for _ in range(1, int(max_new_tokens)):
            out, logits, logits_policy = hf_forward_last_logits(model, next_tok.view(-1, 1), past)
            past = out.past_key_values
            next_tok = torch.argmax(logits, dim=-1)
            generated += int(next_tok.numel())
    wall_ms = (time.perf_counter() - t0) * 1000.0
    peak_alloc = float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved() / 1024**3) if torch.cuda.is_available() else 0.0
    del model, tokenizer, input_ids, out, past, next_tok
    cleanup_cuda()
    return {
        "status": "Success",
        "completed_requests": len(input_ids_list),
        "generated_tokens": int(generated),
        "tokens_per_sec": float(generated / max(1e-6, wall_ms / 1000.0)),
        "wall_clock_total_runtime_ms": round(float(wall_ms), 3),
        "cuda_alloc_peak_gb": round(peak_alloc, 4),
        "cuda_reserved_peak_gb": round(peak_reserved, 4),
        "hf_logits_policy": str(logits_policy),
        "error_reason": "",
    }


def run_vllm_worker(model_name: str, input_ids_list: List[List[int]], max_new_tokens: int, gpu_memory_utilization: float) -> Dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    max_model_len = max(len(x) for x in input_ids_list) + int(max_new_tokens) + 16
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=int(max_model_len),
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_new_tokens), ignore_eos=True)
    prompts = [TokensPrompt(prompt_token_ids=list(ids)) for ids in input_ids_list]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    generated = 0
    completed = 0
    for out in outs:
        if out.outputs:
            n = len(out.outputs[0].token_ids)
            generated += int(n)
            if n >= int(max_new_tokens):
                completed += 1
    del llm, outs, prompts
    cleanup_cuda()
    return {
        "status": "Success" if completed == len(input_ids_list) else "Degraded",
        "completed_requests": int(completed),
        "generated_tokens": int(generated),
        "tokens_per_sec": float(generated / max(1e-6, wall_ms / 1000.0)),
        "wall_clock_total_runtime_ms": round(float(wall_ms), 3),
        "submitted_concurrency": int(len(input_ids_list)),
        "actual_resident_concurrency_observed": -1,
        "vllm_enable_prefix_caching": 0,
        "vllm_concurrency_note": "LLM.generate submits this many prompts; vLLM may internally queue requests instead of keeping all prompts resident simultaneously.",
        "error_reason": "",
    }


def build_engine(model_name: str, method: str, gpu_mem_frac: float, max_new_tokens: int) -> ManagedInferenceEngine:
    args = dict(COMMON_BASE)
    args["model_name"] = str(model_name)
    args["gpu_mem_frac"] = float(gpu_mem_frac)
    args["max_new_tokens"] = int(max_new_tokens)
    args.update(GROUP_ARGS[str(method)])
    return ManagedInferenceEngine(**args)


def apply_offline_budget_planner(engine: ManagedInferenceEngine, submitted_batch: int, max_new_tokens: int) -> Dict[str, Any]:
    """Set deterministic decode active cap from fixed-budget KV demand."""
    sched = engine.scheduler
    block_size = int(getattr(sched.pool, "B", 16) or 16)
    kv_total_blocks = int(getattr(sched.pool, "N_total", 0) or 0)
    low_blocks = int(getattr(sched.pool, "N_wm_low", 0) or 0)
    snapkv = getattr(sched, "snapkv", None)
    retain_budget_tokens = int(getattr(snapkv, "retain_budget_tokens", 0) or 0)
    retain_ratio = float(getattr(snapkv, "retain_ratio", 0.0) or 0.0)
    if retain_budget_tokens > 0:
        retained_tokens = int(retain_budget_tokens)
        compression_mode = "fixed_budget"
    else:
        retained_tokens = int(max_new_tokens)
        if retain_ratio > 0:
            retained_tokens = max(1, int(round(float(max_new_tokens) / max(1e-6, retain_ratio))))
        compression_mode = "ratio_estimate"
    blocks_per_seq = int((int(retained_tokens) + int(max_new_tokens) + block_size - 1) // block_size)
    available_blocks = max(0, int(kv_total_blocks) - int(low_blocks))
    safe_cap = max(1, int(available_blocks // max(1, blocks_per_seq)))
    planned_cap = max(1, min(int(submitted_batch), int(safe_cap)))
    engine.decode_active_cap_initial = int(planned_cap)
    engine.max_decode_active_cap = int(planned_cap)
    engine.decode_active_cap_min = int(planned_cap)
    return {
        "offline_budget_planner_enabled": 1,
        "offline_planner_submitted_batch": int(submitted_batch),
        "offline_planner_block_size": int(block_size),
        "offline_planner_kv_total_blocks": int(kv_total_blocks),
        "offline_planner_low_watermark_blocks": int(low_blocks),
        "offline_planner_available_blocks": int(available_blocks),
        "offline_planner_retain_budget_tokens": int(retain_budget_tokens),
        "offline_planner_retained_tokens": int(retained_tokens),
        "offline_planner_max_new_tokens": int(max_new_tokens),
        "offline_planner_blocks_per_seq": int(blocks_per_seq),
        "offline_planner_safe_active_cap": int(safe_cap),
        "offline_planner_decode_active_cap": int(planned_cap),
        "offline_planner_compression_mode": str(compression_mode),
    }


def run_unifiedkv_worker(model_name: str, method: str, input_ids_list: List[List[int]], max_new_tokens: int, gpu_mem_frac: float, timeout_s: float) -> Dict[str, Any]:
    engine: Optional[ManagedInferenceEngine] = None
    step_rows: List[Dict[str, Any]] = []
    try:
        engine = build_engine(model_name, method, gpu_mem_frac, max_new_tokens)
        planner = apply_offline_budget_planner(engine, len(input_ids_list), int(max_new_tokens))
        engine.tokenizer.eos_token_id = None
        engine._reset_online_runtime(clear_request_counter=True)
        engine._return_details_online = False
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        engine._online_total_t0 = t0
        for i, ids in enumerate(input_ids_list):
            ids_cpu = torch.tensor(ids, dtype=torch.long).cpu()
            req = OnlineRequest(
                request_id=int(i),
                prompt="<synthetic_token_ids>",
                max_new_tokens=int(max_new_tokens),
                arrival_step=int(engine._online_step),
                state=REQ_WAITING_PREFILL,
                input_ids_cpu=ids_cpu,
                prompt_token_len=int(ids_cpu.numel()),
                submit_time_s=float(t0),
            )
            engine._requests[int(i)] = req
            engine.waiting_queue.append(int(i))
        last_progress_t = t0
        progress_every_steps = 50
        progress_every_s = 30.0
        while engine.has_pending_requests():
            elapsed_s = time.perf_counter() - t0
            if elapsed_s > float(timeout_s):
                raise TimeoutError(f"worker timeout after {timeout_s}s")
            stat = engine.step()
            step_rows.append(dict(stat))
            step_id = int(stat.get("step", getattr(engine, "_online_step", len(step_rows))))
            now = time.perf_counter()
            if (len(step_rows) == 1 or step_id % progress_every_steps == 0 or (now - last_progress_t) >= progress_every_s):
                last_progress_t = now
                completed_now = 0
                try:
                    completed_now = sum(1 for r in engine._requests.values() if str(getattr(r, "state", "")) == REQ_DONE)
                except Exception:
                    completed_now = -1
                kv_total = int(stat.get("kv_total_blocks", 0) or 0)
                kv_peak = int(stat.get("kv_peak_used_blocks", 0) or 0)
                n_free = int(stat.get("n_free", stat.get("global_min_n_free", -1)) or -1)
                used_blocks = (kv_total - n_free) if kv_total > 0 and n_free >= 0 else kv_peak
                used_pct = (100.0 * used_blocks / kv_total) if kv_total > 0 else 0.0
                progress = {
                    "progress": str(method),
                    "step": step_id,
                    "elapsed_s": round(float(elapsed_s), 2),
                    "completed_requests": int(completed_now),
                    "waiting_queue": len(getattr(engine, "waiting_queue", []) or []),
                    "decode_backend": str(stat.get("decode_backend", stat.get("decode_path_selected", ""))),
                    "decode_page16_native_steps": int(stat.get("decode_page16_native_steps", 0) or 0),
                    "decode_rebuild_steps": int(stat.get("decode_rebuild_steps", 0) or 0),
                    "decode_materialize_kv_bytes": int(stat.get("decode_materialize_kv_bytes", 0) or 0),
                    "kv_total_blocks": kv_total,
                    "n_free": n_free,
                    "kv_used_blocks": int(used_blocks),
                    "kv_used_pct": round(float(used_pct), 2),
                    "kv_peak_used_blocks": kv_peak,
                    "decode_active_cap": int(stat.get("decode_active_cap", 0) or 0),
                    "p2_attempted_steps": int(stat.get("p2_attempted_steps", 0) or 0),
                    "p2_success_steps": int(stat.get("p2_success_steps", 0) or 0),
                    "p2_ready_actual_reclaim_blocks": int(stat.get("p2_ready_actual_reclaim_blocks", 0) or 0),
                    "p2_ready_offload_blocks_total": int(stat.get("p2_ready_offload_blocks_total", 0) or 0),
                    "offload_success": int(stat.get("offloader_delta_step.offload_success", stat.get("offload_success", 0)) or 0),
                    "prefetch_success": int(stat.get("offloader_delta_step.prefetch_success", stat.get("prefetch_success", 0)) or 0),
                    "page16_kernel_ms": float(stat.get("decode_page16_native_kernel_ms", 0.0) or 0.0),
                }
                print(json.dumps(progress, ensure_ascii=False), flush=True)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        finished = engine.collect_finished()
        completed = sum(1 for r in finished if str(r.get("state")) == REQ_DONE and len(r.get("token_ids") or []) >= int(max_new_tokens))
        generated = sum(len(r.get("token_ids") or []) for r in finished)
        errors = [str(r.get("error", "")) for r in finished if str(r.get("error", "")).strip()]
        metrics = summarize_step_rows(step_rows)
        peak_alloc = float(torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0
        peak_reserved = float(torch.cuda.max_memory_reserved() / 1024**3) if torch.cuda.is_available() else 0.0
        status = "Success" if completed == len(input_ids_list) and generated == len(input_ids_list) * int(max_new_tokens) else "Degraded"
        result = {
            "status": status,
            "completed_requests": int(completed),
            "generated_tokens": int(generated),
            "tokens_per_sec": float(generated / max(1e-6, wall_ms / 1000.0)),
            "wall_clock_total_runtime_ms": round(float(wall_ms), 3),
            "cuda_alloc_peak_gb": round(peak_alloc, 4),
            "cuda_reserved_peak_gb": round(peak_reserved, 4),
            "error_reason": "; ".join(errors[:3]),
        }
        result.update(metrics)
        result.update(planner)
        return result
    finally:
        if engine is not None:
            try:
                engine._reset_online_runtime(clear_request_counter=False)
            except Exception:
                pass
            del engine
        cleanup_cuda()


def worker_main(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.model_name)
    input_ids_list = build_fixed_input_ids(tokenizer, int(args.input_len), int(args.concurrency))
    row: Dict[str, Any] = {
        "method": args.method,
        "method_label": METHOD_LABELS.get(args.method, args.method),
        "concurrency": int(args.concurrency),
        "input_len": int(args.input_len),
        "max_new_tokens": int(args.max_new_tokens),
        "actual_input_len_min": min(len(x) for x in input_ids_list),
        "actual_input_len_max": max(len(x) for x in input_ids_list),
        "forced_decode_tokens": int(args.max_new_tokens),
        "prompt_variation": "deterministic_per_request_non_special_token_pattern",
    }
    try:
        if args.method == "hf_vanilla":
            result = run_hf_worker(args.model_name, input_ids_list, int(args.max_new_tokens))
        elif args.method == "vllm":
            result = run_vllm_worker(args.model_name, input_ids_list, int(args.max_new_tokens), float(args.vllm_gpu_memory_utilization))
        else:
            frac_map = parse_frac_map(args.gpu_mem_frac_map)
            if args.method not in frac_map:
                raise ValueError(f"missing gpu_mem_frac for {args.method}")
            result = run_unifiedkv_worker(
                args.model_name,
                args.method,
                input_ids_list,
                int(args.max_new_tokens),
                float(frac_map[args.method]),
                float(args.worker_timeout_s),
            )
            result["gpu_mem_frac"] = float(frac_map[args.method])
        row.update(result)
        if int(row.get("generated_tokens", 0)) != int(args.concurrency) * int(args.max_new_tokens) and row.get("status") == "Success":
            row["status"] = "Degraded"
            row["error_reason"] = (str(row.get("error_reason", "")) + "; generated token count mismatch").strip("; ")
    except Exception as exc:
        err = str(exc) or repr(exc)
        row.update({
            "status": "Failed/OOM" if is_oom_error(err) else "Failed",
            "completed_requests": 0,
            "generated_tokens": 0,
            "tokens_per_sec": 0.0,
            "wall_clock_total_runtime_ms": 0.0,
            "error_reason": err,
            "traceback_tail": traceback.format_exc()[-4000:],
        })
    out_path = Path(args.worker_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def orchestrator_main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_csv(args.methods)
    concurrencies = parse_int_csv(args.concurrency_list)
    rows: List[Dict[str, Any]] = []
    for method in methods:
        consecutive_oom = 0
        for c in concurrencies:
            cell_dir = out_dir / f"{method}_c{c}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            result_path = cell_dir / "result.json"
            log_path = cell_dir / "run.log"
            if consecutive_oom >= 2:
                row = {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "concurrency": int(c),
                    "input_len": int(args.input_len),
                    "max_new_tokens": int(args.max_new_tokens),
                    "status": "Failed/OOM",
                    "tokens_per_sec": 0.0,
                    "completed_requests": 0,
                    "generated_tokens": 0,
                    "skipped_due_to_consecutive_oom": 1,
                    "error_reason": "skipped after two consecutive deterministic OOM points",
                }
                result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append(row)
                print(json.dumps({"skip": method, "concurrency": c, "reason": row["error_reason"]}), flush=True)
                continue
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--method", method,
                "--model-name", args.model_name,
                "--input-len", str(args.input_len),
                "--max-new-tokens", str(args.max_new_tokens),
                "--concurrency", str(c),
                "--gpu-mem-frac-map", args.gpu_mem_frac_map,
                "--vllm-gpu-memory-utilization", str(args.vllm_gpu_memory_utilization),
                "--worker-timeout-s", str(args.worker_timeout_s),
                "--worker-out", str(result_path),
            ]
            print(json.dumps({"start": method, "concurrency": c, "out": str(result_path)}), flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=float(args.subprocess_timeout_s))
            if result_path.exists():
                row = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                row = {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "concurrency": int(c),
                    "input_len": int(args.input_len),
                    "max_new_tokens": int(args.max_new_tokens),
                    "status": "Failed",
                    "tokens_per_sec": 0.0,
                    "completed_requests": 0,
                    "generated_tokens": 0,
                    "error_reason": f"worker produced no result rc={proc.returncode}",
                }
            row["worker_returncode"] = int(proc.returncode)
            rows.append(row)
            print(json.dumps({"end": method, "concurrency": c, "status": row.get("status"), "tokens_per_sec": row.get("tokens_per_sec")}, ensure_ascii=False), flush=True)
            if str(row.get("status")) == "Failed/OOM":
                consecutive_oom += 1
            else:
                consecutive_oom = 0
    payload = {
        "meta": {
            "task": "offline_fixed_input_1024_output_capacity",
            "model_name": args.model_name,
            "input_len_definition": "actual model input_ids length, including BOS/special token when tokenizer defines one",
            "input_len": int(args.input_len),
            "max_new_tokens": int(args.max_new_tokens),
            "methods": methods,
            "concurrency_list": concurrencies,
            "gpu_mem_frac_map": parse_frac_map(args.gpu_mem_frac_map),
            "vllm_gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
            "skip_policy": "run ascending concurrency; after two consecutive deterministic OOM points, mark higher concurrency as Failed/OOM",
        },
        "rows": rows,
    }
    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_file}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline fixed-token input + 1024 output capacity benchmark")
    p.add_argument("--model-name", type=str, default="/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--methods", type=str, default="hf_vanilla,vllm,legacy_off_raw_page16,off_compress_page16,p2_page16_offline")
    p.add_argument("--concurrency-list", type=str, default="1,2,4,8,16,32")
    p.add_argument("--input-len", type=int, default=32768)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--gpu-mem-frac-map", type=str, default="legacy_off_raw_page16:0.70,off_compress_page16:0.70,p2_page16_offline:0.70")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.96)
    p.add_argument("--worker-timeout-s", type=float, default=7200.0)
    p.add_argument("--subprocess-timeout-s", type=float, default=9000.0)
    p.add_argument("--out-dir", type=str, default="benchmarks/results/probes/offline_32k_capacity")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--method", type=str, default="")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--worker-out", type=str, default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.worker:
        worker_main(args)
    else:
        orchestrator_main(args)


if __name__ == "__main__":
    main()
