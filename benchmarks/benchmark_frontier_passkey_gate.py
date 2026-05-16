import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from engine import ManagedInferenceEngine


LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
DEFAULT_GPU_MEM_FRACS = [0.92, 0.90, 0.88, 0.86, 0.84]
REFERENCE_GROUP = {
    "p2_only_compress": "off_compress",
    "main_auto_compress": "p2_only_compress",
}

COMMON_BASE = {
    "model_name": LOCAL_MODEL_PATH,
    "cpu_mem_gb": 32.0,
    "chunk_size": 1024,
    "max_new_tokens": 32,
    "prefill_batch_size": 4,
    "decode_micro_batch_size": 0,
    "decode_active_cap_initial": 0,
    "max_decode_active_cap": 0,
    "sink_len": 16,
    "obs_len": 16,
    "decode_window_sink_len": 16,
    "decode_path_mode": "rebuild",
    "decode_paged_flash_enabled": False,
}

OFF_RAW_ARGS = {
    "retain_ratio": 1.0,
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": False,
}

OFF_COMPRESS_ARGS = {
    "retain_ratio": 0.10,
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": False,
}

P2_ONLY_COMPRESS_ARGS = {
    "retain_ratio": 0.10,
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": False,
    "offload_budget_blocks": 256,
    "prefetch_budget_blocks": 256,
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

MAIN_AUTO_COMPRESS_ARGS = {
    "retain_ratio": 0.10,
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": True,
    "offload_budget_blocks": 256,
    "prefetch_budget_blocks": 256,
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
    "decode_window_pressure_margin_blocks": 48,
    "decode_window_emergency_check_interval": 4,
    "decode_window_emergency_margin_blocks": 16,
}

GROUP_ARGS = {
    "off_raw": OFF_RAW_ARGS,
    "off_compress": OFF_COMPRESS_ARGS,
    "p2_only_compress": P2_ONLY_COMPRESS_ARGS,
    "main_auto_compress": MAIN_AUTO_COMPRESS_ARGS,
}


def parse_float_list(s: str) -> List[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty gpu_mem_frac list")
    return vals


def build_engine(model_name: str, group: str, gpu_mem_frac: float, max_new_tokens: int) -> ManagedInferenceEngine:
    kwargs = dict(COMMON_BASE)
    kwargs["model_name"] = model_name
    kwargs["gpu_mem_frac"] = float(gpu_mem_frac)
    kwargs["max_new_tokens"] = int(max_new_tokens)
    kwargs.update(GROUP_ARGS[group])
    return ManagedInferenceEngine(**kwargs)


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
    m = re.fullmatch(r"(?:answer\s*[:锛歖\s*)?(\d{6})", first or "", flags=re.IGNORECASE)
    return m.group(1) if m else ""


def build_passkey_cases(tokenizer, input_len: int, concurrency: int, seed: int) -> Tuple[List[str], List[str], List[int]]:
    rng = random.Random(seed ^ (int(input_len) * 1009) ^ (int(concurrency) * 917))
    prompts: List[str] = []
    answers: List[str] = []
    actual_tokens: List[int] = []
    filler = " Long-context filler text about memory systems, cache scheduling, and request interleaving."

    def fill_to(tokens_needed: int, base_ids: List[int]) -> List[int]:
        if tokens_needed <= 0:
            return []
        reps = (tokens_needed + len(base_ids) - 1) // len(base_ids)
        return (base_ids * reps)[:tokens_needed]

    filler_ids = tokenizer(filler, add_special_tokens=False).input_ids or [32]
    for seq_id in range(int(concurrency)):
        answer = f"{rng.randint(100000, 999999)}"
        intro = (
            f"[seq={seq_id}] Read the long context carefully.\n"
            "Return only the 6-digit passkey with no extra words.\n"
        )
        needle = f"\nPASSKEY_RECORD: passkey::{answer}::end\n"
        question = (
            "\nQuestion: What is the passkey in PASSKEY_RECORD?\n"
            "Instruction: Output exactly one line with only the 6-digit number.\n"
        )
        intro_ids = tokenizer(intro, add_special_tokens=False).input_ids
        needle_ids = tokenizer(needle, add_special_tokens=False).input_ids
        question_ids = tokenizer(question, add_special_tokens=False).input_ids
        fixed = len(intro_ids) + len(needle_ids) + len(question_ids)
        remaining = max(0, int(input_len) - fixed)
        left = remaining // 2
        right = remaining - left
        ids = intro_ids + fill_to(left, filler_ids) + needle_ids + fill_to(right, filler_ids) + question_ids
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
        prompts.append(prompt)
        answers.append(answer)
        actual_tokens.append(int(actual))
    return prompts, answers, actual_tokens


def eval_passkey(outputs: List[str], answers: List[str]) -> Dict[str, float]:
    preds_strict = [first_6digit_strict(o) for o in outputs]
    preds_compat = [first_6digit_compat(o) for o in outputs]
    valid_strict = sum(1 for p in preds_strict if bool(p))
    valid_compat = sum(1 for p in preds_compat if bool(p))
    em_strict = sum(1 for p, a in zip(preds_strict, answers) if p == a)
    em_compat = sum(1 for p, a in zip(preds_compat, answers) if p == a)
    pred_nonempty = [p for p in preds_compat if p]
    dominant_ratio = 0.0
    if pred_nonempty:
        counts: Dict[str, int] = {}
        for p in pred_nonempty:
            counts[p] = counts.get(p, 0) + 1
        dominant_ratio = max(counts.values()) / max(1, len(pred_nonempty))
    n = max(1, len(answers))
    return {
        "passkey_first_em": float(em_strict / n),
        "valid_rate": float(valid_strict / n),
        "passkey_first_em_compat": float(em_compat / n),
        "valid_rate_compat": float(valid_compat / n),
        "dominant_pred_ratio": float(dominant_ratio),
    }


def load_perf_rows(path: Path, input_len: int) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("perf_frontier_rows", []))
    out = [r for r in rows if int(r.get("input_len", 0)) == int(input_len)]
    if not out:
        raise ValueError(f"no perf_frontier_rows for input_len={input_len}")
    return out


def perf_rank_key(row: Dict) -> Tuple[int, float, float, str]:
    run = dict(row.get("runnable", {}))
    return (
        -int(row.get("max_supported_concurrency_runnable", 0)),
        -float(run.get("tokens_per_sec", 0.0)),
        float(run.get("decode_step_p95_ms", 1e18)),
        str(row.get("group", "")),
    )


def evaluate_group(
    model_name: str,
    group: str,
    input_len: int,
    concurrency: int,
    frac_candidates: List[float],
    max_new_tokens: int,
    seed: int,
    cache: Dict[Tuple[str, int, int], Dict],
) -> Dict:
    key = (str(group), int(input_len), int(concurrency))
    if key in cache:
        return dict(cache[key])

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompts, answers, actual_prompt_tokens = build_passkey_cases(tokenizer, input_len, concurrency, seed)
    out = {
        "group": str(group),
        "input_len": int(input_len),
        "concurrency": int(concurrency),
        "gpu_mem_frac_effective": 0.0,
        "success": 0,
        "tokens_per_sec": 0.0,
        "decode_step_p95_ms": 0.0,
        "cuda_free_min_gb": 0.0,
        "passkey_first_em": 0.0,
        "valid_rate": 0.0,
        "passkey_first_em_compat": 0.0,
        "valid_rate_compat": 0.0,
        "dominant_pred_ratio": 0.0,
        "actual_prompt_tokens": actual_prompt_tokens,
        "error_reason": "",
    }

    for frac in frac_candidates:
        engine = None
        try:
            engine = build_engine(model_name, group, float(frac), max_new_tokens)
            outputs, metrics = engine.generate(prompts, return_metrics=True)
            quality = eval_passkey(outputs, answers)
            out.update({
                "gpu_mem_frac_effective": float(frac),
                "success": 1,
                "tokens_per_sec": float(metrics.get("tokens_per_sec", 0.0)),
                "decode_step_p95_ms": float(metrics.get("decode_step_p95_ms", 0.0)),
                "cuda_free_min_gb": float(metrics.get("cuda_free_min_gb", 0.0)),
                **quality,
            })
            break
        except Exception as exc:
            out["error_reason"] = str(exc)
        finally:
            cleanup_engine(engine)

    cache[key] = dict(out)
    return out


def absolute_gate_pass(row: Dict, args: argparse.Namespace) -> bool:
    return (
        int(row.get("success", 0)) == 1
        and float(row.get("passkey_first_em", 0.0)) >= float(args.abs_passkey_first_em_min)
        and float(row.get("valid_rate", 0.0)) >= float(args.abs_valid_rate_min)
        and float(row.get("passkey_first_em_compat", 0.0)) >= float(args.abs_passkey_first_em_compat_min)
        and float(row.get("valid_rate_compat", 0.0)) >= float(args.abs_valid_rate_compat_min)
        and float(row.get("dominant_pred_ratio", 1.0)) < float(args.abs_dominant_pred_ratio_max)
    )


def relative_gate_pass(candidate: Dict, reference: Dict, args: argparse.Namespace, candidate_group: str) -> Tuple[bool, Dict[str, float]]:
    if candidate_group == "p2_only_compress":
        em_drop = float(args.rel_p2_passkey_drop_max)
        valid_drop = float(args.rel_p2_valid_drop_max)
    else:
        em_drop = float(args.rel_p3_passkey_drop_max)
        valid_drop = float(args.rel_p3_valid_drop_max)
    delta = {
        "delta_passkey_first_em": float(candidate.get("passkey_first_em", 0.0)) - float(reference.get("passkey_first_em", 0.0)),
        "delta_valid_rate": float(candidate.get("valid_rate", 0.0)) - float(reference.get("valid_rate", 0.0)),
        "delta_passkey_first_em_compat": float(candidate.get("passkey_first_em_compat", 0.0)) - float(reference.get("passkey_first_em_compat", 0.0)),
        "delta_valid_rate_compat": float(candidate.get("valid_rate_compat", 0.0)) - float(reference.get("valid_rate_compat", 0.0)),
    }
    ok = (
        int(candidate.get("success", 0)) == 1
        and int(reference.get("success", 0)) == 1
        and delta["delta_passkey_first_em"] >= -em_drop
        and delta["delta_valid_rate"] >= -valid_drop
        and delta["delta_passkey_first_em_compat"] >= -em_drop
        and delta["delta_valid_rate_compat"] >= -valid_drop
    )
    return ok, delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Passkey gate for perf frontier candidates")
    parser.add_argument("--model-name", type=str, default=LOCAL_MODEL_PATH)
    parser.add_argument("--perf-json", type=str, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gpu-mem-fracs", type=str, default="0.92,0.90,0.88,0.86,0.84")
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--abs-passkey-first-em-min", type=float, default=0.90)
    parser.add_argument("--abs-valid-rate-min", type=float, default=0.95)
    parser.add_argument("--abs-passkey-first-em-compat-min", type=float, default=0.95)
    parser.add_argument("--abs-valid-rate-compat-min", type=float, default=0.98)
    parser.add_argument("--abs-dominant-pred-ratio-max", type=float, default=0.30)
    parser.add_argument("--rel-p2-passkey-drop-max", type=float, default=0.005)
    parser.add_argument("--rel-p2-valid-drop-max", type=float, default=0.005)
    parser.add_argument("--rel-p3-passkey-drop-max", type=float, default=0.010)
    parser.add_argument("--rel-p3-valid-drop-max", type=float, default=0.010)
    parser.add_argument("--out", type=str, default="benchmark_frontier_passkey_gate_results.json")
    args = parser.parse_args()

    frac_candidates = parse_float_list(args.gpu_mem_fracs)
    rows = load_perf_rows(Path(args.perf_json), int(args.input_len))
    ranked = [r for r in sorted(rows, key=perf_rank_key) if int(r.get("max_supported_concurrency_runnable", 0)) > 0]
    selected = ranked[: max(1, int(args.top_k))]

    cache: Dict[Tuple[str, int, int], Dict] = {}
    absolute_rows: List[Dict] = []
    pair_rows: List[Dict] = []

    for row in selected:
        group = str(row.get("group"))
        conc = int(row.get("max_supported_concurrency_runnable", 0))
        abs_eval = evaluate_group(args.model_name, group, int(args.input_len), conc, frac_candidates, int(args.max_new_tokens), int(args.seed), cache)
        abs_eval["absolute_gate_pass"] = int(absolute_gate_pass(abs_eval, args))
        absolute_rows.append(abs_eval)

    for row in selected:
        candidate_group = str(row.get("group"))
        if candidate_group not in REFERENCE_GROUP:
            continue
        ref_group = str(REFERENCE_GROUP[candidate_group])
        ref_row = next((r for r in rows if str(r.get("group")) == ref_group), None)
        if ref_row is None:
            continue
        matched_conc = min(
            int(row.get("max_supported_concurrency_runnable", 0)),
            int(ref_row.get("max_supported_concurrency_runnable", 0)),
        )
        if matched_conc <= 0:
            continue
        candidate_eval = evaluate_group(args.model_name, candidate_group, int(args.input_len), matched_conc, frac_candidates, int(args.max_new_tokens), int(args.seed), cache)
        ref_eval = evaluate_group(args.model_name, ref_group, int(args.input_len), matched_conc, frac_candidates, int(args.max_new_tokens), int(args.seed), cache)
        rel_pass, delta = relative_gate_pass(candidate_eval, ref_eval, args, candidate_group)
        pair_rows.append({
            "candidate_group": candidate_group,
            "reference_group": ref_group,
            "input_len": int(args.input_len),
            "matched_concurrency": int(matched_conc),
            "candidate": candidate_eval,
            "reference": ref_eval,
            **delta,
            "relative_gate_pass": int(rel_pass),
        })

    payload = {
        "meta": {
            "task": "frontier_passkey_gate",
            "model_name": args.model_name,
            "perf_json": str(args.perf_json),
            "input_len": int(args.input_len),
            "top_k": int(args.top_k),
            "gpu_mem_fracs": [float(x) for x in frac_candidates],
            "max_new_tokens": int(args.max_new_tokens),
            "seed": int(args.seed),
        },
        "selected_rows": selected,
        "absolute_rows": absolute_rows,
        "pair_rows": pair_rows,
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
