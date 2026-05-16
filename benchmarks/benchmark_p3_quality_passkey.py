import json
import os
import random
import re
import time
from collections import Counter
from typing import Dict, List, Tuple

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


LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
CONCURRENCY = 8
MAX_NEW_TOKENS = 32
MEM_FRACS = [0.25, 0.05]
SEED = 20260318

COMMON_BASE = {
    "model_name": LOCAL_MODEL_PATH,
    "cpu_mem_gb": 32.0,
    "chunk_size": 1024,
    "max_new_tokens": MAX_NEW_TOKENS,
    "prefill_batch_size": 4,
    "decode_micro_batch_size": 0,
    "decode_window_sink_len": 64,
    "decode_path_mode": "auto",
    "decode_paged_flash_enabled": True,
    # Isolate decode/P3 impact in passkey quality: avoid prefill compression loss.
    "retain_ratio": 1.0,
}

P3_OFF_ARGS = {
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": False,
}

P3_MAIN_AUTO_ARGS = {
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

METRIC_PROFILE = "strict+compat"
PROMPT_PROFILE = "chat_32"


def compression_profile(args: Dict) -> str:
    sink = int(args.get("sink_len", 64))
    obs = int(args.get("obs_len", 64))
    retain = args.get("retain_ratio", "default_0.20")
    decode_sink = int(args.get("decode_window_sink_len", sink))
    decode_recent = int(args.get("decode_window_recent_len", P3_MAIN_AUTO_ARGS["decode_window_recent_len"]))
    return f"sink={sink};obs={obs};retain={retain};decode_sink={decode_sink};decode_recent={decode_recent}"


def pressure_level(mem_frac: float) -> str:
    return "high" if float(mem_frac) <= 0.10 + 1e-9 else "normal"


def pct(base: float, now: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return (now - base) / base * 100.0


def build_passkey_cases(n: int, seed: int) -> List[Dict[str, str]]:
    random.seed(seed)
    cases = []
    filler = (
        "Long-context filler paragraph for retrieval stress testing. "
        "The text is intentionally verbose and irrelevant to the target key. "
    )
    for i in range(n):
        key = f"{random.randint(100000, 999999)}"
        # Calibrated to avoid degenerate constant-guess collapse on 7B baseline.
        noise_1 = filler * 20
        noise_2 = filler * 20
        user_prompt = (
            f"[seq={i}] You are doing a memory retrieval test.\n"
            "Read the context carefully.\n"
            f"{noise_1}\n"
            f"PASSKEY_RECORD: passkey::{key}::end\n"
            f"{noise_2}\n"
            "Question: What is the passkey in PASSKEY_RECORD?\n"
            "Instruction: Output exactly one line with only the 6-digit number.\n"
            "Do not output any words, punctuation, or explanation."
        )
        cases.append({"seq_id": str(i), "answer": key, "user_prompt": user_prompt})
    return cases


def _format_chat_prompts(tokenizer, cases: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    prompts: List[str] = []
    answers: List[str] = []
    for c in cases:
        answers.append(str(c["answer"]))
        user_prompt = str(c["user_prompt"])
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)
                continue
            except Exception:
                pass
        prompts.append(user_prompt)
    return prompts, answers


def _first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def first_6digit_compat(text: str) -> str:
    m = re.search(r"\b(\d{6})\b", text)
    if not m:
        return ""
    return m.group(1)


def first_6digit_strict(text: str) -> str:
    first = _first_nonempty_line(text).strip("`").strip()
    m = re.fullmatch(r"(?:answer\s*[:：]\s*)?(\d{6})", first or "", flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def eval_passkey(outputs: List[str], answers: List[str]) -> Dict[str, float]:
    preds_strict = [first_6digit_strict(o) for o in outputs]
    preds_compat = [first_6digit_compat(o) for o in outputs]

    valid_strict = sum(1 for p in preds_strict if bool(p))
    valid_compat = sum(1 for p in preds_compat if bool(p))

    first_em_strict = sum(1 for p, a in zip(preds_strict, answers) if p == a)
    first_em_compat = sum(1 for p, a in zip(preds_compat, answers) if p == a)

    contains_em = sum(1 for o, a in zip(outputs, answers) if a in o)
    pred_nonempty = [p for p in preds_compat if p]
    dominant_pred = ""
    dominant_ratio = 0.0
    if pred_nonempty:
        c = Counter(pred_nonempty)
        dominant_pred, dominant_cnt = c.most_common(1)[0]
        dominant_ratio = dominant_cnt / max(1, len(pred_nonempty))

    n = max(1, len(answers))
    return {
        "valid_rate": valid_strict / n,
        "passkey_first_em": first_em_strict / n,
        "valid_rate_compat": valid_compat / n,
        "passkey_first_em_compat": first_em_compat / n,
        "passkey_contains_em": contains_em / n,
        "dominant_pred": dominant_pred,
        "dominant_pred_ratio": dominant_ratio,
        "preds_strict": preds_strict,
        "preds_compat": preds_compat,
    }


def _skip_reason_counts(metrics: Dict) -> Dict[str, int]:
    delta = metrics.get("offloader_delta", {}) or {}
    return {
        "no_candidate_blocks": int(delta.get("window_prune_skip_no_candidate_blocks", 0)),
        "trigger_not_met": int(delta.get("window_prune_skip_trigger_not_met", 0)),
        "min_drop_not_met": int(delta.get("window_prune_skip_min_drop_not_met", 0)),
    }


def _non_trigger_reason(skip_reasons: Dict[str, int], prunes: int, auto_steps: int) -> str:
    if prunes > 0:
        return ""
    if auto_steps <= 0:
        return "auto_not_activated"
    ranked = sorted(skip_reasons.items(), key=lambda kv: kv[1], reverse=True)
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]
    return "no_prune_events"


def run_case(
    name: str,
    gpu_mem_frac: float,
    extra_args: Dict,
    cases: List[Dict[str, str]],
    mode: str,
) -> Dict:
    print(f"\n===== CASE: {name} =====")
    args = dict(COMMON_BASE)
    args["gpu_mem_frac"] = float(gpu_mem_frac)
    args.update(extra_args)
    profile = compression_profile(args)

    engine = ManagedInferenceEngine(**args)
    prompts, answers = _format_chat_prompts(engine.tokenizer, cases)

    t0 = time.perf_counter()
    outputs, metrics, details = engine.generate(prompts, return_metrics=True, return_details=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    quality = eval_passkey(outputs, answers)

    prunes = int(metrics["decode_window_prunes"])
    auto_steps = int(metrics.get("decode_window_auto_active_steps", 0))
    skip_reasons = _skip_reason_counts(metrics)
    level = pressure_level(gpu_mem_frac)
    expected = int(level == "high" and mode == "main_auto")
    observed = int(auto_steps > 0 and prunes > 0)
    non_trigger_reason = _non_trigger_reason(skip_reasons, prunes, auto_steps)
    manual_enabled = int((metrics.get("decode_window_status", {}) or {}).get("manual_enabled", 0))

    if mode == "main_auto" and manual_enabled != 0:
        raise RuntimeError(f"manual_enabled must be 0 in main_auto mode, got {manual_enabled}")

    out = {
        "name": name,
        "mode": mode,
        "gpu_mem_frac": float(gpu_mem_frac),
        "pressure_level": level,
        "auto_trigger_expected": expected,
        "auto_trigger_observed": observed,
        "non_trigger_reason": non_trigger_reason,
        "manual_enabled": manual_enabled,
        "metrics": metrics,
        "details": details,
        "outputs": outputs,
        "quality": quality,
        "skip_reasons": skip_reasons,
        "wall_ms": round(wall_ms, 3),
        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        "metric_profile": METRIC_PROFILE,
        "prompt_profile": PROMPT_PROFILE,
        "compression_profile": profile,
    }

    print(
        json.dumps(
            {
                "name": name,
                "mode": mode,
                "gpu_mem_frac": float(gpu_mem_frac),
                "pressure_level": level,
                "auto_trigger_expected": expected,
                "auto_trigger_observed": observed,
                "non_trigger_reason": non_trigger_reason,
                "tokens_per_sec": metrics["tokens_per_sec"],
                "decode_step_p95_ms": metrics["decode_step_p95_ms"],
                "thrash_win16": metrics["thrash_win16"],
                "decode_window_prunes": metrics["decode_window_prunes"],
                "decode_window_tokens_dropped": metrics["decode_window_tokens_dropped"],
                "window_prune_skip_no_candidate_blocks": skip_reasons["no_candidate_blocks"],
                "window_prune_skip_trigger_not_met": skip_reasons["trigger_not_met"],
                "window_prune_skip_min_drop_not_met": skip_reasons["min_drop_not_met"],
                "passkey_first_em": quality["passkey_first_em"],
                "valid_rate": quality["valid_rate"],
                "passkey_first_em_compat": quality["passkey_first_em_compat"],
                "valid_rate_compat": quality["valid_rate_compat"],
                "passkey_contains_em": quality["passkey_contains_em"],
                "dominant_pred": quality["dominant_pred"],
                "dominant_pred_ratio": quality["dominant_pred_ratio"],
                "manual_enabled": manual_enabled,
                "decode_window_status": metrics["decode_window_status"],
                "decode_path_mode": metrics.get("decode_path_mode", ""),
                "decode_path_selected": metrics.get("decode_path_selected", ""),
                "decode_path_fallback_count": metrics.get("decode_path_fallback_count", 0),
                "decode_path_fallback_reason_topk": metrics.get("decode_path_fallback_reason_topk", {}),
                "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
                "metric_profile": METRIC_PROFILE,
                "prompt_profile": PROMPT_PROFILE,
                "compression_profile": profile,
                "wall_ms": out["wall_ms"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _build_row(mem_frac: float, off: Dict, auto: Dict) -> Dict:
    off_m = off["metrics"]
    auto_m = auto["metrics"]
    off_q = off["quality"]
    auto_q = auto["quality"]
    skip = auto.get("skip_reasons", {})
    row = {
        "experiment_mode": "main_auto",
        "comparison": "off_vs_main_auto",
        "gpu_mem_frac": float(mem_frac),
        "pressure_level": str(auto.get("pressure_level", pressure_level(mem_frac))),
        "auto_trigger_expected": int(auto.get("auto_trigger_expected", 0)),
        "auto_trigger_observed": int(auto.get("auto_trigger_observed", 0)),
        "non_trigger_reason": str(auto.get("non_trigger_reason", "")),
        "manual_enabled": int(auto.get("manual_enabled", 0)),
        "off_tps": float(off_m["tokens_per_sec"]),
        "main_auto_tps": float(auto_m["tokens_per_sec"]),
        "delta_tps_pct": pct(float(off_m["tokens_per_sec"]), float(auto_m["tokens_per_sec"])),
        "off_p95_ms": float(off_m["decode_step_p95_ms"]),
        "main_auto_p95_ms": float(auto_m["decode_step_p95_ms"]),
        "delta_p95_pct": pct(float(off_m["decode_step_p95_ms"]), float(auto_m["decode_step_p95_ms"])),
        "off_thrash_win16": float(off_m["thrash_win16"]),
        "main_auto_thrash_win16": float(auto_m["thrash_win16"]),
        "main_auto_prunes": int(auto_m["decode_window_prunes"]),
        "main_auto_drop_tokens": int(auto_m["decode_window_tokens_dropped"]),
        "main_auto_auto_active_steps": int(auto_m.get("decode_window_auto_active_steps", 0)),
        "main_auto_skip_no_candidate_blocks": int(skip.get("no_candidate_blocks", 0)),
        "main_auto_skip_trigger_not_met": int(skip.get("trigger_not_met", 0)),
        "main_auto_skip_min_drop_not_met": int(skip.get("min_drop_not_met", 0)),
        "off_decode_append_fail_count": int(off_m.get("decode_append_fail_count", 0)),
        "main_auto_decode_append_fail_count": int(auto_m.get("decode_append_fail_count", 0)),
        "off_decode_backpressure_events": int(off_m.get("decode_backpressure_events", 0)),
        "main_auto_decode_backpressure_events": int(auto_m.get("decode_backpressure_events", 0)),
        "off_decode_no_progress_steps": int(off_m.get("decode_no_progress_steps", 0)),
        "main_auto_decode_no_progress_steps": int(auto_m.get("decode_no_progress_steps", 0)),
        "off_decode_path_selected": str(off_m.get("decode_path_selected", "")),
        "main_auto_decode_path_selected": str(auto_m.get("decode_path_selected", "")),
        "off_decode_path_fallback_count": int(off_m.get("decode_path_fallback_count", 0)),
        "main_auto_decode_path_fallback_count": int(auto_m.get("decode_path_fallback_count", 0)),
        "off_decode_path_fallback_reason_topk": dict(off_m.get("decode_path_fallback_reason_topk", {}) or {}),
        "main_auto_decode_path_fallback_reason_topk": dict(
            auto_m.get("decode_path_fallback_reason_topk", {}) or {}
        ),
        "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
        "metric_profile": str(auto.get("metric_profile", METRIC_PROFILE)),
        "prompt_profile": str(auto.get("prompt_profile", PROMPT_PROFILE)),
        "compression_profile": str(auto.get("compression_profile", "")),
        "off_passkey_first_em": float(off_q["passkey_first_em"]),
        "main_auto_passkey_first_em": float(auto_q["passkey_first_em"]),
        "delta_passkey_first_em": float(auto_q["passkey_first_em"]) - float(off_q["passkey_first_em"]),
        "off_valid_rate": float(off_q["valid_rate"]),
        "main_auto_valid_rate": float(auto_q["valid_rate"]),
        "delta_valid_rate": float(auto_q["valid_rate"]) - float(off_q["valid_rate"]),
        "off_passkey_first_em_compat": float(off_q["passkey_first_em_compat"]),
        "main_auto_passkey_first_em_compat": float(auto_q["passkey_first_em_compat"]),
        "delta_passkey_first_em_compat": float(auto_q["passkey_first_em_compat"]) - float(off_q["passkey_first_em_compat"]),
        "off_valid_rate_compat": float(off_q["valid_rate_compat"]),
        "main_auto_valid_rate_compat": float(auto_q["valid_rate_compat"]),
        "delta_valid_rate_compat": float(auto_q["valid_rate_compat"]) - float(off_q["valid_rate_compat"]),
        "off_passkey_contains_em": float(off_q["passkey_contains_em"]),
        "main_auto_passkey_contains_em": float(auto_q["passkey_contains_em"]),
        "delta_passkey_contains_em": float(auto_q["passkey_contains_em"]) - float(off_q["passkey_contains_em"]),
        "off_dominant_pred": str(off_q.get("dominant_pred", "")),
        "main_auto_dominant_pred": str(auto_q.get("dominant_pred", "")),
        "off_dominant_pred_ratio": float(off_q.get("dominant_pred_ratio", 0.0)),
        "main_auto_dominant_pred_ratio": float(auto_q.get("dominant_pred_ratio", 0.0)),
    }
    return row


def main():
    cases = build_passkey_cases(CONCURRENCY, SEED)
    rows = []
    warnings = []

    for mem_frac in MEM_FRACS:
        tag = str(mem_frac).replace(".", "")
        off = run_case(
            name=f"off_{tag}",
            gpu_mem_frac=mem_frac,
            extra_args=P3_OFF_ARGS,
            cases=cases,
            mode="off",
        )
        auto = run_case(
            name=f"main_auto_{tag}",
            gpu_mem_frac=mem_frac,
            extra_args=P3_MAIN_AUTO_ARGS,
            cases=cases,
            mode="main_auto",
        )
        row = _build_row(mem_frac, off, auto)
        rows.append(row)

        if row["auto_trigger_expected"] == 1 and row["auto_trigger_observed"] == 0:
            warn = {
                "mem_frac": float(mem_frac),
                "reason": row["non_trigger_reason"],
                "rerun_suggestion": {
                    "concurrency_steps": [8, 12, 16],
                    "max_new_tokens_steps": [512, 768, 1024],
                },
            }
            warnings.append(warn)
            print(
                "WARNING auto trigger not observed at high pressure "
                f"mem_frac={mem_frac:.2f}, reason={row['non_trigger_reason']}. "
                "Suggested rerun: concurrency [8,12,16], max_new_tokens [512,768,1024]."
            )

    print("\n===== FINAL TABLE =====")
    print(
        "gpu_mem_frac,pressure_level,auto_trigger_expected,auto_trigger_observed,non_trigger_reason,"
        "off_tps,main_auto_tps,delta_tps_pct,"
        "off_p95_ms,main_auto_p95_ms,delta_p95_pct,"
        "off_passkey_first_em,main_auto_passkey_first_em,delta_passkey_first_em,"
        "off_valid_rate,main_auto_valid_rate,delta_valid_rate,"
        "off_passkey_first_em_compat,main_auto_passkey_first_em_compat,delta_passkey_first_em_compat,"
        "off_valid_rate_compat,main_auto_valid_rate_compat,delta_valid_rate_compat,"
        "off_dominant_pred_ratio,main_auto_dominant_pred_ratio"
    )
    for r in rows:
        print(
            f"{r['gpu_mem_frac']:.2f},{r['pressure_level']},{r['auto_trigger_expected']},{r['auto_trigger_observed']},"
            f"{r['non_trigger_reason']},"
            f"{r['off_tps']:.4f},{r['main_auto_tps']:.4f},{r['delta_tps_pct']:+.2f},"
            f"{r['off_p95_ms']:.3f},{r['main_auto_p95_ms']:.3f},{r['delta_p95_pct']:+.2f},"
            f"{r['off_passkey_first_em']:.4f},{r['main_auto_passkey_first_em']:.4f},{r['delta_passkey_first_em']:+.4f},"
            f"{r['off_valid_rate']:.4f},{r['main_auto_valid_rate']:.4f},{r['delta_valid_rate']:+.4f},"
            f"{r['off_passkey_first_em_compat']:.4f},{r['main_auto_passkey_first_em_compat']:.4f},{r['delta_passkey_first_em_compat']:+.4f},"
            f"{r['off_valid_rate_compat']:.4f},{r['main_auto_valid_rate_compat']:.4f},{r['delta_valid_rate_compat']:+.4f},"
            f"{r['off_dominant_pred_ratio']:.4f},{r['main_auto_dominant_pred_ratio']:.4f}"
        )

    payload = {
        "meta": {
            "seed": SEED,
            "concurrency": CONCURRENCY,
            "max_new_tokens": MAX_NEW_TOKENS,
            "mem_fracs": MEM_FRACS,
            "task": "passkey_retrieval_em",
            "groups": ["off", "main_auto"],
            "strict_definition": "first non-empty line must be exactly 6 digits",
            "compat_definition": "first 6-digit match anywhere in output",
            "trigger_policy": "expected_at_high_pressure_but_not_hard_fail",
            "metric_profile": METRIC_PROFILE,
            "prompt_profile": PROMPT_PROFILE,
            "compression_profile": compression_profile(dict(COMMON_BASE, **P3_MAIN_AUTO_ARGS)),
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
            "alloc_conf_value": str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
            "trigger_expectation_warnings": warnings,
        },
        "rows": rows,
    }
    with open("benchmark_p3_quality_passkey_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\nSaved: benchmark_p3_quality_passkey_results.json")


if __name__ == "__main__":
    main()




