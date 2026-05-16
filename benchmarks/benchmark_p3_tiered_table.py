import gc
import json
import os
import time
from statistics import mean
from typing import Dict, List

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
PROMPT_REPEAT = 100
MAX_NEW_TOKENS = 512
MEM_FRACS = [0.30, 0.25, 0.10, 0.05]

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
}

P3_OFF_ARGS = {
    "decode_window_enabled": False,
    "decode_window_auto_on_pressure": False,
}

# Single shared auto configuration for all mem_frac levels.
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

METRIC_PROFILE = "perf_stability"
PROMPT_PROFILE = "stress_repeat_100"


def compression_profile(args: Dict) -> str:
    sink = int(args.get("sink_len", 64))
    obs = int(args.get("obs_len", 64))
    retain = args.get("retain_ratio", "default_0.20")
    decode_sink = int(args.get("decode_window_sink_len", sink))
    decode_recent = int(args.get("decode_window_recent_len", P3_MAIN_AUTO_ARGS["decode_window_recent_len"]))
    return f"sink={sink};obs={obs};retain={retain};decode_sink={decode_sink};decode_recent={decode_recent}"


def pressure_level(mem_frac: float) -> str:
    return "high" if float(mem_frac) <= 0.10 + 1e-9 else "normal"


def build_prompts(n: int, repeat: int) -> List[str]:
    base = (
        "This is a long-context stress prompt for concurrent decode benchmarking. "
        "Keep generating coherent technical explanation about cache management, "
        "prefill/decode behavior, and memory-pressure fallback strategy. "
    )
    prompts = []
    for i in range(n):
        prompts.append(f"[seq={i}] {(base * repeat)} Continue with detailed and non-repetitive content.")
    return prompts


def pct(base: float, now: float) -> float:
    if abs(base) < 1e-12:
        return 0.0
    return (now - base) / base * 100.0


def token_agreement(a: List[List[int]], b: List[List[int]], start: int = 0, end: int = None) -> float:
    eq = 0
    total = 0
    for xa, xb in zip(a, b):
        hi = min(len(xa), len(xb))
        if end is not None:
            hi = min(hi, end)
        lo = min(start, hi)
        for i in range(lo, hi):
            total += 1
            if xa[i] == xb[i]:
                eq += 1
    if total == 0:
        return 1.0
    return eq / total


def avg_logprob(details: Dict) -> float:
    vals = []
    for seq in details.get("token_logprobs", []):
        vals.extend(seq)
    if not vals:
        return 0.0
    return float(mean(vals))


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
    prompts: List[str],
    mode: str,
) -> Dict:
    print(f"\n===== CASE: {name} =====")
    args = dict(COMMON_BASE)
    args["gpu_mem_frac"] = float(gpu_mem_frac)
    args.update(extra_args)
    profile = compression_profile(args)

    engine = ManagedInferenceEngine(**args)

    t0 = time.perf_counter()
    outputs, metrics, details = engine.generate(prompts, return_metrics=True, return_details=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0

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
        "sample_output": outputs[0][:160] if outputs else "",
        "wall_ms": round(wall_ms, 3),
        "skip_reasons": skip_reasons,
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
                "decode_window_prunes": prunes,
                "decode_window_tokens_dropped": metrics["decode_window_tokens_dropped"],
                "decode_window_auto_active_steps": auto_steps,
                "window_prune_skip_no_candidate_blocks": skip_reasons["no_candidate_blocks"],
                "window_prune_skip_trigger_not_met": skip_reasons["trigger_not_met"],
                "window_prune_skip_min_drop_not_met": skip_reasons["min_drop_not_met"],
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
    off_d = off["details"]
    auto_d = auto["details"]
    auto_status = auto_m.get("decode_window_status", {})
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
        "off_wall_ms": float(off["wall_ms"]),
        "main_auto_wall_ms": float(auto["wall_ms"]),
        "delta_wall_pct": pct(float(off["wall_ms"]), float(auto["wall_ms"])),
        "off_thrash_win16": float(off_m["thrash_win16"]),
        "main_auto_thrash_win16": float(auto_m["thrash_win16"]),
        "main_auto_prunes": int(auto_m["decode_window_prunes"]),
        "main_auto_drop_tokens": int(auto_m["decode_window_tokens_dropped"]),
        "main_auto_auto_active_steps": int(auto_m.get("decode_window_auto_active_steps", 0)),
        "main_auto_prune_frozen_count": int(auto_status.get("prune_frozen_count", 0)),
        "main_auto_skip_no_candidate_blocks": int(skip.get("no_candidate_blocks", 0)),
        "main_auto_skip_trigger_not_met": int(skip.get("trigger_not_met", 0)),
        "main_auto_skip_min_drop_not_met": int(skip.get("min_drop_not_met", 0)),
        "off_decode_append_fail_count": int(off_m.get("decode_append_fail_count", 0)),
        "main_auto_decode_append_fail_count": int(auto_m.get("decode_append_fail_count", 0)),
        "off_decode_backpressure_events": int(off_m.get("decode_backpressure_events", 0)),
        "main_auto_decode_backpressure_events": int(auto_m.get("decode_backpressure_events", 0)),
        "off_decode_no_progress_steps": int(off_m.get("decode_no_progress_steps", 0)),
        "main_auto_decode_no_progress_steps": int(auto_m.get("decode_no_progress_steps", 0)),
        "main_auto_mode_level": int(auto_status.get("mode_level", 0)),
        "main_auto_activations": int(auto_status.get("auto_activations", 0)),
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
        "agree_all": token_agreement(off_d["token_ids"], auto_d["token_ids"]),
        "agree_first_32": token_agreement(off_d["token_ids"], auto_d["token_ids"], 0, 32),
        "agree_first_100": token_agreement(off_d["token_ids"], auto_d["token_ids"], 0, 100),
        "agree_after_128": token_agreement(off_d["token_ids"], auto_d["token_ids"], 128, None),
        "off_avg_logprob": avg_logprob(off_d),
        "main_auto_avg_logprob": avg_logprob(auto_d),
        "delta_avg_logprob": avg_logprob(auto_d) - avg_logprob(off_d),
    }
    return row


def main():
    prompts = build_prompts(CONCURRENCY, PROMPT_REPEAT)
    rows = []
    warnings = []

    for mem_frac in MEM_FRACS:
        mem_tag = str(mem_frac).replace(".", "")
        off = run_case(
            name=f"off_{mem_tag}",
            gpu_mem_frac=mem_frac,
            extra_args=P3_OFF_ARGS,
            prompts=prompts,
            mode="off",
        )
        auto = run_case(
            name=f"main_auto_{mem_tag}",
            gpu_mem_frac=mem_frac,
            extra_args=P3_MAIN_AUTO_ARGS,
            prompts=prompts,
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
        "off_thrash_win16,main_auto_thrash_win16,"
        "main_auto_prunes,main_auto_drop_tokens,main_auto_auto_active_steps,"
        "main_auto_skip_no_candidate_blocks,main_auto_skip_trigger_not_met,main_auto_skip_min_drop_not_met,"
        "agree_first_32,agree_first_100,agree_after_128,agree_all,delta_avg_logprob"
    )
    for r in rows:
        print(
            f"{r['gpu_mem_frac']:.2f},{r['pressure_level']},{r['auto_trigger_expected']},{r['auto_trigger_observed']},"
            f"{r['non_trigger_reason']},"
            f"{r['off_tps']:.4f},{r['main_auto_tps']:.4f},{r['delta_tps_pct']:+.2f},"
            f"{r['off_p95_ms']:.3f},{r['main_auto_p95_ms']:.3f},{r['delta_p95_pct']:+.2f},"
            f"{r['off_thrash_win16']:.6f},{r['main_auto_thrash_win16']:.6f},"
            f"{r['main_auto_prunes']},{r['main_auto_drop_tokens']},{r['main_auto_auto_active_steps']},"
            f"{r['main_auto_skip_no_candidate_blocks']},{r['main_auto_skip_trigger_not_met']},{r['main_auto_skip_min_drop_not_met']},"
            f"{r['agree_first_32']:.4f},{r['agree_first_100']:.4f},{r['agree_after_128']:.4f},{r['agree_all']:.4f},"
            f"{r['delta_avg_logprob']:.6f}"
        )

    payload = {
        "meta": {
            "concurrency": CONCURRENCY,
            "prompt_repeat": PROMPT_REPEAT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "mem_fracs": MEM_FRACS,
            "groups": ["off", "main_auto"],
            "trigger_policy": "expected_at_high_pressure_but_not_hard_fail",
            "high_pressure_mem_fracs": [0.10, 0.05],
            "normal_mem_fracs": [0.30, 0.25],
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
            "alloc_conf_value": str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
            "metric_profile": METRIC_PROFILE,
            "prompt_profile": PROMPT_PROFILE,
            "compression_profile": compression_profile(dict(COMMON_BASE, **P3_MAIN_AUTO_ARGS)),
            "trigger_expectation_warnings": warnings,
        },
        "rows": rows,
    }
    with open("benchmark_p3_tiered_table_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\nSaved: benchmark_p3_tiered_table_results.json")


if __name__ == "__main__":
    main()
