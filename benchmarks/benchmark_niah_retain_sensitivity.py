import argparse
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

import benchmark_niah_single_limit as niah


RETENTION_SCAN = [1.00, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.10, 0.08, 0.05]
DEPTHS = [0.10, 0.50, 0.90]
PROFILES = [
    {"name": "window_16_16_recent512", "sink_len": 16, "obs_len": 16, "recent_len": 512},
    {"name": "window_32_32_recent512", "sink_len": 32, "obs_len": 32, "recent_len": 512},
    {"name": "window_64_64_recent512", "sink_len": 64, "obs_len": 64, "recent_len": 512},
]
MODES = ["off", "main_auto"]
TARGET_LEN = 32000
GPU_MEM_FRAC = 0.70
SEED = 20260323
CASE_TIMEOUT_S = 0
GPU_MEM_FRAC_FALLBACK_STEP = 0.02
GPU_MEM_FRAC_FALLBACK_TRIES = 4


def _compression_profile_str(profile: Dict, retain_ratio: float) -> str:
    return (
        f"sink={int(profile['sink_len'])};"
        f"obs={int(profile['obs_len'])};"
        f"retain={float(retain_ratio):.3f};"
        f"decode_sink={int(profile['sink_len'])};"
        f"decode_recent={int(profile['recent_len'])}"
    )


def _configure_for_profile(profile: Dict, retain_ratio: float):
    niah.COMMON_BASE["sink_len"] = int(profile["sink_len"])
    niah.COMMON_BASE["obs_len"] = int(profile["obs_len"])
    niah.COMMON_BASE["decode_window_sink_len"] = int(profile["sink_len"])
    niah.MODE_MAIN_AUTO_ARGS["decode_window_recent_len"] = int(profile["recent_len"])
    niah.PREFILL_COMPRESS_ARGS["retain_ratio"] = float(max(0.0, min(1.0, retain_ratio)))


def _run_setting(
    mode: str,
    profile: Dict,
    retain_ratio: float,
    target_len: int,
    seed: int,
    case_timeout_s: int,
    gpu_mem_frac: float,
) -> Dict:
    _configure_for_profile(profile, retain_ratio)
    engine = None
    depth_runs: List[Dict] = []
    stage = "retain_sensitivity"
    try:
        engine = niah._build_engine(float(gpu_mem_frac), mode, "compress")
    except Exception as exc:
        err = str(exc)
        summary = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "prefill_track": "compress",
            "stage": f"{stage}_summary",
            "target_len": int(target_len),
            "actual_prompt_tokens": 0,
            "depth": "mean(0.10,0.50,0.90)",
            "success": 0,
            "passkey_first_em": 0.0,
            "valid_rate": 0.0,
            "em_given_valid": 0.0,
            "passkey_first_em_compat": 0.0,
            "valid_rate_compat": 0.0,
            "passkey_contains_em": 0.0,
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
            "non_trigger_reason": "engine_build_failed",
            "error": err,
            "pred_strict": "",
            "pred_compat": "",
            "wall_ms": 0.0,
            "profile_name": str(profile["name"]),
            "retain_ratio": float(retain_ratio),
            "gpu_mem_frac": float(gpu_mem_frac),
            "pressure_level": "high" if float(gpu_mem_frac) <= 0.10 + 1e-9 else "normal",
            "auto_trigger_expected": int(float(gpu_mem_frac) <= 0.10 + 1e-9 and mode == "main_auto"),
            "auto_trigger_observed": 0,
            "metric_profile": "strict+compat",
            "prompt_profile": "chat_32",
            "compression_profile": _compression_profile_str(profile, retain_ratio),
            "analysis_note": "engine_build_failed",
        }
        return {"summary": summary, "depth_runs": depth_runs}
    for depth in DEPTHS:
        rec = niah._run_single_case(
            engine=engine,
            mode=mode,
            prefill_track="compress",
            target_len=int(target_len),
            depth=float(depth),
            seed=int(seed),
            stage=stage,
            case_timeout_s=int(case_timeout_s),
        )
        rec["profile_name"] = str(profile["name"])
        rec["retain_ratio"] = float(retain_ratio)
        rec["metric_profile"] = "strict+compat"
        rec["prompt_profile"] = "chat_32"
        rec["compression_profile"] = _compression_profile_str(profile, retain_ratio)
        depth_runs.append(rec)

    summary = niah._track_summary_rec(
        mode=mode,
        prefill_track="compress",
        stage=f"{stage}_summary",
        target_len=int(target_len),
        depth_runs=depth_runs,
    )
    summary["profile_name"] = str(profile["name"])
    summary["retain_ratio"] = float(retain_ratio)
    summary["gpu_mem_frac"] = float(gpu_mem_frac)
    summary["pressure_level"] = "high" if float(gpu_mem_frac) <= 0.10 + 1e-9 else "normal"
    summary["auto_trigger_expected"] = int(summary["pressure_level"] == "high" and mode == "main_auto")
    summary["auto_trigger_observed"] = int(summary.get("auto_trigger_observed", 0))
    summary["metric_profile"] = "strict+compat"
    summary["prompt_profile"] = "chat_32"
    summary["compression_profile"] = _compression_profile_str(profile, retain_ratio)
    summary["analysis_note"] = ""

    if summary["auto_trigger_expected"] == 1 and summary["auto_trigger_observed"] == 0:
        summary["analysis_note"] = "high_pressure_non_trigger"

    if engine is not None:
        del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"summary": summary, "depth_runs": depth_runs}


def _run_setting_with_fallback(
    mode: str,
    profile: Dict,
    retain_ratio: float,
    target_len: int,
    seed: int,
    case_timeout_s: int,
    gpu_mem_frac: float,
    fallback_step: float,
    fallback_tries: int,
) -> Dict:
    attempts: List[Dict] = []
    last = None
    for i in range(max(1, int(fallback_tries))):
        mem_frac_try = max(0.10, float(gpu_mem_frac) - float(i) * float(fallback_step))
        run = _run_setting(
            mode=mode,
            profile=profile,
            retain_ratio=retain_ratio,
            target_len=target_len,
            seed=seed,
            case_timeout_s=case_timeout_s,
            gpu_mem_frac=mem_frac_try,
        )
        summary = dict(run["summary"])
        summary["gpu_mem_frac"] = float(mem_frac_try)
        err_text = str(summary.get("error", "") or "")
        attempts.append(
            {
                "try_idx": int(i + 1),
                "gpu_mem_frac": float(mem_frac_try),
                "success": int(summary.get("success", 0)),
                "error": err_text,
            }
        )
        last = {"summary": summary, "depth_runs": run["depth_runs"], "attempts": attempts}
        if int(summary.get("success", 0)) == 1:
            break
        if err_text and ("cuda out of memory" in err_text.lower() or "outofmemory" in err_text.lower()):
            summary["analysis_note"] = "resource_limit_oom"
    return last if last is not None else {"summary": {}, "depth_runs": [], "attempts": attempts}


def _classify_note(row: Dict, current_em: float) -> str:
    err = str(row.get("error", "") or "").lower()
    if "out of memory" in err or "outofmemory" in err:
        return "resource_limit_oom"
    if str(row.get("non_trigger_reason", "")) == "engine_build_failed":
        return "engine_build_failed"
    em = float(row.get("passkey_first_em", 0.0))
    valid = float(row.get("valid_rate", 0.0))
    if valid < 1e-12:
        return "real_semantic_loss_or_format_collapse"
    if em + 1e-9 < current_em:
        return "likely_semantic_loss_under_more_aggressive_compression"
    return "mostly_protocol_aligned_or_no_visible_semantic_loss"


def main():
    parser = argparse.ArgumentParser(description="NIAH retention sensitivity with current vs SnapKV-like profiles.")
    parser.add_argument("--target-len", type=int, default=TARGET_LEN)
    parser.add_argument("--gpu-mem-frac", type=float, default=GPU_MEM_FRAC)
    parser.add_argument("--retain-ratios", type=str, default="1.00,0.90,0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.10,0.08,0.05")
    parser.add_argument("--profiles", type=str, default="")
    parser.add_argument("--modes", type=str, default="off")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--case-timeout-s", type=int, default=CASE_TIMEOUT_S)
    parser.add_argument("--gpu-mem-frac-fallback-step", type=float, default=GPU_MEM_FRAC_FALLBACK_STEP)
    parser.add_argument("--gpu-mem-frac-fallback-tries", type=int, default=GPU_MEM_FRAC_FALLBACK_TRIES)
    parser.add_argument("--out", type=str, default="benchmark_niah_retain_sensitivity_results.json")
    args = parser.parse_args()

    gpu_mem_frac = float(args.gpu_mem_frac)
    retain_scan = [float(x.strip()) for x in str(args.retain_ratios).split(",") if x.strip()]
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    selected_profiles = [p.strip() for p in str(args.profiles).split(",") if p.strip()]
    profiles = [p for p in PROFILES if not selected_profiles or p["name"] in selected_profiles]

    rows: List[Dict] = []
    depth_records: List[Dict] = []
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        for profile in profiles:
            for ratio in retain_scan:
                run = _run_setting_with_fallback(
                    mode=mode,
                    profile=profile,
                    retain_ratio=float(ratio),
                    target_len=int(args.target_len),
                    seed=int(args.seed),
                    case_timeout_s=int(args.case_timeout_s),
                    gpu_mem_frac=float(gpu_mem_frac),
                    fallback_step=float(max(0.0, args.gpu_mem_frac_fallback_step)),
                    fallback_tries=int(max(1, args.gpu_mem_frac_fallback_tries)),
                )
                summary = dict(run["summary"])
                summary["gpu_mem_frac_fallback_trace"] = list(run.get("attempts", []))
                rows.append(summary)
                depth_records.extend(run["depth_runs"])
                print(
                    json.dumps(
                        {
                            "ts": ts,
                            "mode": mode,
                            "profile_name": profile["name"],
                            "retain_ratio": float(ratio),
                            "target_len": int(args.target_len),
                            "passkey_first_em": float(summary.get("passkey_first_em", 0.0)),
                            "valid_rate": float(summary.get("valid_rate", 0.0)),
                            "passkey_first_em_compat": float(summary.get("passkey_first_em_compat", 0.0)),
                            "valid_rate_compat": float(summary.get("valid_rate_compat", 0.0)),
                            "auto_trigger_observed": int(summary.get("auto_trigger_observed", 0)),
                            "non_trigger_reason": str(summary.get("non_trigger_reason", "")),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    baselines: Dict[str, float] = {}
    for row in rows:
        if row["profile_name"] == "current_64_64_recent256":
            baselines[f"{row['mode']}|{row['retain_ratio']}"] = float(row.get("passkey_first_em", 0.0))
    for row in rows:
        key = f"{row['mode']}|{row['retain_ratio']}"
        current_em = float(baselines.get(key, float(row.get("passkey_first_em", 0.0))))
        row["analysis_note"] = _classify_note(row, current_em)

    curve_rows = []
    for mode in modes:
        for profile in [p["name"] for p in PROFILES]:
            subset = [r for r in rows if r["mode"] == mode and r["profile_name"] == profile]
            subset_sorted = sorted(subset, key=lambda x: -float(x["retain_ratio"]))
            curve_rows.append(
                {
                    "mode": mode,
                    "profile_name": profile,
                    "points": [
                        {
                            "retain_ratio": float(r["retain_ratio"]),
                            "passkey_first_em": float(r.get("passkey_first_em", 0.0)),
                            "valid_rate": float(r.get("valid_rate", 0.0)),
                            "passkey_first_em_compat": float(r.get("passkey_first_em_compat", 0.0)),
                            "valid_rate_compat": float(r.get("valid_rate_compat", 0.0)),
                            "non_trigger_reason": str(r.get("non_trigger_reason", "")),
                        }
                        for r in subset_sorted
                    ],
                }
            )

    payload = {
        "meta": {
            "task": "niah_retain_sensitivity_current_vs_snapkv_like",
            "timestamp": ts,
            "target_len": int(args.target_len),
            "depths": DEPTHS,
            "modes": modes,
            "gpu_mem_frac": float(gpu_mem_frac),
            "retain_ratios": retain_scan,
            "gpu_mem_frac_fallback_step": float(max(0.0, args.gpu_mem_frac_fallback_step)),
            "gpu_mem_frac_fallback_tries": int(max(1, args.gpu_mem_frac_fallback_tries)),
            "profiles": profiles,
            "metric_profile": "strict+compat",
            "prompt_profile": "chat_32",
            "strict_definition": niah.STRICT_DEFINITION,
            "compat_definition": "first 6-digit match anywhere in output",
            "alloc_conf_enabled": int(ALLOC_CONF_ENABLED),
            "alloc_conf_value": str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
            "conclusion_template": [
                "protocol_gap_or_prompt_format_gap",
                "real_semantic_loss_under_aggressive_compression",
            ],
        },
        "rows": rows,
        "curves": curve_rows,
        "depth_records": depth_records,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n===== RETAIN SENSITIVITY SUMMARY =====")
    print(
        "mode,profile,retain_ratio,passkey_first_em,valid_rate,passkey_first_em_compat,valid_rate_compat,auto_trigger_observed,non_trigger_reason,analysis_note"
    )
    for r in sorted(rows, key=lambda x: (x["mode"], x["profile_name"], -float(x["retain_ratio"]))):
        print(
            f"{r['mode']},{r['profile_name']},{float(r['retain_ratio']):.2f},"
            f"{float(r.get('passkey_first_em', 0.0)):.4f},{float(r.get('valid_rate', 0.0)):.4f},"
            f"{float(r.get('passkey_first_em_compat', 0.0)):.4f},{float(r.get('valid_rate_compat', 0.0)):.4f},"
            f"{int(r.get('auto_trigger_observed', 0))},{str(r.get('non_trigger_reason', ''))},{str(r.get('analysis_note', ''))}"
        )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
