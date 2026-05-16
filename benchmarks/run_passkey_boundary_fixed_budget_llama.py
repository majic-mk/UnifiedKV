#!/usr/bin/env python3
"""Binary-search Passkey max runnable context length between 32K and 64K.

This wrapper only invokes benchmark_quality_passkey_v3.py. It does not modify the
engine or benchmark core. A length is considered runnable when the full 11-depth
x 20-key sweep completes successfully.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
PY = "/root/miniconda3/bin/python"
MODEL = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
OUT_DIR = ROOT / "benchmarks/results/paper/quality_v7/passkey_boundary_fixed_budget_llama"
SEED_SUMMARY = ROOT / "benchmarks/results/paper/quality_v6/passkey_fixed_budget_adaptive_llama_smallpool/passkey_adaptive_summary.json"
METHOD_ORDER = [
    "off_compress_page16_b2048",
    "off_compress_page16_b4096",
    "off_compress_page16_b1024",
    "hf_vanilla",
]
GPU_MAP = "hf_vanilla:0.60,off_compress_page16_b1024:0.08,off_compress_page16_b2048:0.10,off_compress_page16_b4096:0.12"
DEPTHS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
KEYS_PER_DEPTH = 20
EXPECTED_CASES = 11 * KEYS_PER_DEPTH
KNOWN_SUCCESS = 32768
KNOWN_FAIL = 65536
STEP = 1024


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def safe_method(method: str) -> str:
    return method.replace("/", "_")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first_row(path: Path) -> Optional[Dict[str, Any]]:
    data = load_json(path)
    if not data:
        return None
    rows = data.get("rows") or []
    if not rows:
        return None
    return dict(rows[0])


def result_path(out_dir: Path, method: str, length: int) -> Path:
    return out_dir / f"passkey_boundary_{safe_method(method)}_L{int(length)}.json"


def row_completed(row: Dict[str, Any]) -> int:
    return int(row.get("completed", 0) or 0)


def row_n_cases(row: Dict[str, Any]) -> int:
    return int(row.get("n_cases", 0) or 0)


def row_em(row: Dict[str, Any]) -> float:
    return float(row.get("passkey_em", 0.0) or 0.0)


def is_success_row(row: Optional[Dict[str, Any]]) -> bool:
    if row is None:
        return False
    return (
        str(row.get("status", "")) == "Success"
        and row_completed(row) == EXPECTED_CASES
        and row_n_cases(row) == EXPECTED_CASES
        and not str(row.get("error_reason", "")).strip()
    )


def seed_known_entry(method: str, length: int) -> Optional[Dict[str, Any]]:
    data = load_json(SEED_SUMMARY)
    if not data:
        return None
    e = (((data.get("entries") or {}).get(method) or {}).get(str(length)))
    if not isinstance(e, dict):
        return None
    return dict(e)


def normalize_entry(method: str, length: int, path: Path, rc: int) -> Dict[str, Any]:
    row = first_row(path)
    if row is None:
        return {
            "source": "measured_failed",
            "method": method,
            "input_len": int(length),
            "result_file": str(path),
            "status": "Failed/OOM",
            "n_cases": 0,
            "completed": 0,
            "completion_rate": 0.0,
            "passkey_em": 0.0,
            "passkey_compat_em": 0.0,
            "error_reason": f"process_failed_rc={rc}; result_missing_or_unparseable",
        }
    ok = is_success_row(row)
    entry = {
        "source": "measured_success" if ok else "measured_failed",
        "method": method,
        "input_len": int(length),
        "result_file": str(path),
        "status": str(row.get("status", "")) if ok else "Failed/OOM",
        "n_cases": row_n_cases(row),
        "completed": row_completed(row),
        "completion_rate": float(row.get("completion_rate", 0.0) or 0.0),
        "passkey_em": row_em(row),
        "passkey_compat_em": float(row.get("passkey_compat_em", 0.0) or 0.0),
        "error_reason": str(row.get("error_reason", "")) or ("" if ok else f"incomplete_or_failed_rc={rc}"),
        "wall_clock_ms": float(row.get("wall_clock_ms", 0.0) or 0.0),
        "gpu_mem_frac": row.get("gpu_mem_frac"),
        "retain_budget_tokens": row.get("retain_budget_tokens"),
        "selected_writeback_enabled": int(row.get("selected_writeback_enabled", 0) or 0),
        "decode_page16_native_steps": int(row.get("decode_page16_native_steps", 0) or 0),
        "decode_rebuild_steps": int(row.get("decode_rebuild_steps", 0) or 0),
        "decode_materialize_kv_bytes": int(row.get("decode_materialize_kv_bytes", 0) or 0),
        "resident_miss_steps": int(row.get("resident_miss_steps", 0) or 0),
        "writeback_est_required_gb": row.get("writeback_est_required_gb"),
        "writeback_free_gb": row.get("writeback_free_gb"),
        "prefill_writeback_backend": row.get("prefill_writeback_backend"),
        "by_depth": row.get("by_depth", {}),
    }
    return entry


def run_passkey(out: Path, method: str, length: int, model: str, gpu_map: str, resume: bool) -> int:
    if resume and first_row(out) is not None:
        print(f"[{now()}] SKIP existing method={method} length={length} out={out}", flush=True)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        "benchmarks/benchmark_quality_passkey_v3.py",
        "--mode", "formal",
        "--model-name", model,
        "--methods", method,
        "--input-lens", str(int(length)),
        "--depths", DEPTHS,
        "--keys-per-depth", str(KEYS_PER_DEPTH),
        "--concurrency", "1",
        "--max-new-tokens", "32",
        "--gpu-mem-frac-map", gpu_map,
        "--out", str(out),
    ]
    env = os.environ.copy()
    env.pop("KV_MIDDLEWARE_DISABLE_SELECTED_WRITEBACK", None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("KV_MIDDLEWARE_BENCH_PROGRESS", "1")
    env.setdefault("KV_MIDDLEWARE_BENCH_FAIL_FAST_ON_FAILURE", "1")
    print(f"[{now()}] RUN method={method} length={length} out={out}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print(f"[{now()}] DONE method={method} length={length} rc={proc.returncode}", flush=True)
    return int(proc.returncode)


def write_summary(out_dir: Path, payload: Dict[str, Any]) -> None:
    path = out_dir / "passkey_boundary_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Passkey boundary binary-search runner.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--model-name", default=MODEL)
    ap.add_argument("--methods", default=",".join(METHOD_ORDER))
    ap.add_argument("--gpu-map", default=GPU_MAP)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    resume = not bool(args.no_resume)

    summary: Dict[str, Any] = {
        "meta": {
            "task": "passkey_boundary_binary_search",
            "model_name": str(args.model_name),
            "methods": methods,
            "known_success": KNOWN_SUCCESS,
            "known_fail": KNOWN_FAIL,
            "step_tokens": STEP,
            "depths": [float(x) for x in DEPTHS.split(",")],
            "keys_per_depth": KEYS_PER_DEPTH,
            "cases_per_length": EXPECTED_CASES,
            "concurrency": 1,
            "max_new_tokens": 32,
            "gpu_mem_frac_map": str(args.gpu_map),
            "fail_fast_on_first_failed_batch": True,
            "seed_summary": str(SEED_SUMMARY),
        },
        "methods": {},
        "table": [],
    }

    print(f"[{now()}] passkey boundary runner start out_dir={out_dir}", flush=True)
    for method in methods:
        lo = KNOWN_SUCCESS
        hi = KNOWN_FAIL
        attempts: List[Dict[str, Any]] = []
        seed_lo = seed_known_entry(method, KNOWN_SUCCESS)
        seed_hi = seed_known_entry(method, KNOWN_FAIL)
        if seed_lo:
            attempts.append({"source": "seed_success", **seed_lo})
        if seed_hi:
            attempts.append({"source": "seed_failed", **seed_hi})
        print(f"[{now()}] METHOD_START {method} lo={lo} hi={hi}", flush=True)
        while hi - lo > STEP:
            mid = ((lo + hi) // 2 // STEP) * STEP
            if mid <= lo:
                break
            path = result_path(out_dir, method, mid)
            rc = run_passkey(path, method, mid, str(args.model_name), str(args.gpu_map), resume)
            entry = normalize_entry(method, mid, path, rc)
            attempts.append(entry)
            if entry["source"] == "measured_success":
                lo = mid
                print(f"[{now()}] BOUNDARY_UPDATE method={method} length={mid} PASS em={100*float(entry.get('passkey_em',0.0)):.2f} new_lo={lo} hi={hi}", flush=True)
            else:
                hi = mid
                print(f"[{now()}] BOUNDARY_UPDATE method={method} length={mid} FAIL completed={entry.get('completed')}/{entry.get('n_cases')} new_lo={lo} hi={hi} reason={str(entry.get('error_reason',''))[:240]}", flush=True)
            # Persist partial summary after every length.
            best = None
            for e in attempts:
                if int(e.get("input_len", -1) or -1) == lo and str(e.get("source", "")).endswith("success"):
                    best = e
            summary["methods"][method] = {
                "boundary_len": lo,
                "next_failed_len": hi,
                "boundary_entry": best,
                "attempts": attempts,
            }
            write_summary(out_dir, summary)
        best = None
        for e in attempts:
            if int(e.get("input_len", -1) or -1) == lo and str(e.get("source", "")).endswith("success"):
                best = e
        boundary_em = float(best.get("passkey_em", 0.0) or 0.0) if best else None
        item = {
            "method": method,
            "boundary_len": lo,
            "boundary_label": f"{lo//1024}K",
            "boundary_em": round(100.0 * boundary_em, 2) if boundary_em is not None else "NA",
            "next_failed_len": hi,
            "next_failed_label": f"{hi//1024}K",
            "attempted_lengths": [int(e.get("input_len", 0) or 0) for e in attempts if e.get("input_len")],
        }
        summary["methods"][method] = {
            "boundary_len": lo,
            "next_failed_len": hi,
            "boundary_entry": best,
            "attempts": attempts,
        }
        summary["table"].append(item)
        write_summary(out_dir, summary)
        print(f"[{now()}] METHOD_DONE {method} boundary={lo} next_fail={hi} boundary_em={item['boundary_em']}", flush=True)
    print(f"[{now()}] passkey boundary runner done summary={out_dir / 'passkey_boundary_summary.json'}", flush=True)


if __name__ == "__main__":
    main()

