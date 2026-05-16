#!/usr/bin/env python3
"""Adaptive fixed-budget Passkey runner for UnifiedKV quality experiments.

Runs Passkey from long to short context lengths. For each method, once a
measured longer length reaches 100% strict EM with full completion, all shorter
lengths are recorded as inferred pass in the summary table without generating
raw rows.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
PY = "/root/miniconda3/bin/python"
MODEL = "/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
OUT_DIR = ROOT / "benchmarks/results/paper/quality_v5/passkey_fixed_budget_adaptive_llama"
METHODS = [
    "hf_vanilla",
    "off_compress_page16_b1024",
    "off_compress_page16_b2048",
    "off_compress_page16_b4096",
]
LENGTHS_DESC = [131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024]
LENGTHS_ASC = list(reversed(LENGTHS_DESC))
FULL_DEPTHS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
FULL_KEYS = 20
SMOKE_LENGTHS = "8192,32768"
SMOKE_DEPTHS = "0.0,0.5,1.0"
SMOKE_KEYS = 3
SMOKE_METHODS = "hf_vanilla,off_compress_page16_b2048"
GPU_MAP = "hf_vanilla:0.60,off_compress_page16_b1024:0.35,off_compress_page16_b2048:0.35,off_compress_page16_b4096:0.35"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _safe_method(method: str) -> str:
    return str(method).replace("/", "_")


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
    return out_dir / f"passkey_adaptive_{_safe_method(method)}_L{int(length)}.json"


def summary_path(out_dir: Path) -> Path:
    return out_dir / "passkey_adaptive_summary.json"


def row_n_cases(row: Dict[str, Any]) -> int:
    return int(row.get("n_cases", 0) or 0)


def row_completed(row: Dict[str, Any]) -> int:
    return int(row.get("completed", 0) or 0)


def row_em(row: Dict[str, Any]) -> float:
    return float(row.get("passkey_em", 0.0) or 0.0)


def row_is_measured_pass(row: Dict[str, Any], expected_cases: int) -> bool:
    return (
        str(row.get("status", "")) == "Success"
        and row_n_cases(row) == int(expected_cases)
        and row_completed(row) == int(expected_cases)
        and abs(row_em(row) - 1.0) < 1e-12
        and not str(row.get("error_reason", "")).strip()
    )


def telemetry_ok(method: str, row: Dict[str, Any]) -> bool:
    if method == "hf_vanilla":
        return True
    return (
        int(row.get("selected_writeback_enabled", 0) or 0) == 1
        and int(row.get("decode_rebuild_steps", 0) or 0) == 0
        and int(row.get("decode_materialize_kv_bytes", 0) or 0) == 0
        and int(row.get("resident_miss_steps", 0) or 0) == 0
    )


def run_passkey(
    out: Path,
    method: str,
    input_lens: str,
    depths: str,
    keys_per_depth: int,
    model: str,
    max_new_tokens: int,
    concurrency: int,
    gpu_map: str,
    resume: bool,
) -> int:
    if resume and first_row(out) is not None:
        print(f"[{_now()}] SKIP existing result method={method} input_lens={input_lens} out={out}", flush=True)
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        "benchmarks/benchmark_quality_passkey_v3.py",
        "--mode",
        "formal",
        "--model-name",
        model,
        "--methods",
        method,
        "--input-lens",
        input_lens,
        "--depths",
        depths,
        "--keys-per-depth",
        str(int(keys_per_depth)),
        "--concurrency",
        str(int(concurrency)),
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--gpu-mem-frac-map",
        gpu_map,
        "--out",
        str(out),
    ]
    env = os.environ.copy()
    env.pop("KV_MIDDLEWARE_DISABLE_SELECTED_WRITEBACK", None)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("KV_MIDDLEWARE_BENCH_PROGRESS", "1")
    env.setdefault("KV_MIDDLEWARE_BENCH_FAIL_FAST_ON_FAILURE", "1")
    print(f"[{_now()}] RUN method={method} input_lens={input_lens} depths={depths} keys={keys_per_depth} out={out}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print(f"[{_now()}] DONE method={method} input_lens={input_lens} rc={proc.returncode}", flush=True)
    return int(proc.returncode)


def summarize_entries(entries: Dict[str, Dict[int, Dict[str, Any]]], out_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    table: List[Dict[str, Any]] = []
    method_summaries: List[Dict[str, Any]] = []
    for method in METHODS:
        method_entries = entries.get(method, {})
        item: Dict[str, Any] = {"method": method}
        measured_lengths: List[int] = []
        inferred_lengths: List[int] = []
        failed_lengths: List[int] = []
        pending_lengths: List[int] = []
        for length in LENGTHS_ASC:
            e = method_entries.get(length)
            if not e:
                item[str(length)] = "Pending"
                pending_lengths.append(length)
                continue
            source = str(e.get("source", ""))
            if source == "inferred":
                item[str(length)] = "100.0*"
                inferred_lengths.append(length)
            elif source == "measured":
                measured_lengths.append(length)
                item[str(length)] = round(100.0 * float(e.get("passkey_em", 0.0) or 0.0), 2)
            elif source == "failed":
                item[str(length)] = "Failed/OOM"
                failed_lengths.append(length)
            else:
                item[str(length)] = "Pending"
                pending_lengths.append(length)
        measured_vals = [float(e.get("passkey_em", 0.0) or 0.0) for e in method_entries.values() if e.get("source") == "measured"]
        item["Avg_measured"] = round(100.0 * sum(measured_vals) / len(measured_vals), 2) if measured_vals else "NA"
        item["Coverage"] = f"{len(measured_lengths) + len(inferred_lengths)}/{len(LENGTHS_ASC)}"
        table.append(item)
        stop_reason = "pending"
        inferred_from = None
        for e in method_entries.values():
            if e.get("source") == "inferred":
                stop_reason = "inferred_shorter_lengths"
                inferred_from = int(e.get("inferred_from_length", 0) or 0)
                break
        if not pending_lengths and not inferred_lengths:
            stop_reason = "exhausted_all_lengths"
        method_summaries.append({
            "method": method,
            "measured_lengths": sorted(measured_lengths),
            "inferred_lengths": sorted(inferred_lengths),
            "failed_lengths": sorted(failed_lengths),
            "pending_lengths": sorted(pending_lengths),
            "stop_reason": stop_reason,
            "inferred_from_length": inferred_from,
        })
    payload = {
        "meta": meta,
        "table_note": "* indicates inferred pass for shorter contexts after the same method achieved measured 100% EM at a longer context length under the full depth sweep.",
        "table": table,
        "method_summaries": method_summaries,
        "entries": {m: {str(k): v for k, v in sorted(vv.items())} for m, vv in entries.items()},
    }
    summary_path(out_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_existing_entries(out_dir: Path) -> Dict[str, Dict[int, Dict[str, Any]]]:
    entries: Dict[str, Dict[int, Dict[str, Any]]] = {m: {} for m in METHODS}
    for method in METHODS:
        for length in LENGTHS_DESC:
            path = result_path(out_dir, method, length)
            row = first_row(path)
            if not row:
                continue
            expected = 11 * FULL_KEYS
            if str(row.get("status", "")) == "Success" and row_completed(row) == expected and row_n_cases(row) == expected:
                source = "measured"
            else:
                source = "failed"
            entries[method][length] = {
                "source": source,
                "input_len": int(length),
                "method": method,
                "result_file": str(path),
                "status": str(row.get("status", "")),
                "n_cases": row_n_cases(row),
                "completed": row_completed(row),
                "completion_rate": float(row.get("completion_rate", 0.0) or 0.0),
                "passkey_em": row_em(row),
                "passkey_compat_em": float(row.get("passkey_compat_em", 0.0) or 0.0),
                "error_reason": str(row.get("error_reason", "")),
                "telemetry_ok": telemetry_ok(method, row),
            }
    return entries


def record_row(entries: Dict[str, Dict[int, Dict[str, Any]]], method: str, length: int, path: Path, rc: int) -> Dict[str, Any]:
    expected = 11 * FULL_KEYS
    row = first_row(path)
    if row is None:
        e = {
            "source": "failed",
            "input_len": int(length),
            "method": method,
            "result_file": str(path),
            "status": "Failed/OOM",
            "n_cases": 0,
            "completed": 0,
            "completion_rate": 0.0,
            "passkey_em": 0.0,
            "passkey_compat_em": 0.0,
            "error_reason": f"process_failed_rc={rc}; result_missing_or_unparseable",
            "telemetry_ok": method == "hf_vanilla",
        }
    else:
        completed = row_completed(row)
        n_cases = row_n_cases(row)
        ok_status = str(row.get("status", "")) == "Success" and completed == expected and n_cases == expected and not str(row.get("error_reason", "")).strip()
        e = {
            "source": "measured" if ok_status else "failed",
            "input_len": int(length),
            "method": method,
            "result_file": str(path),
            "status": str(row.get("status", "")) if ok_status else "Failed/OOM",
            "n_cases": n_cases,
            "completed": completed,
            "completion_rate": float(row.get("completion_rate", 0.0) or 0.0),
            "passkey_em": row_em(row),
            "passkey_compat_em": float(row.get("passkey_compat_em", 0.0) or 0.0),
            "error_reason": str(row.get("error_reason", "")) or ("" if ok_status else f"incomplete_or_failed_rc={rc}"),
            "telemetry_ok": telemetry_ok(method, row),
            "selected_writeback_enabled": int(row.get("selected_writeback_enabled", 0) or 0),
            "decode_rebuild_steps": int(row.get("decode_rebuild_steps", 0) or 0),
            "decode_materialize_kv_bytes": int(row.get("decode_materialize_kv_bytes", 0) or 0),
            "resident_miss_steps": int(row.get("resident_miss_steps", 0) or 0),
            "by_depth": row.get("by_depth", {}),
        }
    entries.setdefault(method, {})[int(length)] = e
    return e


def infer_shorter(entries: Dict[str, Dict[int, Dict[str, Any]]], method: str, source_length: int) -> None:
    for length in LENGTHS_DESC:
        if int(length) >= int(source_length):
            continue
        entries.setdefault(method, {})[int(length)] = {
            "source": "inferred",
            "input_len": int(length),
            "method": method,
            "status": "Inferred/Pass",
            "passkey_em": 1.0,
            "passkey_compat_em": 1.0,
            "inferred_from_length": int(source_length),
            "n_cases": 0,
            "completed": 0,
            "completion_rate": 1.0,
            "telemetry_ok": True,
        }


def verify_smoke(smoke_path: Path) -> None:
    data = load_json(smoke_path)
    if not data:
        raise RuntimeError(f"smoke result missing: {smoke_path}")
    rows = data.get("rows") or []
    if not rows:
        raise RuntimeError("smoke result has no rows")
    for row in rows:
        method = str(row.get("method", ""))
        if str(row.get("status", "")) != "Success":
            raise RuntimeError(f"smoke failed for {method} L={row.get('input_len')}: {row.get('error_reason')}")
        if method != "hf_vanilla" and not telemetry_ok(method, row):
            raise RuntimeError(f"smoke telemetry invalid for {method} L={row.get('input_len')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Adaptive fixed-budget Passkey runner.")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--model-name", default=MODEL)
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--gpu-map", default=GPU_MAP, help="Comma-separated method:gpu_mem_frac map for this run.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unsupported methods for adaptive plan: {unknown}")
    resume = not bool(args.no_resume)
    meta = {
        "task": "passkey_adaptive_fixed_budget",
        "model_name": str(args.model_name),
        "methods": methods,
        "length_order": LENGTHS_DESC,
        "depths": [float(x) for x in FULL_DEPTHS.split(",")],
        "keys_per_depth": FULL_KEYS,
        "cases_per_measured_length": 11 * FULL_KEYS,
        "concurrency": 1,
        "max_new_tokens": 32,
        "gpu_mem_frac_map": str(args.gpu_map),
        "adaptive_rule": "run lengths from 128K to 1K; if a measured longer length reaches 100% strict EM with full completion, mark all shorter lengths as inferred 100.0* for that method",
        "fail_fast_on_first_failed_batch": True,
    }

    print(f"[{_now()}] adaptive passkey runner start out_dir={out_dir}", flush=True)
    if not args.skip_smoke:
        smoke_path = out_dir / "passkey_smoke_hf_b2048.json"
        rc = run_passkey(
            smoke_path,
            SMOKE_METHODS,
            SMOKE_LENGTHS,
            SMOKE_DEPTHS,
            SMOKE_KEYS,
            str(args.model_name),
            32,
            1,
            str(args.gpu_map),
            resume,
        )
        if rc != 0 and first_row(smoke_path) is None:
            raise RuntimeError(f"smoke command failed rc={rc}")
        verify_smoke(smoke_path)
        print(f"[{_now()}] smoke verified", flush=True)

    entries = load_existing_entries(out_dir)
    summarize_entries(entries, out_dir, meta)
    for method in methods:
        print(f"[{_now()}] METHOD_START {method}", flush=True)
        inferred = any(e.get("source") == "inferred" for e in entries.get(method, {}).values())
        if inferred:
            print(f"[{_now()}] METHOD_SKIP inferred entries already present method={method}", flush=True)
            continue
        for length in LENGTHS_DESC:
            if length in entries.get(method, {}) and entries[method][length].get("source") == "inferred":
                print(f"[{_now()}] STOP shorter lengths already inferred method={method} from={entries[method][length].get('inferred_from_length')}", flush=True)
                break
            path = result_path(out_dir, method, length)
            rc = run_passkey(
                path,
                method,
                str(int(length)),
                FULL_DEPTHS,
                FULL_KEYS,
                str(args.model_name),
                32,
                1,
                str(args.gpu_map),
                resume,
            )
            e = record_row(entries, method, length, path, rc)
            summarize_entries(entries, out_dir, meta)
            print(
                f"[{_now()}] RESULT method={method} length={length} source={e.get('source')} status={e.get('status')} em={100.0*float(e.get('passkey_em',0.0) or 0.0):.2f} completed={e.get('completed')}/{e.get('n_cases')} telemetry_ok={e.get('telemetry_ok')}",
                flush=True,
            )
            if row_is_measured_pass(e, 11 * FULL_KEYS):
                infer_shorter(entries, method, int(length))
                summarize_entries(entries, out_dir, meta)
                print(f"[{_now()}] INFER_SHORTER method={method} source_length={length}", flush=True)
                break
        print(f"[{_now()}] METHOD_DONE {method}", flush=True)
    summarize_entries(entries, out_dir, meta)
    print(f"[{_now()}] adaptive passkey runner done summary={summary_path(out_dir)}", flush=True)


if __name__ == "__main__":
    main()
