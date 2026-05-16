import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple


BENCH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_ROOT.parent
RESULT_DIR = PROJECT_ROOT / "results" / "experiment_results"
TIERED_JSON = BENCH_ROOT / "benchmark_p3_tiered_table_results.json"
QUALITY_JSON = BENCH_ROOT / "benchmark_p3_quality_passkey_results.json"
NIAH_SENS_JSON = BENCH_ROOT / "benchmark_niah_retain_sensitivity_results.json"


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def has_gpu() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return False


def load_rows(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("rows", []))


def _num_cols(rows: List[Dict]) -> List[str]:
    if not rows:
        return []
    cols = []
    for k, v in rows[0].items():
        if isinstance(v, (int, float)):
            cols.append(k)
    return cols


def _bucket_key(row: Dict) -> str:
    return f"{row.get('pressure_level', 'unknown')}|{row.get('gpu_mem_frac')}"


def _aggregate_numeric(rows_list: List[List[Dict]]) -> Dict[str, Dict[str, float]]:
    # keyed by pressure_level|gpu_mem_frac then metric
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for rows in rows_list:
        for r in rows:
            key = _bucket_key(r)
            if key not in buckets:
                buckets[key] = {}
            for c in _num_cols([r]):
                if c == "gpu_mem_frac":
                    continue
                buckets[key].setdefault(c, []).append(float(r[c]))

    out: Dict[str, Dict[str, float]] = {}
    for key, metric_map in buckets.items():
        out[key] = {}
        for metric, vals in metric_map.items():
            if not vals:
                continue
            out[key][f"{metric}_mean"] = float(sum(vals) / len(vals))
            out[key][f"{metric}_std"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
            out[key][f"{metric}_n"] = int(len(vals))
    return out


def _collect_high_pressure_non_trigger(rows_list: List[List[Dict]]) -> List[Dict]:
    out: List[Dict] = []
    for ridx, rows in enumerate(rows_list, start=1):
        for row in rows:
            if int(row.get("auto_trigger_expected", 0)) == 1 and int(row.get("auto_trigger_observed", 0)) == 0:
                out.append(
                    {
                        "repeat": int(ridx),
                        "gpu_mem_frac": row.get("gpu_mem_frac"),
                        "pressure_level": row.get("pressure_level"),
                        "non_trigger_reason": row.get("non_trigger_reason", ""),
                        "rerun_suggestion": {
                            "concurrency_steps": [8, 12, 16],
                            "max_new_tokens_steps": [512, 768, 1024],
                        },
                    }
                )
    return out


def _copy_json(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing result file: {src}")
    shutil.copy2(src, dst)


def run_unified(repeats: int, python_bin: str) -> Tuple[Path, Dict]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULT_DIR / f"paper_unified_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tiered_runs: List[List[Dict]] = []
    quality_runs: List[List[Dict]] = []

    for i in range(1, repeats + 1):
        print(f"\n===== REPEAT {i}/{repeats} : throughput/stability =====")
        run_cmd([python_bin, "benchmark_p3_tiered_table.py"], BENCH_ROOT)
        tiered_dst = out_dir / f"tiered_rep{i}.json"
        _copy_json(TIERED_JSON, tiered_dst)
        tiered_runs.append(load_rows(tiered_dst))

        print(f"\n===== REPEAT {i}/{repeats} : quality/passkey =====")
        run_cmd([python_bin, "benchmark_p3_quality_passkey.py"], BENCH_ROOT)
        quality_dst = out_dir / f"quality_rep{i}.json"
        _copy_json(QUALITY_JSON, quality_dst)
        quality_runs.append(load_rows(quality_dst))

    print("\n===== RETAIN SENSITIVITY (current vs snapkv-like) =====")
    run_cmd(
        [
            python_bin,
            "benchmark_niah_retain_sensitivity.py",
            "--modes",
            "off",
            "--gpu-mem-frac",
            "0.70",
            "--gpu-mem-frac-fallback-step",
            "0.02",
            "--gpu-mem-frac-fallback-tries",
            "6",
        ],
        BENCH_ROOT,
    )
    sens_dst = out_dir / "niah_retain_sensitivity.json"
    _copy_json(NIAH_SENS_JSON, sens_dst)
    sens_payload = json.loads(sens_dst.read_text(encoding="utf-8"))

    summary = {
        "meta": {
            "timestamp": ts,
            "repeats": int(repeats),
            "python_bin": python_bin,
            "groups": ["off", "main_auto"],
            "trigger_statement": "P3 is an auto safety valve: trigger is expected in high pressure, but not hard-required per bucket.",
            "table_1": "throughput_latency_stability_off_vs_main_auto",
            "table_2": "quality_strict_and_compat_off_vs_main_auto",
            "table_3": "retain_sensitivity_current_vs_snapkv_like",
        },
        "tiered_aggregate": _aggregate_numeric(tiered_runs),
        "quality_aggregate": _aggregate_numeric(quality_runs),
        "retain_sensitivity": sens_payload,
        "high_pressure_non_trigger_events": {
            "tiered": _collect_high_pressure_non_trigger(tiered_runs),
            "quality": _collect_high_pressure_non_trigger(quality_runs),
        },
    }
    summary_path = out_dir / "paper_unified_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir, summary


def main():
    parser = argparse.ArgumentParser(description="Run unified paper experiment matrix and aggregate results.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats for each benchmark.")
    parser.add_argument(
        "--python-bin",
        type=str,
        default=os.environ.get("PYTHON_BIN", "/root/miniconda3/bin/python"),
        help="Python executable used to run benchmark scripts.",
    )
    parser.add_argument(
        "--allow-no-gpu",
        action="store_true",
        help="Bypass GPU availability check (not recommended).",
    )
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("repeats must be > 0")

    if not args.allow_no_gpu and not has_gpu():
        raise RuntimeError("No GPU detected. Start a GPU instance then rerun this script.")

    out_dir, summary = run_unified(repeats=args.repeats, python_bin=args.python_bin)
    print("\n===== DONE =====")
    print(f"Output dir: {out_dir}")
    print(f"Summary keys: {list(summary.keys())}")


if __name__ == "__main__":
    main()
