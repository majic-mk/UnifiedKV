import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_int_list(s: str) -> List[int]:
    vals = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return vals


def load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows", []))


def run_candidate(script_path: Path, out_prefix: Path, args: argparse.Namespace, max_prefill_active: int, prefill_batch_size: int, prefill_token_budget_per_step: int) -> Dict[str, Any]:
    prefix = out_prefix.parent / f"{out_prefix.name}_mpa{max_prefill_active}_pbs{prefill_batch_size}_ptb{prefill_token_budget_per_step}"
    cmd = [
        sys.executable,
        str(script_path.parent / "benchmark_p2_online_mixed_flow.py"),
        "--groups", "off_compress,main_auto_compress",
        "--content-source", args.content_source,
        "--length-spec", args.length_spec,
        "--gpu-mem-frac", str(float(args.gpu_mem_frac)),
        "--max-new-tokens", str(int(args.max_new_tokens)),
        "--decode-micro-batch-size", str(int(args.decode_micro_batch_size)),
        "--decode-active-cap-initial", str(int(args.decode_active_cap_initial)),
        "--max-decode-active-cap", str(int(args.max_decode_active_cap)),
        "--max-prefill-active", str(int(max_prefill_active)),
        "--prefill-batch-size", str(int(prefill_batch_size)),
        "--prefill-token-budget-per-step", str(int(prefill_token_budget_per_step)),
        "--initial-submit", str(int(args.initial_submit)),
        "--arrival-batch", str(int(args.arrival_batch)),
        "--arrival-interval-decode-steps", str(int(args.arrival_interval_decode_steps)),
        "--out-prefix", str(prefix),
    ]
    if args.longbench_path:
        cmd.extend(["--longbench-path", args.longbench_path])
    if args.ruler_path:
        cmd.extend(["--ruler-path", args.ruler_path])
    subprocess.run(cmd, cwd=str(script_path.parent), check=False)
    result_path = prefix.with_name(prefix.name + "_results.json")
    rows = load_rows(result_path)
    target = next((r for r in rows if str(r.get("group")) == "main_auto_compress"), rows[0] if rows else {})
    target.update({
        "candidate_max_prefill_active": int(max_prefill_active),
        "candidate_prefill_batch_size": int(prefill_batch_size),
        "candidate_prefill_token_budget_per_step": int(prefill_token_budget_per_step),
    })
    return target


def dominates(a: Dict[str, Any], b: Dict[str, Any], p95_ceiling: float) -> bool:
    if float(a.get("decode_step_p95_ms", 1e18)) > p95_ceiling:
        return False
    if float(b.get("decode_step_p95_ms", 1e18)) > p95_ceiling:
        return True
    return (
        float(a.get("tokens_per_sec", 0.0)) >= float(b.get("tokens_per_sec", 0.0))
        and float(a.get("decode_step_p95_ms", 1e18)) <= float(b.get("decode_step_p95_ms", 1e18))
        and (
            float(a.get("tokens_per_sec", 0.0)) > float(b.get("tokens_per_sec", 0.0))
            or float(a.get("decode_step_p95_ms", 1e18)) < float(b.get("decode_step_p95_ms", 1e18))
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduler tuning grid for online mixed flow")
    parser.add_argument("--content-source", type=str, default="synthetic")
    parser.add_argument("--longbench-path", type=str, default="")
    parser.add_argument("--ruler-path", type=str, default="")
    parser.add_argument("--length-spec", type=str, default="8192:16,16384:16,32768:16")
    parser.add_argument("--gpu-mem-frac", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--decode-micro-batch-size", type=int, default=16)
    parser.add_argument("--decode-active-cap-initial", type=int, default=16)
    parser.add_argument("--max-decode-active-cap", type=int, default=16)
    parser.add_argument("--initial-submit", type=int, default=16)
    parser.add_argument("--arrival-batch", type=int, default=4)
    parser.add_argument("--arrival-interval-decode-steps", type=int, default=8)
    parser.add_argument("--max-prefill-active-list", type=str, default="8,16,24,32")
    parser.add_argument("--prefill-batch-size-list", type=str, default="4,8,16")
    parser.add_argument("--prefill-token-budget-list", type=str, default="8192,16384,24576,32768")
    parser.add_argument("--out-prefix", type=str, default="benchmark_scheduler_tuning")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    out_prefix = Path(args.out_prefix)
    candidates: List[Dict[str, Any]] = []
    baseline: Dict[str, Any] = {}
    for mpa in parse_int_list(args.max_prefill_active_list):
        for pbs in parse_int_list(args.prefill_batch_size_list):
            for ptb in parse_int_list(args.prefill_token_budget_list):
                rec = run_candidate(script_path, out_prefix, args, mpa, pbs, ptb)
                candidates.append(rec)
                if not baseline:
                    baseline = rec
                print(json.dumps({
                    "max_prefill_active": mpa,
                    "prefill_batch_size": pbs,
                    "prefill_token_budget_per_step": ptb,
                    "tokens_per_sec": rec.get("tokens_per_sec", 0.0),
                    "decode_step_p95_ms": rec.get("decode_step_p95_ms", 0.0),
                    "success_rate": rec.get("success_rate", 0.0),
                }, ensure_ascii=False), flush=True)
    p95_ceiling = float(baseline.get("decode_step_p95_ms", 1e18)) * 1.25 if baseline else 1e18
    pareto = []
    for cand in candidates:
        if float(cand.get("success_rate", 0.0)) < 0.999:
            continue
        if int(cand.get("decode_retry_timeout_fail_count", 0)) > 0 or int(cand.get("decode_no_progress_steps", 0)) > 0:
            continue
        if any(dominates(other, cand, p95_ceiling) for other in candidates if other is not cand):
            continue
        pareto.append(cand)
    payload = {
        "meta": {
            "task": "scheduler_tuning",
            "length_spec": args.length_spec,
            "gpu_mem_frac": float(args.gpu_mem_frac),
            "p95_ceiling": float(p95_ceiling),
        },
        "baseline": baseline,
        "candidates": candidates,
        "pareto": pareto,
    }
    out_json = out_prefix.with_name(out_prefix.name + "_results.json")
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
