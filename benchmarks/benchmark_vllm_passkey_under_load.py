import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

from benchmark_vllm_common import (
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_SERVER_HOST,
    VLLM_SERVED_MODEL_NAME,
    build_vllm_offload_extra_args,
    build_passkey_cases,
    first_6digit_compat,
    first_6digit_strict,
    load_tokenizer,
    max_prompt_tokens_for_request,
    resolve_vllm_max_model_len,
    run_prompt_batch,
    start_vllm_server,
    stop_vllm_server,
)


def parse_csv_ints(text: str) -> List[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return vals


def parse_csv_floats(text: str) -> List[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty float list")
    return vals


def iter_batches(cases: Sequence[Dict[str, Any]], concurrency: int) -> List[List[Dict[str, Any]]]:
    size = max(1, int(concurrency))
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


def aggregate_rows(
    input_len: int,
    concurrency: int,
    gpu_memory_utilization: float,
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
    total_wall_ms = sum(float(r["batch_wall_ms"]) for r in batch_rows)
    ttft_ms_vals = [float(r["ttft_ms"]) for r in batch_rows if float(r["ttft_ms"]) > 0]
    itl_vals = [float(r["avg_itl_ms"]) for r in batch_rows if float(r["avg_itl_ms"]) > 0]
    depth_totals: Dict[str, Dict[str, float]] = {}

    for row in batch_rows:
        for depth, stats in dict(row.get("by_depth", {}) or {}).items():
            bucket = depth_totals.setdefault(depth, {"n": 0.0, "em": 0.0, "valid": 0.0, "em_compat": 0.0, "valid_compat": 0.0})
            n = float(stats.get("n", 0))
            bucket["n"] += n
            bucket["em"] += float(stats.get("passkey_first_em", 0.0)) * n
            bucket["valid"] += float(stats.get("valid_rate", 0.0)) * n
            bucket["em_compat"] += float(stats.get("passkey_first_em_compat", 0.0)) * n
            bucket["valid_compat"] += float(stats.get("valid_rate_compat", 0.0)) * n

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
        "method": "vllm",
        "input_len": int(input_len),
        "concurrency": int(concurrency),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "success": int(not error_reason),
        "n_cases": int(total_cases),
        "completed": int(total_completed),
        "completion_rate": float(total_completed / max(1, total_cases)),
        "valid_rate": float(total_valid / max(1, total_cases)),
        "passkey_first_em": float(total_em / max(1, total_cases)),
        "valid_rate_compat": float(total_valid_compat / max(1, total_cases)),
        "passkey_first_em_compat": float(total_em_compat / max(1, total_cases)),
        "actual_prompt_tokens_mean": float(mean(int(x) for x in actual_prompt_tokens)) if actual_prompt_tokens else 0.0,
        "tokens_per_sec_global": float((1000.0 * total_generated_tokens / total_wall_ms) if total_wall_ms > 0 else 0.0),
        "ttft_p99_ms": float(percentile(ttft_ms_vals, 0.99)),
        "itl_p99_ms": float(percentile(itl_vals, 0.99)),
        "min_free_blocks": -1,
        "min_free_block_ratio": -1.0,
        "wall_clock_total_runtime_ms": float(round(total_wall_ms, 3)),
        "by_depth": by_depth,
        "error_reason": str(error_reason),
    }


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


def run_concurrency(
    base_url: str,
    tokenizer,
    input_len: int,
    max_new_tokens: int,
    depths: Sequence[float],
    keys_per_depth: int,
    concurrency: int,
    seed: int,
    request_timeout_s: float,
    gpu_memory_utilization: float,
    prompt_budget_tokens: int,
) -> Dict[str, Any]:
    cases = build_passkey_cases(
        tokenizer,
        input_len,
        depths,
        keys_per_depth,
        seed,
        max_prompt_tokens=int(prompt_budget_tokens),
    )
    actual_prompt_tokens = [int(c["actual_prompt_tokens"]) for c in cases]
    batches = iter_batches(cases, int(concurrency))
    batch_rows: List[Dict[str, Any]] = []

    for batch in batches:
        prompts = [str(x["prompt"]) for x in batch]
        t0 = time.perf_counter()
        outputs = run_prompt_batch(
            base_url=base_url,
            prompts=prompts,
            max_new_tokens=int(max_new_tokens),
            tokenizer=tokenizer,
            request_timeout_s=float(request_timeout_s),
        )
        batch_wall_ms = (time.perf_counter() - t0) * 1000.0
        texts = [str(x.get("output_text", "")) for x in outputs]
        quality = eval_outputs(texts, batch)
        ttft_vals = [float(x.get("ttft_ms", 0.0)) for x in outputs if float(x.get("ttft_ms", 0.0)) > 0]
        itl_vals = [float(x.get("avg_itl_ms", 0.0)) for x in outputs if float(x.get("avg_itl_ms", 0.0)) > 0]
        batch_rows.append(
            {
                "n_cases": int(len(batch)),
                "completed": int(round(float(quality["completion_rate"]) * len(batch))),
                "passkey_first_em": float(quality["passkey_first_em"]),
                "valid_rate": float(quality["valid_rate"]),
                "passkey_first_em_compat": float(quality["passkey_first_em_compat"]),
                "valid_rate_compat": float(quality["valid_rate_compat"]),
                "generated_tokens": int(sum(int(x.get("completion_tokens", 0)) for x in outputs)),
                "batch_wall_ms": float(round(batch_wall_ms, 3)),
                "ttft_ms": float(percentile(ttft_vals, 0.99)),
                "avg_itl_ms": float(percentile(itl_vals, 0.99)),
                "by_depth": dict(quality.get("by_depth", {})),
            }
        )

    return aggregate_rows(
        input_len=input_len,
        concurrency=concurrency,
        gpu_memory_utilization=gpu_memory_utilization,
        batch_rows=batch_rows,
        actual_prompt_tokens=actual_prompt_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="32K passkey under load for vLLM.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--input-len", type=int, default=32000)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--depths", type=str, default="0.1,0.5,0.9")
    parser.add_argument("--keys-per-depth", type=int, default=20)
    parser.add_argument("--concurrency-list", type=str, default="1,4,8,16")
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--server-log", type=str, default="")
    parser.add_argument("--swap-space", type=float, default=None)
    parser.add_argument("--cpu-offload-gb", type=float, default=None)
    parser.add_argument("--out", type=str, default="benchmark_vllm_passkey_under_load.json")
    args = parser.parse_args()

    depths = parse_csv_floats(args.depths)
    concurrencies = parse_csv_ints(args.concurrency_list)
    tokenizer = load_tokenizer(args.model_name)
    server_proc = None
    launched_server = False
    endpoint = str(args.base_url).strip()
    requested_max_model_len = int(args.input_len) + int(args.max_new_tokens) + 512
    effective_max_model_len = resolve_vllm_max_model_len(args.model_name, requested_max_model_len)
    prompt_budget_tokens = max_prompt_tokens_for_request(effective_max_model_len, int(args.max_new_tokens))
    if not endpoint:
        server_proc, port = start_vllm_server(
            model_name=args.model_name,
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=effective_max_model_len,
            host=DEFAULT_SERVER_HOST,
            log_path=Path(args.server_log) if str(args.server_log).strip() else None,
            extra_args=build_vllm_offload_extra_args(
                swap_space=args.swap_space,
                cpu_offload_gb=args.cpu_offload_gb,
            ),
        )
        endpoint = f"http://{DEFAULT_SERVER_HOST}:{int(port)}"
        launched_server = True

    rows: List[Dict[str, Any]] = []
    try:
        for concurrency in concurrencies:
            row = run_concurrency(
                base_url=endpoint,
                tokenizer=tokenizer,
                input_len=int(args.input_len),
                max_new_tokens=int(args.max_new_tokens),
                depths=depths,
                keys_per_depth=int(args.keys_per_depth),
                concurrency=int(concurrency),
                seed=int(args.seed),
                request_timeout_s=float(args.request_timeout_s),
                gpu_memory_utilization=float(args.gpu_memory_utilization),
                prompt_budget_tokens=int(prompt_budget_tokens),
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "concurrency": row["concurrency"],
                        "completion_rate": row["completion_rate"],
                        "valid_rate": row["valid_rate"],
                        "passkey_first_em": row["passkey_first_em"],
                        "ttft_p99_ms": row["ttft_p99_ms"],
                        "itl_p99_ms": row["itl_p99_ms"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if launched_server:
            stop_vllm_server(server_proc)

    payload = {
        "meta": {
            "task": "passkey_under_load_vllm",
            "model_name": str(args.model_name),
            "served_model_name": VLLM_SERVED_MODEL_NAME,
            "input_len": int(args.input_len),
            "max_new_tokens": int(args.max_new_tokens),
            "depths": [float(x) for x in depths],
            "keys_per_depth": int(args.keys_per_depth),
            "concurrency_list": [int(x) for x in concurrencies],
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "swap_space": None if args.swap_space is None else float(args.swap_space),
            "cpu_offload_gb": None if args.cpu_offload_gb is None else float(args.cpu_offload_gb),
            "requested_max_model_len": int(requested_max_model_len),
            "effective_max_model_len": int(effective_max_model_len),
            "prompt_budget_tokens": int(prompt_budget_tokens),
            "seed": int(args.seed),
            "report_order": ["completion rate", "valid rate", "EM"],
            "valid_definition": "request completed normally and the target answer field is parseable",
            "em_definition": "request completed normally and the first parsed answer exactly matches the target passkey",
            "ttft_definition": "request arrival to first streamed content chunk",
            "itl_definition": "per-request average interval between streamed content chunks; used as a practical proxy for inter-token latency",
            "min_free_blocks_note": "vLLM server path does not expose KV block pool metrics in this runner, so min_free_blocks is set to -1.",
        },
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

