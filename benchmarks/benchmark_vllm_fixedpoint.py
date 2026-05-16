import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from benchmark_vllm_common import (
    DEFAULT_REQUEST_TIMEOUT_S,
    DEFAULT_SERVER_HOST,
    VLLM_SERVED_MODEL_NAME,
    aggregate_streaming_metrics,
    build_vllm_offload_extra_args,
    build_synthetic_prompts_for_cell,
    load_tokenizer,
    max_prompt_tokens_for_request,
    resolve_vllm_max_model_len,
    run_prompt_batch,
    start_vllm_server,
    stop_vllm_server,
)


def valid_synthetic_output(text: str) -> bool:
    return bool(str(text).strip())


def run_case(
    model_name: str,
    input_len: int,
    concurrency: int,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    repeats: int,
    base_url: str = "",
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    log_path: str = "",
    swap_space: float | None = None,
    cpu_offload_gb: float | None = None,
) -> Dict[str, Any]:
    tokenizer = load_tokenizer(model_name)
    server_proc = None
    launched_server = False
    endpoint = str(base_url).strip()
    requested_max_model_len = int(input_len) + int(max_new_tokens) + 512
    effective_max_model_len = resolve_vllm_max_model_len(model_name, requested_max_model_len)
    prompt_budget_tokens = max_prompt_tokens_for_request(effective_max_model_len, int(max_new_tokens))
    prompts, actual_prompt_tokens = build_synthetic_prompts_for_cell(
        tokenizer,
        int(input_len),
        int(concurrency),
        max_prompt_tokens=int(prompt_budget_tokens),
    )
    if not endpoint:
        try:
            server_proc, port = start_vllm_server(
                model_name=model_name,
                gpu_memory_utilization=float(gpu_memory_utilization),
                max_model_len=effective_max_model_len,
                host=DEFAULT_SERVER_HOST,
                log_path=Path(log_path) if str(log_path).strip() else None,
                extra_args=build_vllm_offload_extra_args(
                    swap_space=swap_space,
                    cpu_offload_gb=cpu_offload_gb,
                ),
            )
            endpoint = f"http://{DEFAULT_SERVER_HOST}:{int(port)}"
            launched_server = True
        except Exception as exc:
            return {
                "task": "vllm_fixedpoint",
                "model_name": str(model_name),
                "served_model_name": VLLM_SERVED_MODEL_NAME,
                "input_len": int(input_len),
                "concurrency": int(concurrency),
                "max_new_tokens": int(max_new_tokens),
                "gpu_memory_utilization": float(gpu_memory_utilization),
                "swap_space": None if swap_space is None else float(swap_space),
                "cpu_offload_gb": None if cpu_offload_gb is None else float(cpu_offload_gb),
                "requested_max_model_len": int(requested_max_model_len),
                "effective_max_model_len": int(effective_max_model_len),
                "prompt_budget_tokens": int(prompt_budget_tokens),
                "repeats": int(repeats),
                "actual_prompt_tokens_mean": float(sum(actual_prompt_tokens) / max(1, len(actual_prompt_tokens))),
                "total_generated_tokens": 0,
                "status": "Failed/OOM",
                "completion_rate": 0.0,
                "valid_completion_rate": 0.0,
                "oom_failure_count": 1,
                "tokens_per_sec": 0.0,
                "ttft_p95_ms": 0.0,
                "ttft_p99_ms": 0.0,
                "itl_p95_ms": 0.0,
                "itl_p99_ms": 0.0,
                "min_free_blocks": -1,
                "min_free_block_ratio": -1.0,
                "wall_clock_total_runtime_ms": 0.0,
                "error_reason": f"server_start_failed: {exc}",
                "meta_note": "vLLM server path does not expose KV block pool metrics in this runner, so min_free_blocks and min_free_block_ratio are set to -1.",
                "rows": [],
            }

    request_rows: List[Dict[str, Any]] = []
    batch_wall_ms_list: List[float] = []
    try:
        for repeat_idx in range(int(repeats)):
            t0 = time.perf_counter()
            batch = run_prompt_batch(
                base_url=endpoint,
                prompts=prompts,
                max_new_tokens=int(max_new_tokens),
                tokenizer=tokenizer,
                request_timeout_s=float(request_timeout_s),
            )
            batch_wall_ms_list.append((time.perf_counter() - t0) * 1000.0)
            for item in batch:
                item["repeat_idx"] = int(repeat_idx)
            request_rows.extend(batch)
    finally:
        if launched_server:
            stop_vllm_server(server_proc)

    agg = aggregate_streaming_metrics(
        request_rows=request_rows,
        requested_repeats=int(repeats),
        valid_fn=valid_synthetic_output,
        batch_wall_ms_list=batch_wall_ms_list,
    )
    total_generated_tokens = sum(int(row.get("completion_tokens", 0)) for row in request_rows)
    return {
        "task": "vllm_fixedpoint",
        "model_name": str(model_name),
        "served_model_name": VLLM_SERVED_MODEL_NAME,
        "input_len": int(input_len),
        "concurrency": int(concurrency),
        "max_new_tokens": int(max_new_tokens),
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "swap_space": None if swap_space is None else float(swap_space),
        "cpu_offload_gb": None if cpu_offload_gb is None else float(cpu_offload_gb),
        "requested_max_model_len": int(requested_max_model_len),
        "effective_max_model_len": int(effective_max_model_len),
        "prompt_budget_tokens": int(prompt_budget_tokens),
        "repeats": int(repeats),
        "actual_prompt_tokens_mean": float(sum(actual_prompt_tokens) / max(1, len(actual_prompt_tokens))),
        "total_generated_tokens": int(total_generated_tokens),
        "status": agg["status"],
        "completion_rate": agg["completion_rate"],
        "valid_completion_rate": agg["valid_completion_rate"],
        "oom_failure_count": agg["oom_failure_count"],
        "tokens_per_sec": agg["tokens_per_sec"],
        "ttft_p95_ms": agg["ttft_p95_ms"],
        "ttft_p99_ms": agg["ttft_p99_ms"],
        "itl_p95_ms": agg["itl_p95_ms"],
        "itl_p99_ms": agg["itl_p99_ms"],
        "min_free_blocks": agg["min_free_blocks"],
        "min_free_block_ratio": agg["min_free_block_ratio"],
        "wall_clock_total_runtime_ms": agg["wall_clock_total_runtime_ms"],
        "error_reason": agg["error_reason"],
        "meta_note": "vLLM server path does not expose KV block pool metrics in this runner, so min_free_blocks and min_free_block_ratio are set to -1.",
        "rows": request_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single vLLM synthetic fixed-point evaluation.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--server-log", type=str, default="")
    parser.add_argument("--swap-space", type=float, default=None)
    parser.add_argument("--cpu-offload-gb", type=float, default=None)
    parser.add_argument("--out", type=str, default="benchmark_vllm_fixedpoint.json")
    args = parser.parse_args()

    payload = run_case(
        model_name=str(args.model_name),
        input_len=int(args.input_len),
        concurrency=int(args.concurrency),
        max_new_tokens=int(args.max_new_tokens),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        repeats=int(args.repeats),
        base_url=str(args.base_url),
        request_timeout_s=float(args.request_timeout_s),
        log_path=str(args.server_log),
        swap_space=args.swap_space,
        cpu_offload_gb=args.cpu_offload_gb,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

