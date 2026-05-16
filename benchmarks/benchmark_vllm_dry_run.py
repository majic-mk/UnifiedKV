import argparse
import json
from pathlib import Path

from benchmark_vllm_fixedpoint import run_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single vLLM dry run point.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--request-timeout-s", type=float, default=900)
    parser.add_argument("--server-log", type=str, default="")
    parser.add_argument("--swap-space", type=float, default=None)
    parser.add_argument("--cpu-offload-gb", type=float, default=None)
    parser.add_argument("--out", type=str, default="benchmark_vllm_dry_run.json")
    args = parser.parse_args()

    payload = run_case(
        model_name=str(args.model_name),
        input_len=int(args.input_len),
        concurrency=int(args.concurrency),
        max_new_tokens=int(args.max_new_tokens),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        repeats=1,
        base_url=str(args.base_url),
        request_timeout_s=float(args.request_timeout_s),
        log_path=str(args.server_log),
        swap_space=args.swap_space,
        cpu_offload_gb=args.cpu_offload_gb,
    )
    payload["task"] = "vllm_dry_run"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

