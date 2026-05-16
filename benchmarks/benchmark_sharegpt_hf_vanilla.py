import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
CONFIGS_DIR = BENCHMARKS_DIR / "configs"
CORE_DIR = PROJECT_ROOT / "core"
for path in (BENCHMARKS_DIR, CONFIGS_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_internal_common import is_hf_style_method, parse_methods
from benchmark_sharegpt_serving import (
    eval_sharegpt_output,
    load_records,
    resolve_dataset_path,
    scan_sharegpt_candidates,
    select_sharegpt_samples,
    summarize_int_distribution,
)
from benchmark_vllm_common import load_tokenizer
from hf_style_common import run_hf_style_prompt_batches


def summarize_samples(samples: Sequence[Dict[str, Any]], candidate_count: int) -> Dict[str, Any]:
    prompts = [int(x["prompt_tokens"]) for x in samples]
    targets = [int(x["target_tokens"]) for x in samples]
    return {
        "candidate_count": int(candidate_count),
        "sample_count": int(len(samples)),
        "prompt_tokens": summarize_int_distribution(prompts),
        "target_tokens": summarize_int_distribution(targets),
        "sample_signature": [
            {
                "record_idx": int(x["record_idx"]),
                "turn_idx": int(x["turn_idx"]),
                "prompt_tokens": int(x["prompt_tokens"]),
                "target_tokens": int(x["target_tokens"]),
            }
            for x in samples
        ],
    }


def build_payload(
    args: argparse.Namespace,
    dataset_path: Path,
    sample_source: str,
    sample_summary: Dict[str, Any],
    methods: Sequence[str],
    concurrencies: Sequence[int],
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "meta": {
            "task": "sharegpt_hf_style_batch",
            "serving_mode": "chunked_batch",
            "model_name": args.model_name,
            "dataset": str(dataset_path),
            "methods": list(methods),
            "sample_count": int(sample_summary.get("sample_count", 0)),
            "requested_sample_count": int(args.sample_count),
            "concurrency_list": [int(x) for x in concurrencies],
            "prompt_len_range": [int(args.prompt_len_min), int(args.prompt_len_max)],
            "target_len_range": [int(args.target_len_min), int(args.target_len_max)],
            "max_new_tokens": int(args.max_new_tokens),
            "repeats": int(args.repeats),
            "seed": int(args.seed),
            "sample_source": str(sample_source),
            "sample_summary": sample_summary,
            "note": (
                "HF-style batch runner. hf_vanilla uses standard Transformers generate(); "
                "off_raw requires strict block-only direct decode and is marked Unsupported when unavailable."
            ),
        },
        "rows": list(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ShareGPT HF-style batch runner with per-cell checkpoint writes.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", default="hf_vanilla")
    parser.add_argument("--sample-count", type=int, default=153)
    parser.add_argument("--concurrency-list", default="1")
    parser.add_argument("--prompt-len-min", type=int, default=4096)
    parser.add_argument("--prompt-len-max", type=int, default=32768)
    parser.add_argument("--target-len-min", type=int, default=128)
    parser.add_argument("--target-len-max", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--samples-json", default="")
    parser.add_argument("--save-samples-json", default="")
    parser.add_argument("--out", default="benchmarks/results/probes/sharegpt_hf_style/result.json")
    args = parser.parse_args()

    methods = parse_methods(args.methods, allow_vllm=False)
    for method in methods:
        if not is_hf_style_method(method):
            raise ValueError(f"benchmark_sharegpt_hf_vanilla only supports hf-style methods, got: {method}")
    concurrencies = [int(x.strip()) for x in str(args.concurrency_list).split(",") if str(x).strip()]
    if not concurrencies:
        raise ValueError("empty concurrency_list")

    sample_source = "scan"
    dataset_path = resolve_dataset_path(args.dataset)
    tokenizer = load_tokenizer(args.model_name)
    if str(args.samples_json).strip():
        sample_path = Path(str(args.samples_json))
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            samples = [dict(x) for x in payload.get("samples", [])]
            candidate_count = int(payload.get("candidate_count", len(samples)))
        else:
            samples = [dict(x) for x in payload]
            candidate_count = len(samples)
        if not samples:
            raise RuntimeError(f"empty samples_json: {sample_path}")
        sample_source = str(sample_path)
    else:
        records = load_records(dataset_path)
        candidates = scan_sharegpt_candidates(
            tokenizer,
            records,
            int(args.prompt_len_min),
            int(args.prompt_len_max),
            int(args.target_len_min),
            int(args.target_len_max),
            progress_prefix="hf_style_sharegpt",
            progress_every=5000,
        )
        candidate_count = len(candidates)
        if len(candidates) < int(args.sample_count):
            raise RuntimeError(
                f"insufficient ShareGPT candidates: {len(candidates)} < requested sample_count={int(args.sample_count)}"
            )
        samples = select_sharegpt_samples(candidates, int(args.sample_count), int(args.seed))
        if str(args.save_samples_json).strip():
            sample_path = Path(str(args.save_samples_json))
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_payload = {
                "candidate_count": int(candidate_count),
                "sample_count": int(len(samples)),
                "seed": int(args.seed),
                "prompt_len_range": [int(args.prompt_len_min), int(args.prompt_len_max)],
                "target_len_range": [int(args.target_len_min), int(args.target_len_max)],
                "samples": samples,
            }
            sample_path.write_text(json.dumps(sample_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            sample_source = str(sample_path)

    sample_summary = summarize_samples(samples, candidate_count)
    public_sample_summary = {k: v for k, v in sample_summary.items() if k != "sample_signature"}
    print(json.dumps({"sample_summary": public_sample_summary, "sample_source": sample_source}, ensure_ascii=False), flush=True)

    rows: List[Dict[str, Any]] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for method in methods:
        for concurrency in concurrencies:
            print(json.dumps({"starting_method": method, "concurrency": int(concurrency)}, ensure_ascii=False), flush=True)
            row = run_hf_style_prompt_batches(
                model_name=args.model_name,
                method=method,
                items=samples,
                concurrency=int(concurrency),
                max_new_tokens=int(args.max_new_tokens),
                repeats=int(args.repeats),
                evaluate_output_fn=eval_sharegpt_output,
                tokenizer=tokenizer,
            )
            row["actual_prompt_tokens_mean"] = float(sum(int(x["prompt_tokens"]) for x in samples) / max(1, len(samples)))
            rows.append(row)
            out.write_text(
                json.dumps(
                    build_payload(args, dataset_path, sample_source, sample_summary, methods, concurrencies, rows),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "method": row.get("method"),
                        "concurrency": row.get("concurrency"),
                        "status": row.get("status"),
                        "tokens_per_sec": row.get("tokens_per_sec"),
                        "generated_tokens": row.get("generated_tokens"),
                        "wall_clock_total_runtime_ms": row.get("wall_clock_total_runtime_ms"),
                        "error_reason": row.get("error_reason"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
