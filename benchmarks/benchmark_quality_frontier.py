import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "core") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "core"))

from engine import ManagedInferenceEngine

LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
GROUP_ARGS = {
    "off_raw": {"retain_ratio": 1.0, "decode_window_enabled": False, "decode_window_auto_on_pressure": False},
    "off_compress": {"retain_ratio": 0.10, "decode_window_enabled": False, "decode_window_auto_on_pressure": False},
    "main_auto_compress": {"retain_ratio": 0.10, "decode_window_enabled": False, "decode_window_auto_on_pressure": True},
}


def iter_json_records(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
        return
    obj = json.loads(text)
    if isinstance(obj, list):
        for item in obj:
            yield item
    elif isinstance(obj, dict):
        for key in ["data", "records", "examples"]:
            val = obj.get(key)
            if isinstance(val, list):
                for item in val:
                    yield item
                return
        yield obj


def extract_prompt(record: Dict[str, Any]) -> str:
    if isinstance(record.get("prompt"), str) and record["prompt"].strip():
        return record["prompt"]
    parts = [
        str(record.get("instruction", "") or "").strip(),
        str(record.get("context", "") or record.get("document", "") or record.get("text", "")).strip(),
        str(record.get("input", "") or record.get("question", "") or record.get("query", "")).strip(),
    ]
    parts = [p for p in parts if p]
    return "\n\n".join(parts) if parts else json.dumps(record, ensure_ascii=False)


def extract_answers(record: Dict[str, Any]) -> List[str]:
    raw = record.get("answers", record.get("answer", record.get("target", [])))
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if raw is None:
        return []
    return [str(raw)]


def load_frontier(frontier_json: str, group: str, input_len: int) -> Dict[str, Any]:
    payload = json.loads(Path(frontier_json).read_text(encoding="utf-8"))
    for row in payload.get("frontier_rows", []):
        if str(row.get("group")) == str(group) and int(row.get("input_len", 0)) == int(input_len):
            return row
    raise ValueError(f"missing frontier row for group={group}, input_len={input_len}")


def build_engine(model_name: str, group: str, gpu_mem_frac: float, max_new_tokens: int) -> ManagedInferenceEngine:
    kwargs = {
        "model_name": model_name,
        "cpu_mem_gb": 32.0,
        "chunk_size": 1024,
        "max_new_tokens": int(max_new_tokens),
        "prefill_batch_size": 4,
        "decode_micro_batch_size": 0,
        "sink_len": 16,
        "obs_len": 16,
        "decode_window_sink_len": 16,
        "decode_path_mode": "rebuild",
        "decode_paged_flash_enabled": False,
        "gpu_mem_frac": float(gpu_mem_frac),
    }
    kwargs.update(GROUP_ARGS[group])
    return ManagedInferenceEngine(**kwargs)


def cleanup_engine(engine: ManagedInferenceEngine) -> None:
    try:
        if hasattr(engine, "_reset_online_runtime"):
            engine._reset_online_runtime(clear_request_counter=False)
    except Exception:
        pass
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality harness for single-seq and frontier settings")
    parser.add_argument("--model-name", type=str, default=LOCAL_MODEL_PATH)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--groups", type=str, default="off_raw,off_compress,main_auto_compress")
    parser.add_argument("--setting", type=str, default="single", choices=["single", "frontier"])
    parser.add_argument("--frontier-json", type=str, default="")
    parser.add_argument("--input-len", type=int, default=32768)
    parser.add_argument("--single-gpu-mem-frac", type=float, default=0.70)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--out", type=str, default="benchmark_quality_frontier_results.json")
    args = parser.parse_args()

    records = []
    for idx, item in enumerate(iter_json_records(Path(args.dataset_path))):
        if idx >= int(args.limit):
            break
        records.append({"prompt": extract_prompt(item), "answers": extract_answers(item)})
    groups = [g.strip() for g in str(args.groups).split(",") if g.strip()]
    rows = []
    for group in groups:
        concurrency = 1
        gpu_mem_frac = float(args.single_gpu_mem_frac)
        if args.setting == "frontier":
            row = load_frontier(args.frontier_json, group, int(args.input_len))
            concurrency = max(1, int(row.get("max_supported_concurrency", 1)))
            gpu_mem_frac = float(row.get("gpu_mem_frac_best", gpu_mem_frac))
        engine = build_engine(args.model_name, group, gpu_mem_frac, int(args.max_new_tokens))
        prompts = [records[i % len(records)]["prompt"] for i in range(concurrency)]
        outputs, metrics = engine.generate(prompts, return_metrics=True)
        match = 0
        valid = 0
        for i, out in enumerate(outputs):
            answers = records[i % len(records)].get("answers", [])
            if str(out).strip():
                valid += 1
            if answers and any(ans and ans in str(out) for ans in answers):
                match += 1
        rows.append({
            "group": group,
            "setting": args.setting,
            "input_len": int(args.input_len),
            "concurrency": int(concurrency),
            "gpu_mem_frac": float(gpu_mem_frac),
            "tokens_per_sec": float(metrics.get("tokens_per_sec", 0.0)),
            "decode_step_p95_ms": float(metrics.get("decode_step_p95_ms", 0.0)),
            "match_rate": float(match / max(1, len(outputs))),
            "valid_rate": float(valid / max(1, len(outputs))),
            "cuda_free_min_gb": float(metrics.get("cuda_free_min_gb", 0.0)),
        })
        cleanup_engine(engine)
    Path(args.out).write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
