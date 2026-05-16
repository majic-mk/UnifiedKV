# LEGACY_BENCHMARK: retained for historical reference; use benchmark_p3_tiered_table.py and benchmark_p3_quality_passkey.py as paper-facing entry points.
import gc
import json
import time
from typing import Dict

import torch

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from engine import ManagedInferenceEngine


LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
CONCURRENCY = 8
PROMPT_REPEAT = 80
MAX_NEW_TOKENS = 192


def build_prompts(n: int, repeat: int):
    prompts = []
    for i in range(n):
        text = (
            "这是一段高并发长文本压力测试语料，用于比较P3窗口化开启和关闭时的吞吐、延迟和稳定性。"
            * repeat
        )
        prompts.append(f"[seq={i}] {text}\\n请继续生成与主题相关的技术说明。")
    return prompts


def run_case(case_name: str, p3_enabled: bool) -> Dict:
    print(f"\\n===== CASE: {case_name} =====")
    engine = ManagedInferenceEngine(
        model_name=LOCAL_MODEL_PATH,
        gpu_mem_frac=0.05,
        cpu_mem_gb=32.0,
        chunk_size=1024,
        max_new_tokens=MAX_NEW_TOKENS,
        prefill_batch_size=4,
        decode_micro_batch_size=0,
        decode_window_enabled=p3_enabled,
        decode_window_auto_on_pressure=False,
        decode_window_sink_len=64,
        decode_window_recent_len=64,
        decode_window_check_interval=4,
        decode_window_min_trigger_tokens=32,
        decode_window_min_drop_tokens=32,
    )

    # Fix decode length for stable throughput comparison.
    engine.tokenizer.eos_token = None
    prompts = build_prompts(CONCURRENCY, PROMPT_REPEAT)

    t0 = time.perf_counter()
    outputs, metrics = engine.generate(prompts, return_metrics=True)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    metrics["case"] = case_name
    metrics["wall_ms"] = round(wall_ms, 3)
    metrics["sample_output"] = outputs[0][:160]
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def print_summary(off: Dict, on: Dict):
    print("\\n===== SUMMARY (P3 ON vs OFF) =====")
    keys = [
        "tokens_per_sec",
        "decode_step_p95_ms",
        "peak_cuda_mem_gb",
        "thrash_win16",
        "decode_window_prunes",
        "decode_window_tokens_dropped",
        "wall_ms",
    ]
    for key in keys:
        a = float(off.get(key, 0.0))
        b = float(on.get(key, 0.0))
        if abs(a) < 1e-12:
            delta_pct = 0.0
        else:
            delta_pct = (b - a) / a * 100.0
        print(f"{key}: off={a:.4f} on={b:.4f} delta={delta_pct:+.2f}%")


def main():
    off = run_case("p3_off", p3_enabled=False)
    on = run_case("p3_on", p3_enabled=True)
    print_summary(off, on)


if __name__ == "__main__":
    main()
