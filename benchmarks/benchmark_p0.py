# LEGACY_BENCHMARK: retained for historical reference; use benchmark_p3_tiered_table.py and benchmark_p3_quality_passkey.py as paper-facing entry points.
import gc
import json
import os
import time
from typing import Dict, List

import torch

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from engine import ManagedInferenceEngine

LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"


def _seed_all(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_case(name: str, engine: ManagedInferenceEngine, prompts: List[str]) -> Dict:
    print(f"\n=== CASE: {name} ===")
    outputs, metrics = engine.generate(prompts, return_metrics=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"sample_output: {outputs[0][:80] if outputs else ''}")
    return {'name': name, 'metrics': metrics}


def main():
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Local model not found: {LOCAL_MODEL_PATH}")

    _seed_all(42)
    torch.cuda.empty_cache()
    gc.collect()

    t0 = time.time()
    engine = ManagedInferenceEngine(
        model_name=LOCAL_MODEL_PATH,
        max_new_tokens=64,
        gpu_mem_frac=0.35,
        prefill_batch_size=4,
        decode_micro_batch_size=8,
    )

    cases = [
        (
            'single_long',
            ["请总结以下内容并给出关键风险：" + "这是一个长上下文测试段落。" * 800],
        ),
        (
            'concurrent_8_mid',
            ["请概括以下文本：" + "测试文本。" * 200] * 8,
        ),
        (
            'concurrent_16_short',
            ["请回复一句话确认收到。" + "短文本。" * 40] * 16,
        ),
    ]

    all_results = []
    for name, prompts in cases:
        all_results.append(run_case(name, engine, prompts))

    out = {
        'timestamp': int(time.time()),
        'elapsed_sec': round(time.time() - t0, 3),
        'results': all_results,
    }
    out_path = '/root/autodl-tmp/.autodl/kv_cache_middleware/benchmark_p0_result.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nSaved benchmark results to: {out_path}")

    del engine
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
