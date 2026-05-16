import gc
import json
import os
from typing import Dict, List

# Ensure deterministic CuBLAS behavior for reproducibility checks.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


def _set_deterministic_runtime():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Use warn_only to avoid hard failure on unsupported deterministic ops.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _find_eos_pos(ids: List[int], eos_id: int) -> int:
    for i, t in enumerate(ids):
        if t == eos_id:
            return i
    return -1


def _prefix_match_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _first_k_match(a: List[int], b: List[int], k: int = 100) -> int:
    n = min(k, len(a), len(b))
    return sum(1 for i in range(n) if a[i] == b[i])


def _mean_abs_diff(a: List[float], b: List[float], n: int) -> float:
    if n <= 0:
        return 0.0
    return float(sum(abs(a[i] - b[i]) for i in range(n)) / n)


def run_once(decode_micro_batch_size: int, prompts: List[str]):
    # Reset RNG per run so mb1/mb8 comparison is fully reproducible.
    _seed_all(42)
    _set_deterministic_runtime()
    engine = ManagedInferenceEngine(
        model_name=LOCAL_MODEL_PATH,
        max_new_tokens=128,
        gpu_mem_frac=0.35,
        prefill_batch_size=4,
        decode_micro_batch_size=decode_micro_batch_size,
    )
    outputs, metrics, details = engine.generate(
        prompts,
        return_metrics=True,
        return_details=True,
    )
    eos_id = int(engine.tokenizer.eos_token_id)
    del outputs
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return metrics, details, eos_id


def main():
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise FileNotFoundError(f"Local model not found: {LOCAL_MODEL_PATH}")

    _seed_all(42)
    prompts = [
        "请解释Transformer中的KV Cache并举例说明。",
        "给我一个Python异步编程的简洁示例，并说明常见错误。",
        "总结以下文本：" + "这是测试文本。" * 120,
        "写一段简短的项目周报模板。",
    ]

    m_ref, d_ref, eos_id = run_once(1, prompts)
    m_new, d_new, _ = run_once(8, prompts)
    seq_reports: List[Dict] = []
    strong_all = True
    weak_all = True

    for i, (ids_a, ids_b) in enumerate(zip(d_ref['token_ids'], d_new['token_ids'])):
        strong = ids_a == ids_b
        strong_all = strong_all and strong

        same_first100 = _first_k_match(ids_a, ids_b, 100)
        weak = same_first100 >= 95
        weak_all = weak_all and weak

        prefix = _prefix_match_len(ids_a, ids_b)
        lp_a = d_ref['token_logprobs'][i]
        lp_b = d_new['token_logprobs'][i]
        lp_n = min(prefix, len(lp_a), len(lp_b))
        lp_diff = _mean_abs_diff(lp_a, lp_b, lp_n)

        eos_a = _find_eos_pos(ids_a, eos_id)
        eos_b = _find_eos_pos(ids_b, eos_id)
        eos_diff = abs(eos_a - eos_b) if eos_a >= 0 and eos_b >= 0 else None

        seq_reports.append(
            {
                'seq_id': i,
                'strong_match': strong,
                'weak_match_first100_ge95': weak,
                'same_first100': same_first100,
                'prefix_match_len': prefix,
                'eos_pos_ref': eos_a,
                'eos_pos_new': eos_b,
                'eos_pos_abs_diff': eos_diff,
                'avg_abs_logprob_diff_on_prefix': round(lp_diff, 6),
            }
        )

    report = {
        'summary': {
            'strong_consistency_all': strong_all,
            'weak_consistency_all': weak_all,
        },
        'metrics_ref_decode_mb1': m_ref,
        'metrics_new_decode_mb8': m_new,
        'per_sequence': seq_reports,
    }

    out_path = '/root/autodl-tmp/.autodl/kv_cache_middleware/consistency_report.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report['summary'], ensure_ascii=False, indent=2))
    print(f"Saved consistency report to: {out_path}")


if __name__ == '__main__':
    main()
