import argparse
import json
import time
from statistics import mean

from benchmark_niah_single_limit import (
    _build_engine,
    _run_single_case,
    _track_summary_rec,
    _emit_progress,
    PREFILL_COMPRESS_ARGS,
    COMMON_BASE,
)


def _parse_depths(s: str):
    out = []
    for x in str(s).split(','):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def main():
    parser = argparse.ArgumentParser(description='Fixed 32k passkey single-case regression')
    parser.add_argument('--target-len', type=int, default=32000)
    parser.add_argument('--gpu-mem-frac', type=float, default=0.78)
    parser.add_argument('--depths', type=str, default='0.1,0.5,0.9')
    parser.add_argument('--seed', type=int, default=20260322)
    parser.add_argument('--case-timeout-s', type=int, default=300)
    parser.add_argument('--compress-retain-ratio', type=float, default=0.75)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--progress-jsonl', type=str, required=True)
    args = parser.parse_args()

    PREFILL_COMPRESS_ARGS['retain_ratio'] = float(max(0.0, min(1.0, args.compress_retain_ratio)))
    depths = _parse_depths(args.depths)
    if not depths:
        raise ValueError('no depths selected')

    with open(args.progress_jsonl, 'w', encoding='utf-8') as f:
        f.write('')

    engine = _build_engine(float(args.gpu_mem_frac), 'p2_only', 'compress')
    records = []
    for depth in depths:
        rec = _run_single_case(
            engine,
            'p2_only',
            'compress',
            int(args.target_len),
            float(depth),
            int(args.seed),
            'fixed_32k_passkey',
            int(args.case_timeout_s),
        )
        records.append(rec)
        _emit_progress(rec, args.progress_jsonl)

    summary = _track_summary_rec('p2_only', 'compress', 'fixed_32k_passkey_summary', int(args.target_len), records)
    payload = {
        'meta': {
            'task': 'fixed_32k_passkey_single',
            'target_len': int(args.target_len),
            'gpu_mem_frac': float(args.gpu_mem_frac),
            'depths': depths,
            'seed': int(args.seed),
            'case_timeout_s': int(args.case_timeout_s),
            'compress_retain_ratio': float(PREFILL_COMPRESS_ARGS['retain_ratio']),
            'sink_len': int(COMMON_BASE['sink_len']),
            'snapkv_observation_len': int(COMMON_BASE['snapkv_observation_len']),
        },
        'summary': summary,
        'records': records,
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({'ts': time.strftime('%Y-%m-%d %H:%M:%S'), 'stage': 'done', 'out': args.out}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
