#!/root/miniconda3/bin/python
import json, os, subprocess, time
from pathlib import Path
BASE = Path('/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/results/paper/single_seq_qwen_neighbors_rerun_fixed')
BENCH = Path('/root/autodl-tmp/.autodl/kv_cache_middleware/benchmarks/benchmark_niah_single_limit.py')
PY = '/root/miniconda3/bin/python'
MODEL = '/root/autodl-tmp/models/Qwen2.5-7B-Instruct'
SEED = '20260322'
DEPTHS = '0.10,0.50,0.90'

def running_010_parent():
    out = subprocess.run("ps -eo pid,cmd | grep 'off_compress_memfrac_0.10.json' | grep -v grep", shell=True, capture_output=True, text=True)
    return bool(out.stdout.strip())

def read_l_survival_from_json(path: Path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        tracks = data.get('tracks') or []
        if tracks:
            v = tracks[0].get('l_survival_max')
            if isinstance(v, int) and v > 0:
                return v
    except Exception:
        pass
    return None

def append_jsonl(path: Path, obj):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def run_one(frac: str, start_len: int, label: str):
    progress = BASE / f'off_compress_memfrac_{label}_stepdown.progress.jsonl'
    out_json = BASE / f'off_compress_memfrac_{label}_stepdown.json'
    tmp = Path(f'/tmp/quality_stepdown_{label}.json')
    tested = []
    best = None
    if progress.exists():
        progress.unlink()
    if out_json.exists():
        out_json.unlink()
    for target in range(int(start_len), 15999, -1000):
        if tmp.exists():
            tmp.unlink()
        cmd = [
            PY, str(BENCH),
            '--model-name', MODEL,
            '--max-new-tokens', '32',
            '--depths', DEPTHS,
            '--target-em-threshold', '1.0',
            '--target-valid-threshold', '1.0',
            '--gpu-mem-frac-max', str(frac),
            '--gpu-mem-frac-fallback-step', '0',
            '--gpu-mem-frac-fallback-tries', '1',
            '--compress-profile', 'serving',
            '--sink-len', '16', '--obs-len', '16',
            '--child-quality-eval-json-out', str(tmp),
            '--child-mode', 'off',
            '--child-prefill-track', 'compress',
            '--child-target-len', str(target),
            '--child-depths', DEPTHS,
            '--child-stage', 'quality_stepdown',
            '--seed', SEED,
            '--case-timeout-s', '600',
        ]
        t0 = time.time()
        cp = subprocess.run(cmd, capture_output=True, text=True)
        rec = {
            'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
            'label': label,
            'gpu_mem_frac': float(frac),
            'target_len': target,
            'returncode': cp.returncode,
            'wall_s': round(time.time()-t0, 3),
        }
        if tmp.exists():
            try:
                payload = json.loads(tmp.read_text(encoding='utf-8'))
                rec['ok'] = bool(payload.get('ok'))
                rec['stat'] = payload.get('stat', {})
                rec['records'] = payload.get('records', [])
            except Exception as e:
                rec['ok'] = False
                rec['error'] = f'json_load_failed: {e}'
                rec['stdout_tail'] = cp.stdout[-1000:]
                rec['stderr_tail'] = cp.stderr[-1000:]
        else:
            rec['ok'] = False
            rec['error'] = 'missing_child_json'
            rec['stdout_tail'] = cp.stdout[-1000:]
            rec['stderr_tail'] = cp.stderr[-1000:]
        tested.append(rec)
        append_jsonl(progress, rec)
        if rec.get('ok'):
            best = rec
            break
    summary = {
        'gpu_mem_frac': float(frac),
        'start_len': int(start_len),
        'best_quality_len': int(best['target_len']) if best else 0,
        'best_record': best,
        'tested': tested,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    while running_010_parent():
        time.sleep(20)
    tasks = [
        ('0.24', 105000, '0.24'),
        ('0.12', 121000, '0.12'),
    ]
    s010 = read_l_survival_from_json(BASE / 'off_compress_memfrac_0.10.json')
    if s010 is None:
        # fallback to current known lower bound if final json not present
        s010 = 121000
    tasks.append(('0.10', int(s010), '0.10'))
    for frac, start_len, label in tasks:
        run_one(frac, start_len, label)

if __name__ == '__main__':
    main()
