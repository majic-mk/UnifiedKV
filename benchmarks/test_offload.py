from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import torch
import gc
import os
import time

LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"


def test_offload_triggered():
    """专门验证CPU卸载机制被触发"""
    from engine import ManagedInferenceEngine
    from kv_types import OffloadState
    
    print("=== 卸载机制验证测试 ===\n")
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ 本地模型不存在: {LOCAL_MODEL_PATH}")
        return False
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("🔄 加载模型（gpu_mem_frac=0.12，强制触发卸载）...")
    engine = ManagedInferenceEngine(
        model_name=LOCAL_MODEL_PATH,
        gpu_mem_frac=0.05,
        cpu_mem_gb=32.0,
        max_new_tokens=32,
        sink_len=32,
        obs_len=32,
        retain_ratio=0.15,
    )
    
    offload_count = [0]
    prefetch_count = [0]
    evict_count = [0]
    
    original_offload = engine.scheduler.offloader.offload_sequence
    original_prefetch = engine.scheduler.offloader.prefetch_sequence
    original_evict = engine.scheduler.offloader.evict_coldest_sequence
    
    def patched_evict(exclude_seq, *args, **kwargs):
        evict_count[0] += 1
        n_free_before = engine.scheduler.pool.n_free
        print(f"  🚨 驱逐触发：exclude_seq={exclude_seq}，驱逐前空闲块={n_free_before}")
        try:
            result = original_evict(exclude_seq, *args, **kwargs)
            n_free_after = engine.scheduler.pool.n_free
            print(f"  ✅ 驱逐完成：释放块={n_free_after - n_free_before}")
            return result
        except MemoryError as e:
            print(f"  ❌ 驱逐失败：{e}")
            raise
    
    def patched_offload(seq_id):
        offload_count[0] += 1
        n_free_before = engine.scheduler.pool.n_free
        print(f"  🔄 卸载触发：seq_id={seq_id}，卸载前空闲块={n_free_before}")
        result = original_offload(seq_id)
        n_free_after = engine.scheduler.pool.n_free
        cpu_used = engine.scheduler.pool.cpu_used_bytes / 1024**3
        print(f"  ✅ 卸载完成：释放块={n_free_after - n_free_before}，CPU已用={cpu_used:.2f}GB")
        return result
    
    def patched_prefetch(seq_id):
        prefetch_count[0] += 1
        n_free_before = engine.scheduler.pool.n_free
        print(f"  ⬆️ 预取触发：seq_id={seq_id}，预取前空闲块={n_free_before}")
        result = original_prefetch(seq_id)
        n_free_after = engine.scheduler.pool.n_free
        print(f"  ✅ 预取完成：占用块={n_free_before - n_free_after}")
        return result
    
    engine.scheduler.offloader.evict_coldest_sequence = patched_evict
    engine.scheduler.offloader.offload_sequence = patched_offload
    engine.scheduler.offloader.prefetch_sequence = patched_prefetch
    
    original_post_prefill = engine.scheduler.post_prefill_compress
    
    def patched_post_prefill(seq_ids):
        print(f"\n  📊 post_prefill_compress 开始，序列数={len(seq_ids)}")
        pool = engine.scheduler.pool
        print(f"     空闲块={pool.n_free}, N_wm_high={pool.N_wm_high}")
        return original_post_prefill(seq_ids)
    
    engine.scheduler.post_prefill_compress = patched_post_prefill
    
    prompts = ["这是一段很长的测试文本，用于触发卸载机制。" * 600] * 10
    
    print("\n🔄 运行卸载触发测试...")
    pool = engine.scheduler.pool
    print(f"  初始状态：N_total={pool.N_total}, N_wm_low={pool.N_wm_low}, N_wm_high={pool.N_wm_high}")
    print(f"  初始空闲块：{pool.n_free}")
    
    try:
        results = engine.generate(prompts)
        
        print(f"\n  结束空闲块：{pool.n_free}")
        print(f"  CPU内存使用：{pool.cpu_used_bytes/1024**3:.2f}GB / {pool.cpu_budget/1024**3:.1f}GB")
        print(f"  驱逐调用次数：{evict_count[0]}")
        print(f"  卸载总次数：{offload_count[0]}")
        print(f"  预取总次数：{prefetch_count[0]}")
        
        if offload_count[0] > 0:
            print(f"\n✅ 卸载测试通过，卸载触发{offload_count[0]}次，预取{prefetch_count[0]}次")
        else:
            print(f"\n❌ 卸载从未被触发！FlexGen机制未生效")
        
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        
        return offload_count[0] > 0
    except MemoryError as e:
        print(f"\n⚠️ MemoryError: {e}")
        print(f"  驱逐调用次数：{evict_count[0]}")
        print(f"  卸载总次数：{offload_count[0]}")
        print(f"  预取总次数：{prefetch_count[0]}")
        if offload_count[0] > 0:
            print(f"\n✅ 卸载机制已被触发{offload_count[0]}次（虽然最终OOM）")
            return True
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        return False


def test_throughput_comparison():
    """对比有卸载 vs 无卸载的吞吐差异"""
    from engine import ManagedInferenceEngine
    
    print("\n=== 吞吐对比测试 ===\n")
    
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ 本地模型不存在: {LOCAL_MODEL_PATH}")
        return False
    
    prompts = ["测试文本。" * 100] * 10
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print("🔄 测试无卸载场景（gpu_mem_frac=0.50）...")
    engine_no_offload = ManagedInferenceEngine(
        model_name=LOCAL_MODEL_PATH,
        gpu_mem_frac=0.50,
        max_new_tokens=32,
    )
    
    start = time.time()
    results = engine_no_offload.generate(prompts)
    time_no_offload = time.time() - start
    print(f"  无卸载耗时：{time_no_offload:.2f}s")
    
    del engine_no_offload
    torch.cuda.empty_cache()
    gc.collect()
    
    print("\n🔄 测试有卸载场景（gpu_mem_frac=0.15）...")
    engine_with_offload = ManagedInferenceEngine(
        model_name=LOCAL_MODEL_PATH,
        gpu_mem_frac=0.15,
        cpu_mem_gb=32.0,
        max_new_tokens=32,
    )
    
    start = time.time()
    results = engine_with_offload.generate(prompts)
    time_with_offload = time.time() - start
    print(f"  有卸载耗时：{time_with_offload:.2f}s")
    
    overhead = (time_with_offload/time_no_offload - 1) * 100
    print(f"\n  性能差异：{overhead:.1f}%")
    print(f"  （预期：有卸载比无卸载慢20-40%，因为PCIe传输开销）")
    
    del engine_with_offload
    gc.collect()
    torch.cuda.empty_cache()
    
    return True


def collect_metrics(engine):
    """收集卸载相关指标"""
    from kv_types import OffloadState
    
    pool = engine.scheduler.pool
    offloader = engine.scheduler.offloader
    
    metrics = {
        'n_total': pool.N_total,
        'n_free': pool.n_free,
        'n_wm_low': pool.N_wm_low,
        'n_wm_high': pool.N_wm_high,
        'cpu_used_gb': pool.cpu_used_bytes / 1024**3,
        'cpu_budget_gb': pool.cpu_budget / 1024**3,
        'sequences_on_gpu': 0,
        'sequences_on_cpu': 0,
        'sequences_pinned': 0,
    }
    
    seq_states = {}
    for (s, l), e in offloader.page_table.items():
        if s not in seq_states:
            seq_states[s] = e.state
        if e.state == OffloadState.ON_GPU:
            metrics['sequences_on_gpu'] += 1
        elif e.state == OffloadState.ON_CPU:
            metrics['sequences_on_cpu'] += 1
        elif e.state == OffloadState.PINNED:
            metrics['sequences_pinned'] += 1
    
    metrics['unique_sequences'] = len(seq_states)
    return metrics


if __name__ == "__main__":
    success = test_offload_triggered()
    
    if success:
        test_throughput_comparison()
    
    print("\n=== 测试完成 ===")
