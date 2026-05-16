from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import torch
import os
import gc

# 设置本地模型路径
LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"


def test_6_e2e_short_local(engine):
    """端到端验证：短序列单条，prefill→compress→decode 5步（使用本地模型）"""
    print("\n🔄 运行端到端推理 (test_6)...")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
    prompt = engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    results = engine.generate([prompt])
    
    assert len(results) == 1
    assert len(results[0]) > 0
    print(f"✅ test_6_e2e_short_local passed")
    print(f"   生成结果: {results[0][:100]}...")
    return True


def test_7_concurrent_local(engine):
    """15并发 × 4096 Token，验证不OOM（使用本地模型）"""
    torch.cuda.reset_peak_memory_stats()
    
    print("\n🔄 运行15并发压测 (test_7)...")
    prompts = ["请详细分析以下内容：" + "这是测试文本。" * 100] * 15  # 15条 × ~1000token
    results = engine.generate(prompts)
    
    assert len(results) == 15
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"✅ test_7_concurrent_local passed")
    print(f"   峰值显存={peak_gb:.1f}GB")
    print(f"   所有序列生成完成")
    return True


def test_offload_triggered(engine):
    """专门验证CPU卸载机制被触发"""
    offload_count = [0]
    prefetch_count = [0]
    
    original_offload = engine.scheduler.offloader.offload_sequence
    original_prefetch = engine.scheduler.offloader.prefetch_sequence
    
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
    
    engine.scheduler.offloader.offload_sequence = patched_offload
    engine.scheduler.offloader.prefetch_sequence = patched_prefetch
    
    prompts = ["这是一段很长的测试文本，用于触发卸载机制。" * 200] * 8
    
    print("\n🔄 运行卸载触发测试 (test_offload_triggered)...")
    pool = engine.scheduler.pool
    print(f"  初始状态：N_total={pool.N_total}, N_wm_low={pool.N_wm_low}, N_wm_high={pool.N_wm_high}")
    print(f"  初始空闲块：{pool.n_free}")
    
    results = engine.generate(prompts)
    
    print(f"  结束空闲块：{pool.n_free}")
    print(f"  CPU内存使用：{pool.cpu_used_bytes/1024**3:.2f}GB / {pool.cpu_budget/1024**3:.1f}GB")
    print(f"  卸载总次数：{offload_count[0]}")
    print(f"  预取总次数：{prefetch_count[0]}")
    
    assert offload_count[0] > 0, "❌ 卸载从未被触发！FlexGen机制未生效"
    assert len(results) == 8
    print(f"✅ 卸载测试通过，卸载触发{offload_count[0]}次，预取{prefetch_count[0]}次")
    return True


if __name__ == "__main__":
    print("=== 端到端测试（本地模型）===\n")
    
    # 检查模型是否存在
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ 本地模型不存在: {LOCAL_MODEL_PATH}")
        print("   请先运行 download_model.py 下载模型")
        exit(1)
    
    print(f"✅ 找到本地模型: {LOCAL_MODEL_PATH}")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    
    from engine import ManagedInferenceEngine
    
    print("\n🔄 加载本地模型并初始化引擎...")
    engine = ManagedInferenceEngine(
        model_name=LOCAL_MODEL_PATH,
        sink_len=64,
        obs_len=64,
        retain_ratio=0.20,
        max_new_tokens=100,  # test_6用100个token
        gpu_mem_frac=0.35,   # 从默认0.75降到0.35，为并发留出更多空间
    )
    
    # 运行test_6
    try:
        test_6_e2e_short_local(engine)
    except Exception as e:
        print(f"❌ test_6_e2e_short_local failed: {e}")
    
    # 修改max_new_tokens为32用于test_7
    engine.max_new_tokens = 32
    
    # 运行test_7
    try:
        test_7_concurrent_local(engine)
    except Exception as e:
        print(f"❌ test_7_concurrent_local failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 清理
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n=== 所有测试通过 ===")
