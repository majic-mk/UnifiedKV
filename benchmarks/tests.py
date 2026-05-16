from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / 'core'
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import torch
import math


def test_1_pool():
    from pool import PagedKVPool
    dummy = torch.zeros(100, 1024, 1024, device='cuda', dtype=torch.float16)
    pool = PagedKVPool(
        block_size=16, num_layers=4, num_kv_heads=8,
        head_dim=128, dtype=torch.float16
    )
    del dummy
    print(f"  N_total={pool.N_total}, N_wm_low={pool.N_wm_low}")
    assert pool.N_total > 0
    ids = pool.allocate_blocks(50)
    assert len(ids) == 50
    K = torch.randn(8, 50*16, 128, device='cuda', dtype=torch.float16)
    V = torch.randn_like(K)
    pool.write_kv_to_blocks(0, ids, K, V)
    K_r, V_r = pool.read_kv_from_blocks(0, ids, 50*16)
    assert torch.allclose(K, K_r, atol=1e-2), "读写不一致"
    pool.free_blocks_by_ids(ids)
    assert pool.n_free == pool.N_total
    del pool, K, V, K_r, V_r
    torch.cuda.empty_cache()
    print("✅ test_1_pool passed")


def test_2_compress():
    from compress import BlockAlignedSnapKV
    snapkv = BlockAlignedSnapKV(
        block_size=16, sink_len=64, obs_len=64, retain_ratio=0.20
    )
    assert snapkv.Lo == 64
    assert snapkv.B == 16
    K = torch.randn(1, 8, 16500, 128, device='cuda', dtype=torch.float16)
    V = torch.randn_like(K)
    Q_obs = torch.randn(1, 32, 64, 128, device='cuda', dtype=torch.float16)
    K_c, V_c = snapkv.compress(K, V, Q_obs, n_rep=4)
    L_final = K_c.shape[2]
    L_mid_c = L_final - 64 - 64
    assert L_mid_c % 16 == 0, f"中段未块对齐: {L_mid_c}"
    assert 0.10 < L_final / 16500 < 0.40
    assert K_c.shape == V_c.shape
    del K, V, Q_obs, K_c, V_c
    torch.cuda.empty_cache()
    print(f"✅ test_2_compress passed，压缩率={L_final/16500:.1%}")


def test_3_flash_attn():
    try:
        from flash_attn import flash_attn_with_kvcache
        N, B, h_kv, h_q, d = 256, 16, 8, 32, 128
        k_cache = torch.zeros(N, B, h_kv, d, device='cuda', dtype=torch.float16)
        v_cache = torch.zeros_like(k_cache)
        btable = torch.zeros(2, 10, dtype=torch.int32, device='cuda')
        q = torch.randn(2, 1, h_q, d, device='cuda', dtype=torch.float16)
        seqlens = torch.tensor([16, 32], dtype=torch.int32, device='cuda')
        out = flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=seqlens,
            block_table=btable, causal=True
        )
        assert out.shape == (2, 1, h_q, d)
        del k_cache, v_cache, btable, q, seqlens, out
        torch.cuda.empty_cache()
        print("✅ test_3_flash_attn passed")
    except ImportError:
        print("⚠️  flash_attn未安装，decode将使用concat备用方案")


def test_4_offload():
    from pool import PagedKVPool
    from offload import AsyncOffloadManager
    from kv_types import OffloadState
    torch.cuda.empty_cache()
    pool = PagedKVPool(16, 2, 8, 128, torch.float16, cpu_mem_gb=4.0)
    mgr = AsyncOffloadManager(pool, num_layers=2)
    K = torch.randn(8, 160, 128, device='cuda', dtype=torch.float16)
    V = torch.randn_like(K)
    mgr.register(0, 0, K, V)
    assert mgr.page_table[(0,0)].state == OffloadState.ON_GPU
    mgr.offload_sequence(0)
    assert mgr.page_table[(0,0)].state == OffloadState.ON_CPU
    mgr.prefetch_sequence(0)
    mgr.transfer_stream.synchronize()
    assert mgr.page_table[(0,0)].state == OffloadState.ON_GPU
    K_r, V_r = pool.read_kv_from_blocks(0, mgr.page_table[(0,0)].block_ids, 160)
    assert torch.allclose(K, K_r, atol=1e-2), "卸载/预取后数据不一致"
    mgr.release_sequence(0)
    del pool, mgr, K, V, K_r, V_r
    torch.cuda.empty_cache()
    print("✅ test_4_offload passed")


def test_5_chunked_prefill():
    from chunked_prefill import ChunkedPrefillProcessor
    proc = ChunkedPrefillProcessor(chunk_size=2048)
    full_gb, chunk_gb = ChunkedPrefillProcessor.estimate_peak_gb(
        seq_len=32768, num_heads=32, head_dim=128, chunk_size=2048
    )
    assert chunk_gb < full_gb * 0.1, "分块后峰值应小于一次性的10%"
    print(f"✅ test_5_chunked_prefill passed")
    print(f"   一次性peak={full_gb:.1f}GB，分块peak={chunk_gb:.1f}GB")


def test_6_e2e_short():
    """端到端验证：短序列单条，prefill→compress→decode 5步"""
    from engine import ManagedInferenceEngine
    
    torch.cuda.empty_cache()
    print("\n🔄 加载模型并初始化引擎...")
    engine = ManagedInferenceEngine(
        model_name="/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
        sink_len=64,
        obs_len=64,
        retain_ratio=0.20,
        max_new_tokens=5,
    )
    
    print("🔄 运行端到端推理...")
    results = engine.generate(["你好，请介绍一下你自己"])
    
    assert len(results) == 1
    assert len(results[0]) > 0
    print(f"✅ test_6_e2e_short passed")
    print(f"   生成结果: {results[0][:50]}...")
    
    # 清理
    del engine
    torch.cuda.empty_cache()


def test_7_concurrent():
    """15并发 × 4096 Token，验证不OOM"""
    from engine import ManagedInferenceEngine
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print("\n🔄 加载模型并初始化引擎...")
    engine = ManagedInferenceEngine(
        model_name="/root/autodl-tmp/models/Qwen2.5-7B-Instruct",
        max_new_tokens=32,
    )
    
    print("🔄 运行15并发压测...")
    prompts = ["请详细分析以下内容：" + "这是测试文本。" * 200] * 15
    results = engine.generate(prompts)
    
    assert len(results) == 15
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    
    print(f"✅ test_7_concurrent passed")
    print(f"   峰值显存={peak_gb:.1f}GB")
    print(f"   所有序列生成完成")
    
    # 清理
    del engine
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("=== 分步验证 ===\n")
    test_1_pool()
    test_2_compress()
    test_3_flash_attn()
    test_4_offload()
    test_5_chunked_prefill()
    
    # 以下需要实际模型，确认GPU和模型下载后再运行
    test_6_e2e_short()    # 先跑这个，单条短序列
    # test_7_concurrent()   # 确认test_6通过后再跑这个
