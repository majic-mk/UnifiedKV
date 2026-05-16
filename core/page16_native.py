import math
from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - import availability is runtime-specific.
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _page16_decode_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        block_table_ptr,
        seqlens_ptr,
        out_ptr,
        max_blocks: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
        block_d: tl.constexpr,
        n_rep: tl.constexpr,
        softmax_scale: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        kv_h = pid_h // n_rep
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim

        q = tl.load(
            q_ptr + (pid_b * num_q_heads + pid_h) * head_dim + offs_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        seq_len = tl.load(seqlens_ptr + pid_b)

        m_i = tl.full((), -3.4028234663852886e38, tl.float32)
        l_i = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), tl.float32)
        offs_t = tl.arange(0, block_size)

        blk = 0
        while blk < max_blocks:
            token_idx = blk * block_size + offs_t
            token_mask = token_idx < seq_len
            block_id = tl.load(block_table_ptr + pid_b * max_blocks + blk)
            k_offsets = (((block_id * block_size + offs_t[:, None]) * num_kv_heads + kv_h) * head_dim + offs_d[None, :])
            k = tl.load(k_ptr + k_offsets, mask=token_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            scores = tl.sum(k * q[None, :], axis=1) * softmax_scale
            scores = tl.where(token_mask, scores, -3.4028234663852886e38)

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(token_mask, p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)

            v_offsets = (((block_id * block_size + offs_t[:, None]) * num_kv_heads + kv_h) * head_dim + offs_d[None, :])
            v = tl.load(v_ptr + v_offsets, mask=token_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new
            l_i = l_new
            blk += 1

        out = acc / l_i
        tl.store(
            out_ptr + (pid_b * num_q_heads + pid_h) * head_dim + offs_d,
            out,
            mask=d_mask,
        )


def page16_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    softmax_scale: float,
    n_rep: int,
) -> torch.Tensor:
    """Decode-only paged attention over 16-token blocks.

    Args:
        q: [batch, num_q_heads, head_dim], contiguous CUDA tensor.
        k_cache/v_cache: [num_blocks, 16, num_kv_heads, head_dim].
        block_table: [batch, max_blocks] int32 CUDA tensor.
        cache_seqlens: [batch] int32 CUDA tensor, including the current token.
    """
    if not HAS_TRITON:
        raise RuntimeError("page16_native_unavailable:triton_not_available")
    if q.dim() != 3:
        raise RuntimeError(f"page16_native_shape_unsupported:q_dim_{q.dim()}")
    if k_cache.dim() != 4 or v_cache.dim() != 4:
        raise RuntimeError("page16_native_shape_unsupported:cache_rank")
    if int(k_cache.shape[1]) != 16:
        raise RuntimeError(f"page16_native_shape_unsupported:block_size_{int(k_cache.shape[1])}")
    if block_table.dim() != 2:
        raise RuntimeError("page16_native_shape_unsupported:block_table_rank")
    batch, num_q_heads, head_dim = [int(x) for x in q.shape]
    num_kv_heads = int(k_cache.shape[2])
    if batch <= 0 or num_q_heads <= 0 or head_dim <= 0:
        raise RuntimeError("page16_native_shape_unsupported:empty")
    if int(v_cache.shape[1]) != 16 or int(v_cache.shape[2]) != num_kv_heads or int(v_cache.shape[3]) != head_dim:
        raise RuntimeError("page16_native_shape_unsupported:v_cache")
    if num_q_heads % num_kv_heads != 0:
        raise RuntimeError("page16_native_shape_unsupported:gqa")
    if int(block_table.shape[0]) != batch:
        raise RuntimeError("page16_native_shape_unsupported:block_table_batch")
    if int(cache_seqlens.numel()) != batch:
        raise RuntimeError("page16_native_shape_unsupported:seqlens")
    if head_dim > 256:
        raise RuntimeError(f"page16_native_shape_unsupported:head_dim_{head_dim}")

    q = q.contiguous()
    block_table = block_table.contiguous()
    cache_seqlens = cache_seqlens.to(dtype=torch.int32).contiguous()
    out = torch.empty_like(q)
    block_d = 1 << int(math.ceil(math.log2(head_dim)))
    max_blocks = int(block_table.shape[1])
    grid = (batch, num_q_heads)
    _page16_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        cache_seqlens,
        out,
        max_blocks,
        num_q_heads,
        num_kv_heads,
        head_dim,
        16,
        block_d,
        int(n_rep),
        float(softmax_scale),
        num_warps=4,
    )
    return out
