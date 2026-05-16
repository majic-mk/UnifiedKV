import torch
from typing import List, Tuple

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn import flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def build_cu_seqlens(seq_lens: List[int], device='cuda') -> torch.Tensor:
    cu = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    for i, l in enumerate(seq_lens):
        cu[i + 1] = cu[i] + l
    return cu


def varlen_prefill_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    seq_lens: List[int],
) -> torch.Tensor:
    if not HAS_FLASH_ATTN:
        raise ImportError("varlen attention需要flash_attn")

    cu_seqlens = build_cu_seqlens(seq_lens)
    max_seqlen = max(seq_lens)

    out = flash_attn_varlen_func(
        q=Q,
        k=K,
        v=V,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=True,
    )
    return out
