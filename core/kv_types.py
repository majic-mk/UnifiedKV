from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch


class PhaseState(Enum):
    IDLE = 0
    PREFILL = 1
    COMPRESS = 2
    DECODE = 3


class OffloadState(Enum):
    ON_GPU = 0
    ON_CPU = 1
    PINNED = 2
    OFFLOAD_INFLIGHT = 3
    PREFETCH_INFLIGHT = 4
    MIXED = 5


@dataclass
class BlockTableEntry:
    seq_id: int
    layer_id: int
    state: OffloadState
    block_ids: List[int]
    seq_len: int
    logical_seq_len: int = 0
    materialized_blocks: int = 0
    cpu_k: Optional[torch.Tensor] = None
    cpu_v: Optional[torch.Tensor] = None
    last_access: int = 0
    prefill_anchor_k_mean: Optional[torch.Tensor] = None
    mid_base_blocks: int = 0
    mid_deleted_cum: int = 0
    prune_frozen: bool = False
    freeze_reason: str = ""
    pending_block_ids: Optional[List[int]] = None
    transfer_event: Optional[object] = None
    gpu_block_map: List[int] = field(default_factory=list)
    cpu_k_blocks: List[Optional[torch.Tensor]] = field(default_factory=list)
    cpu_v_blocks: List[Optional[torch.Tensor]] = field(default_factory=list)
    pending_gpu_block_map: Optional[List[int]] = None
    pending_block_indices: Optional[List[int]] = None


@dataclass
class DecodeAppendResult:
    ok: bool
    retryable: bool = False
    reason: str = ""
