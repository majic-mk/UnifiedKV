import torch
from typing import List, Tuple
from pool import PagedKVPool
from offload import AsyncOffloadManager


class MultiGPUCoordinator:
    def __init__(
        self,
        num_gpus: int,
        block_size: int,
        num_layers: int,
        num_kv_heads_total: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        cpu_mem_gb_per_gpu: float = 16.0,
    ):
        assert num_kv_heads_total % num_gpus == 0, (
            f"KV heads {num_kv_heads_total} must be divisible by GPU count {num_gpus}"
        )

        self.num_gpus = num_gpus
        self.kv_heads_per_gpu = num_kv_heads_total // num_gpus
        self.pools: List[PagedKVPool] = []
        self.offloaders: List[AsyncOffloadManager] = []

        for gpu_id in range(num_gpus):
            with torch.cuda.device(f"cuda:{gpu_id}"):
                torch.cuda.synchronize()
                model_used = torch.cuda.memory_allocated()
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                reserve = int(0.10 * total)
                available = total - model_used - reserve
                frac = available / total

                pool = PagedKVPool(
                    block_size=block_size,
                    num_layers=num_layers,
                    num_kv_heads=self.kv_heads_per_gpu,
                    head_dim=head_dim,
                    dtype=dtype,
                    gpu_mem_frac=frac,
                    cpu_mem_gb=cpu_mem_gb_per_gpu,
                )
                offloader = AsyncOffloadManager(pool, num_layers)
                self.pools.append(pool)
                self.offloaders.append(offloader)

    def _split_kv_heads(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        shards = []
        for gpu_id in range(self.num_gpus):
            start = gpu_id * self.kv_heads_per_gpu
            end = (gpu_id + 1) * self.kv_heads_per_gpu
            shards.append((K[start:end].contiguous(), V[start:end].contiguous()))
        return shards

    def register_all_gpus(
        self,
        seq_id: int,
        layer_id: int,
        kv_shards: List[Tuple[torch.Tensor, torch.Tensor]],
    ):
        for gpu_id, (K, V) in enumerate(kv_shards):
            with torch.cuda.device(f"cuda:{gpu_id}"):
                self.offloaders[gpu_id].register(
                    seq_id,
                    layer_id,
                    K.to(f"cuda:{gpu_id}"),
                    V.to(f"cuda:{gpu_id}"),
                )

    def register_layer_from_full_kv(
        self,
        seq_id: int,
        layer_id: int,
        K: torch.Tensor,
        V: torch.Tensor,
    ):
        shards = self._split_kv_heads(K, V)
        self.register_all_gpus(seq_id, layer_id, shards)

    def append_decode_token_all_gpus(
        self,
        seq_id: int,
        layer_id: int,
        k_tok: torch.Tensor,
        v_tok: torch.Tensor,
    ):
        # k_tok/v_tok: [h_kv, 1, d]
        shards = self._split_kv_heads(k_tok, v_tok)
        for gpu_id, (k_shard, v_shard) in enumerate(shards):
            with torch.cuda.device(f"cuda:{gpu_id}"):
                self.offloaders[gpu_id].append_decode_token(
                    seq_id,
                    layer_id,
                    k_shard.to(f"cuda:{gpu_id}"),
                    v_shard.to(f"cuda:{gpu_id}"),
                )

    def global_evict(self, current_layer: int, current_seq: int):
        del current_layer
        for offloader in self.offloaders:
            try:
                offloader.evict_coldest_sequence(current_seq)
            except MemoryError:
                continue

    def global_watermark(self) -> str:
        ratios = [p.n_free / p.N_total for p in self.pools]
        worst = min(ratios)
        if worst < 0.15:
            return "VRAM_FULL"
        if worst < 0.35:
            return "WARNING"
        return "OK"

    def release_all(self, seq_id: int):
        for offloader in self.offloaders:
            offloader.release_sequence(seq_id)
