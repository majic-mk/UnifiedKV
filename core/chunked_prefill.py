import torch
from typing import Optional, Tuple, List


class ChunkedPrefillProcessor:
    def __init__(self, chunk_size: int = 2048):
        self.chunk_size = chunk_size

    def run(self, model, input_ids: torch.Tensor, chunk_size: Optional[int] = None) -> Tuple[torch.Tensor, tuple]:
        cs = chunk_size or self.chunk_size
        seq_len = input_ids.shape[1]
        past_kv = None

        for start in range(0, seq_len, cs):
            end = min(start + cs, seq_len)
            chunk_ids = input_ids[:, start:end]

            with torch.no_grad():
                out = model(
                    input_ids=chunk_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            past_kv = out.past_key_values

            used = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            if used / total > 0.90:
                torch.cuda.empty_cache()

        last_logits = out.logits[:, -1:, :]
        return last_logits, past_kv

    @staticmethod
    def estimate_peak_gb(seq_len: int, num_heads: int, head_dim: int, chunk_size: int) -> Tuple[float, float]:
        full_gb = num_heads * seq_len * seq_len * 2 / 1024**3
        chunk_gb = num_heads * chunk_size * seq_len * 2 / 1024**3
        return full_gb, chunk_gb
