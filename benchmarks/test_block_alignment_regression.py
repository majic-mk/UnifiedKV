from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from compress import BlockAlignedSnapKV


def test_partial_middle_block_padding_preserves_fixed_budget_shape():
    torch.manual_seed(7)
    block_size = 16
    sink_len = 16
    observation_len = 16
    sequence_len = 2_795
    budget = 2_048
    device = "cuda" if torch.cuda.is_available() else "cpu"

    compressor = BlockAlignedSnapKV(
        block_size=block_size,
        sink_len=sink_len,
        obs_len=observation_len,
        retain_budget_tokens=budget,
    )
    k = torch.randn(1, 8, sequence_len, 32, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    q_obs = torch.randn(
        1, 32, observation_len, 32, device=device, dtype=torch.float16
    )

    k_compressed, v_compressed = compressor.compress(k, v, q_obs, n_rep=4)

    middle_tokens = sequence_len - sink_len - observation_len
    expected_aligned_middle = (
        (middle_tokens + block_size - 1) // block_size
    ) * block_size
    debug = compressor.last_debug
    assert debug["mid_tokens_before_align"] == middle_tokens
    assert debug["mid_tokens_aligned"] == expected_aligned_middle
    assert max(debug["retained_block_idx"][0]) < expected_aligned_middle // block_size
    assert k_compressed.shape == v_compressed.shape
    assert k_compressed.shape[2] == budget


if __name__ == "__main__":
    test_partial_middle_block_padding_preserves_fixed_budget_shape()
    print("block-alignment regression passed")
