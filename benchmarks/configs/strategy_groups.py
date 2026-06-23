from pathlib import Path
from typing import Dict

LOCAL_MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

HF_STYLE_METHODS = ["hf_vanilla", "off_raw"]
ALL_GROUPS = [
    "legacy_off_raw",
    "legacy_off_raw_page16",
    "off_raw_page16_online",
    "off_compress",
    "off_compress_page16",
    "off_compress_page16_r015",
    "off_compress_page16_b1024",
    "off_compress_page16_b2048",
    "off_compress_b2048_rebuild_ablation",
    "off_compress_page16_b4096",
    "snapkv_dense_b2048",
    "off_compress_page16_r020",
    "off_compress_page16_r030",
    "off_compress_page16_r050",
    "off_compress_page16_r010_s64o64",
    "off_compress_page16_r015_s64o64",
    "p2_only_compress",
    "p2_page16",
    "p2_page16_offline",
    "p2_page16_offline_b2048",
    "p2_page16_online",
    "p2_page16_online_b2048",
    "p2_page16_online_b2048_nop2_lowwm010_floor6",
    "p2_page16_online_b2048_nop2_lowwm010_floor8",
    "p2_page16_online_b2048_nop2_lowwm008_floor7",
    "p2_page16_online_b2048_nop2_lowwm008_floor8",
    "p2_page16_online_b2048_nop2_lowwm008_floor9",
    "p2_page16_online_b2048_nop2_lowwm008_floor6",
    "p2_page16_online_b2048_nop2_lowwm008_floor5",
    "p2_page16_online_b2048_nop2_lowwm010_out128_floor6",
    "p2_page16_online_b2048_nop2_lowwm010_out128_floor8",
    "p2_page16_online_b2048_nop2_lowwm010_out128_floor4",
    "p2_page16_online_b2048_nop2_lowwm010_floor4",
    "p2_page16_online_b2048_nop2_margin128_floor4",
    "p2_page16_online_b2048_nop2_margin64_floor4",
    "p2_page16_online_b2048_baseline_repro",
    "p2_page16_online_b2048_margin64_nolowwm",
    "p2_page16_online_b2048_no_kvadm_p2",
    "p2_page16_online_b2048_wm010_p2emerg",
    "p2_page16_online_b2048_wm008_p2emerg",
    "p2_page16_online_b2048_wm005_p2emerg",
    "p2_page16_online_b2048_wm010_nop2",
    "p2_page16_online_b2048_wm008_nop2",
    "p2_page16_online_b2048_wm005_nop2",
    "p2_page16_offline_r015",
    "p2_page16_offline_r015_hp_sanity",
    "p2_page16_offline_r015_hp_sanity_v2",
    "p2_page16_offline_r020",
    "p2_page16_offline_r015_s64o64",
]
DEFAULT_MAINLINE_GROUPS = ["off_compress", "p2_only_compress"]
DEFAULT_ONLINE_GROUPS = ["p2_page16_online"]

COMMON_BASE: Dict[str, object] = {
    "model_name": LOCAL_MODEL_PATH,
    "cpu_mem_gb": 32.0,
    "chunk_size": 1024,
    "max_new_tokens": 256,
    "decode_micro_batch_size": 0,
    "decode_active_cap_initial": 0,
    "max_decode_active_cap": 0,
    "sink_len": 16,
    "snapkv_observation_len": 16,
    "p2_sink_tokens": 16,
    "p2_recent_tokens": 16,
    "decode_path_mode": "rebuild",
    "decode_paged_flash_enabled": False,
}

LEGACY_OFF_RAW_ARGS: Dict[str, object] = {
    "retain_ratio": 1.0,
    "p2_enabled": False,
}

LEGACY_OFF_RAW_PAGE16_ARGS: Dict[str, object] = {
    **LEGACY_OFF_RAW_ARGS,
    "decode_path_mode": "page16_native",
    "decode_page16_native_strict": True,
}

OFF_RAW_PAGE16_ONLINE_ARGS: Dict[str, object] = {
    **LEGACY_OFF_RAW_PAGE16_ARGS,
    "kv_admission_enabled": True,
    "kv_admission_margin_blocks": 128,
}

OFF_COMPRESS_RATIO_BASE_ARGS: Dict[str, object] = {
    "retain_ratio": 0.10,
    "p2_enabled": False,
}

# Default compression profiles use the paper fixed budget. Ratio-based variants
# remain available only through explicit _rXXX group names.
OFF_COMPRESS_ARGS: Dict[str, object] = {
    "retain_budget_tokens": 2048,
    "selected_writeback_enabled": True,
    "p2_enabled": False,
}

OFF_COMPRESS_PAGE16_RATIO_BASE_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_RATIO_BASE_ARGS,
    "decode_path_mode": "page16_native",
    "decode_page16_native_strict": True,
}

OFF_COMPRESS_PAGE16_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_ARGS,
    "decode_path_mode": "page16_native",
    "decode_page16_native_strict": True,
}

P2_COMMON_ARGS: Dict[str, object] = {
    "p2_enabled": True,
    "p2_min_reclaim_blocks": 32,
    "p2_gain_window_steps": 8,
    "p2_gain_fail_cooldown_steps": 16,
    "offload_budget_blocks": 320,
    "prefetch_budget_blocks": 320,
    "offload_budget_blocks_max": 320,
    "prefetch_budget_blocks_max": 320,
    "ready_decode_eviction_threshold": 48,
    "p2_target_free_blocks": 0,
    "decode_active_cap_floor_ratio": 0.25,
    "p2_cuda_pressure_min_gb": 1.0,
    "p2_recent_tokens": 16,
    "kv_min_resident_ratio": 0.0,
}

P2_RATIO_BASE_ARGS: Dict[str, object] = {
    **P2_COMMON_ARGS,
    "retain_ratio": 0.10,
}

P2_ONLY_COMPRESS_ARGS: Dict[str, object] = {
    **P2_COMMON_ARGS,
    "retain_budget_tokens": 2048,
    "selected_writeback_enabled": True,
}

P2_PAGE16_RATIO_BASE_ARGS: Dict[str, object] = {
    **P2_RATIO_BASE_ARGS,
    "decode_path_mode": "page16_native",
    "decode_page16_native_strict": True,
}

P2_PAGE16_ARGS: Dict[str, object] = {
    **P2_ONLY_COMPRESS_ARGS,
    "decode_path_mode": "page16_native",
    "decode_page16_native_strict": True,
}

P2_PAGE16_OFFLINE_RATIO_BASE_ARGS: Dict[str, object] = {
    **P2_PAGE16_RATIO_BASE_ARGS,
    "kv_admission_enabled": False,
}

# Offline/capacity profile: fixed-budget page16 compression with P2 ready
# offload/prefetch. Keep admission off to preserve fixed-batch semantics.
P2_PAGE16_OFFLINE_ARGS: Dict[str, object] = {
    **P2_PAGE16_ARGS,
    "kv_admission_enabled": False,
}

# Offline fixed-budget profile used for synthetic long-input/long-output capacity.
# KV admission stays disabled; submitted requests beyond the planned active cap
# are managed by P2 ready-block offload/prefetch.
P2_PAGE16_OFFLINE_B2048_ARGS: Dict[str, object] = {
    **P2_PAGE16_OFFLINE_ARGS,
    "retain_budget_tokens": 2048,
    "selected_writeback_enabled": True,
    "kv_admission_enabled": False,
    "online_prefill_admission_enabled": False,
    "wm_low_ratio": 0.08,
    "p2_ready_reclaim_margin_blocks": 128,
    "p2_max_ready_sequences_per_step": 4,
    "p2_max_ready_offload_blocks_per_step": 320,
}

# Online/continuous profile: prevent active-decode future KV growth from
# exhausting the pool. P2 ready reclaim remains enabled as a fallback only.
P2_PAGE16_ONLINE_ARGS: Dict[str, object] = {
    **P2_PAGE16_ARGS,
    "kv_admission_enabled": True,
    "kv_admission_margin_blocks": 128,
    "p2_ready_reclaim_margin_blocks": 128,
    "p2_max_ready_sequences_per_step": 4,
    "p2_max_ready_offload_blocks_per_step": 320,
}

# Online fixed-budget profile used for ShareGPT clean292 gpu_mem_frac sweep.
# retain_budget_tokens takes precedence over retain_ratio inside SnapKV.
P2_PAGE16_ONLINE_B2048_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_ARGS,
    "retain_budget_tokens": 2048,
    "selected_writeback_enabled": True,
    "kv_admission_margin_blocks": 0,
    "kv_admission_include_low_watermark": True,
    "wm_low_ratio": 0.10,
    "kv_admission_output_reserve_tokens": 256,
    # Keep global prefill concurrency high enough for short prompts; pressure is
    # controlled by bucket caps plus a total active prompt-token budget.
    "max_prefill_active": 16,
    # Unified prefill admission: keep bucket caps non-binding and let total
    # active prompt-token budget control dense prefill pressure.
    "online_prefill_admission_enabled": True,
    "online_prefill_short_threshold_tokens": 4096,
    "online_prefill_mid_threshold_tokens": 8192,
    "online_prefill_cap_short": 16,
    "online_prefill_cap_mid": 16,
    "online_prefill_cap_long": 16,
    # 0 means auto: floor(M_prefill / m_kv), aligned down by the engine.
    "online_prefill_active_token_budget": 0,
    "online_prefill_admission_lookahead": 8,
    "online_prefill_cuda_headroom_gb": 0.5,
    "online_prefill_min_effective_chunk": 128,
}

# Online b2048 admission/P2 tuning probes. Keep fixed-budget + page16 strict
# semantics, but make P2 ready reclaim emergency-only by removing the extra
# reclaim margin. These profiles are for ShareGPT clean292 convergence only.
P2_PAGE16_ONLINE_B2048_WM010_P2EMERG_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_ARGS,
    "wm_low_ratio": 0.10,
    "kv_admission_margin_blocks": 0,
    "kv_admission_include_low_watermark": True,
    "p2_ready_reclaim_margin_blocks": 0,
}

P2_PAGE16_ONLINE_B2048_WM008_P2EMERG_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_ARGS,
    "wm_low_ratio": 0.08,
    "kv_admission_margin_blocks": 0,
    "kv_admission_include_low_watermark": True,
    "p2_ready_reclaim_margin_blocks": 0,
}

P2_PAGE16_ONLINE_B2048_WM005_P2EMERG_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_ARGS,
    "wm_low_ratio": 0.05,
    "kv_admission_margin_blocks": 0,
    "kv_admission_include_low_watermark": True,
    "p2_ready_reclaim_margin_blocks": 0,
}

P2_PAGE16_ONLINE_B2048_WM010_NOP2_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_WM010_P2EMERG_ARGS,
    "p2_enabled": False,
}

P2_PAGE16_ONLINE_B2048_WM008_NOP2_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_WM008_P2EMERG_ARGS,
    "p2_enabled": False,
}

P2_PAGE16_ONLINE_B2048_WM005_NOP2_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_WM005_P2EMERG_ARGS,
    "p2_enabled": False,
}


# Diagnostic only: remove KV-pool admission and let P2 handle runtime pool
# pressure. Keep online prefill admission because it guards CUDA prefill peaks,
# not KV pool occupancy. This is not a default candidate unless it proves stable.
P2_PAGE16_ONLINE_B2048_NO_KVADM_P2_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_ARGS,
    "kv_admission_enabled": False,
    "kv_admission_margin_blocks": 0,
    "kv_admission_include_low_watermark": False,
    "wm_low_ratio": 0.10,
    "p2_ready_reclaim_margin_blocks": 128,
    "p2_enabled": True,
}


# Reproduce the high-throughput online b2048 baseline before low-watermark was
# added to KV admission. This keeps the original margin-only admission behavior.
P2_PAGE16_ONLINE_B2048_BASELINE_REPRO_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_ARGS,
    "kv_admission_enabled": True,
    "kv_admission_margin_blocks": 128,
    "kv_admission_include_low_watermark": False,
    "wm_low_ratio": 0.15,
    "p2_ready_reclaim_margin_blocks": 128,
    "p2_enabled": True,
}

# Slightly more aggressive margin-only admission probe. No low-watermark hard
# reserve; keeps a small fixed margin to avoid fully unconstrained pending.
P2_PAGE16_ONLINE_B2048_MARGIN64_NOLOWWM_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_BASELINE_REPRO_ARGS,
    "kv_admission_margin_blocks": 64,
}


# Online mainline candidates: admission-driven, no P2 offload. Keep low-watermark
# out of admission and prevent generic pressure downscale from pinning decode cap
# at 2. These are the clean online profiles to compare with the historical
# high-throughput b2048 result.
P2_PAGE16_ONLINE_B2048_NOP2_MARGIN128_FLOOR4_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_ARGS,
    "p2_enabled": False,
    "kv_admission_enabled": True,
    "kv_admission_margin_blocks": 128,
    "kv_admission_include_low_watermark": False,
    "wm_low_ratio": 0.15,
    "decode_active_cap_min": 4,
    "decode_active_cap_floor_ratio": 0.80,
}

P2_PAGE16_ONLINE_B2048_NOP2_MARGIN64_FLOOR4_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_MARGIN128_FLOOR4_ARGS,
    "kv_admission_margin_blocks": 64,
}


# Online unified low-watermark admission: no P2 offload in online serving, but
# reserve the shared P2 low watermark as KV-pool safety headroom.
P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_ARGS,
    "p2_enabled": False,
    "kv_admission_enabled": True,
    "kv_admission_margin_blocks": 0,
    "kv_admission_include_low_watermark": True,
    "wm_low_ratio": 0.10,
    "decode_active_cap_min": 4,
    "decode_active_cap_floor_ratio": 0.80,
}


# Same unified low-watermark admission as lowwm010_floor4, but reduce per-request
# output reserve to 128 tokens. The 10% pool low watermark supplies global slack.
P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_OUT128_FLOOR4_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS,
    "kv_admission_output_reserve_tokens": 128,
}


# Higher minimum decode-active probes for online low-watermark admission. These
# test whether floor4 is overly conservative under the no-P2 online path.
P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_OUT128_FLOOR6_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_OUT128_FLOOR4_ARGS,
    "decode_active_cap_min": 6,
    "decode_active_cap_floor_ratio": 0.85,
}

P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_OUT128_FLOOR8_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_OUT128_FLOOR4_ARGS,
    "decode_active_cap_min": 8,
    "decode_active_cap_floor_ratio": 0.90,
}


# Higher minimum decode-active probes for the clean online default candidate:
# 10% low watermark, output reserve 256, no P2 offload.
P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR6_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS,
    "decode_active_cap_min": 6,
    "decode_active_cap_floor_ratio": 0.85,
}

P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR8_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS,
    "decode_active_cap_min": 8,
    "decode_active_cap_floor_ratio": 0.90,
}

# Final online stress probe: loosen low-watermark reserve to 8% and set the
# decode floor from the resulting KV-pool capacity calculation. If this does
# not beat lowwm010_floor6 cleanly, keep lowwm010_floor6 as the default.
P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR7_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS,
    "wm_low_ratio": 0.08,
    "decode_active_cap_min": 7,
    "decode_active_cap_floor_ratio": 0.875,
}

P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR5_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS,
    "wm_low_ratio": 0.08,
    "decode_active_cap_min": 5,
    "decode_active_cap_floor_ratio": 0.8,
}

P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR6_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS,
    "wm_low_ratio": 0.08,
    "decode_active_cap_min": 6,
    "decode_active_cap_floor_ratio": 0.85,
}

P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR8_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS,
    "wm_low_ratio": 0.08,
    "decode_active_cap_min": 8,
    "decode_active_cap_floor_ratio": 0.9,
}

P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR9_ARGS: Dict[str, object] = {
    **P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS,
    "wm_low_ratio": 0.08,
    "decode_active_cap_min": 9,
    "decode_active_cap_floor_ratio": 0.90,
}



OFF_COMPRESS_PAGE16_R015_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_PAGE16_RATIO_BASE_ARGS,
    "retain_ratio": 0.15,
}

# Fixed-budget quality profiles use selected-block writeback by default.
# The legacy materializing writeback path remains available for debugging via
# KV_MIDDLEWARE_DISABLE_SELECTED_WRITEBACK=1.
OFF_COMPRESS_PAGE16_B1024_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_PAGE16_ARGS,
    "retain_budget_tokens": 1024,
    "selected_writeback_enabled": True,
}

OFF_COMPRESS_PAGE16_B2048_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_PAGE16_ARGS,
    "retain_budget_tokens": 2048,
    "selected_writeback_enabled": True,
}

OFF_COMPRESS_PAGE16_B4096_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_PAGE16_ARGS,
    "retain_budget_tokens": 4096,
    "selected_writeback_enabled": True,
}

# 4.8 ablation-only profile: fixed-budget compression with selected writeback,
# but decode falls back to dense PKV rebuild instead of Block-Paged Direct Decode.
OFF_COMPRESS_B2048_REBUILD_ABLATION_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_ARGS,
    "retain_budget_tokens": 2048,
    "block_size": 16,
    "selected_writeback_enabled": True,
    "decode_path_mode": "rebuild",
    "decode_page16_native_strict": False,
    "decode_paged_flash_enabled": False,
    "p2_enabled": False,
    "kv_admission_enabled": False,
}


SNAPKV_DENSE_B2048_ARGS: Dict[str, object] = {
    # SnapKV-style token-level dense baseline: 1-token selection granularity,
    # no page16 direct decode, no selected-writeback, no P2/offload.
    **OFF_COMPRESS_ARGS,
    "block_size": 1,
    "sink_len": 16,
    "snapkv_observation_len": 16,
    "retain_budget_tokens": 2048,
    "decode_path_mode": "rebuild",
    "decode_page16_native_strict": False,
    "decode_paged_flash_enabled": False,
    "selected_writeback_enabled": False,
    "p2_enabled": False,
}

P2_PAGE16_OFFLINE_R015_ARGS: Dict[str, object] = {
    **P2_PAGE16_OFFLINE_RATIO_BASE_ARGS,
    "retain_ratio": 0.15,
}

P2_PAGE16_OFFLINE_R015_HP_SANITY_ARGS: Dict[str, object] = {
    **P2_PAGE16_OFFLINE_R015_ARGS,
    # Quality sanity only: force ready_decode pressure so P2 offload/prefetch is exercised.
    "decode_active_cap_initial": 1,
    "max_decode_active_cap": 1,
    "p2_target_free_blocks": 4500,
    "p2_cuda_pressure_min_gb": 999.0,
    "p2_ready_reclaim_margin_blocks": 128,
    "p2_max_ready_sequences_per_step": 4,
    "p2_max_ready_offload_blocks_per_step": 320,
}

P2_PAGE16_OFFLINE_R015_HP_SANITY_V2_ARGS: Dict[str, object] = {
    **P2_PAGE16_OFFLINE_R015_ARGS,
    # Quality sanity only: raise the actual P2 trigger watermarks so ready_decode
    # offload is exercised without waiting for a natural OOM/low-pool event.
    "decode_active_cap_initial": 1,
    "max_decode_active_cap": 1,
    "wm_low_ratio": 0.98,
    "wm_high_ratio": 0.995,
    "p2_target_free_blocks": 0,
    "p2_cuda_pressure_min_gb": 1.0,
    "p2_ready_reclaim_margin_blocks": 48,
    "p2_max_ready_sequences_per_step": 4,
    "p2_max_ready_offload_blocks_per_step": 320,
}

OFF_COMPRESS_PAGE16_R020_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_PAGE16_RATIO_BASE_ARGS,
    "retain_ratio": 0.20,
}

OFF_COMPRESS_PAGE16_R030_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_PAGE16_RATIO_BASE_ARGS,
    "retain_ratio": 0.30,
}

OFF_COMPRESS_PAGE16_R050_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_PAGE16_RATIO_BASE_ARGS,
    "retain_ratio": 0.50,
}

OFF_COMPRESS_PAGE16_R010_S64O64_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_PAGE16_RATIO_BASE_ARGS,
    "retain_ratio": 0.10,
    "sink_len": 64,
    "snapkv_observation_len": 64,
}

OFF_COMPRESS_PAGE16_R015_S64O64_ARGS: Dict[str, object] = {
    **OFF_COMPRESS_PAGE16_RATIO_BASE_ARGS,
    "retain_ratio": 0.15,
    "sink_len": 64,
    "snapkv_observation_len": 64,
}

P2_PAGE16_OFFLINE_R020_ARGS: Dict[str, object] = {
    **P2_PAGE16_OFFLINE_RATIO_BASE_ARGS,
    "retain_ratio": 0.20,
}

P2_PAGE16_OFFLINE_R015_S64O64_ARGS: Dict[str, object] = {
    **P2_PAGE16_OFFLINE_RATIO_BASE_ARGS,
    "retain_ratio": 0.15,
    "sink_len": 64,
    "snapkv_observation_len": 64,
}

GROUP_ARGS: Dict[str, Dict[str, object]] = {
    "legacy_off_raw": LEGACY_OFF_RAW_ARGS,
    "legacy_off_raw_page16": LEGACY_OFF_RAW_PAGE16_ARGS,
    "off_raw_page16_online": OFF_RAW_PAGE16_ONLINE_ARGS,
    "off_compress": OFF_COMPRESS_ARGS,
    "off_compress_page16": OFF_COMPRESS_PAGE16_ARGS,
    "off_compress_page16_r015": OFF_COMPRESS_PAGE16_R015_ARGS,
    "off_compress_page16_b1024": OFF_COMPRESS_PAGE16_B1024_ARGS,
    "off_compress_page16_b2048": OFF_COMPRESS_PAGE16_B2048_ARGS,
    "off_compress_b2048_rebuild_ablation": OFF_COMPRESS_B2048_REBUILD_ABLATION_ARGS,
    "off_compress_page16_b4096": OFF_COMPRESS_PAGE16_B4096_ARGS,
    "snapkv_dense_b2048": SNAPKV_DENSE_B2048_ARGS,
    "off_compress_page16_r020": OFF_COMPRESS_PAGE16_R020_ARGS,
    "off_compress_page16_r030": OFF_COMPRESS_PAGE16_R030_ARGS,
    "off_compress_page16_r050": OFF_COMPRESS_PAGE16_R050_ARGS,
    "off_compress_page16_r010_s64o64": OFF_COMPRESS_PAGE16_R010_S64O64_ARGS,
    "off_compress_page16_r015_s64o64": OFF_COMPRESS_PAGE16_R015_S64O64_ARGS,
    "p2_only_compress": P2_ONLY_COMPRESS_ARGS,
    # Backward-compatible name for the offline/capacity profile.
    "p2_page16": P2_PAGE16_OFFLINE_ARGS,
    "p2_page16_offline": P2_PAGE16_OFFLINE_ARGS,
    "p2_page16_offline_b2048": P2_PAGE16_OFFLINE_B2048_ARGS,
    "p2_page16_offline_r015": P2_PAGE16_OFFLINE_R015_ARGS,
    "p2_page16_offline_r015_hp_sanity": P2_PAGE16_OFFLINE_R015_HP_SANITY_ARGS,
    "p2_page16_offline_r015_hp_sanity_v2": P2_PAGE16_OFFLINE_R015_HP_SANITY_V2_ARGS,
    "p2_page16_offline_r020": P2_PAGE16_OFFLINE_R020_ARGS,
    "p2_page16_offline_r015_s64o64": P2_PAGE16_OFFLINE_R015_S64O64_ARGS,
    "p2_page16_online": P2_PAGE16_ONLINE_ARGS,
    "p2_page16_online_b2048": P2_PAGE16_ONLINE_B2048_ARGS,
    "p2_page16_online_b2048_nop2_lowwm010_floor6": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR6_ARGS,
    "p2_page16_online_b2048_nop2_lowwm010_floor8": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR8_ARGS,
    "p2_page16_online_b2048_nop2_lowwm008_floor7": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR7_ARGS,
    "p2_page16_online_b2048_nop2_lowwm008_floor8": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR8_ARGS,
    "p2_page16_online_b2048_nop2_lowwm008_floor9": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR9_ARGS,
    "p2_page16_online_b2048_nop2_lowwm008_floor6": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR6_ARGS,
    "p2_page16_online_b2048_nop2_lowwm008_floor5": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM008_FLOOR5_ARGS,
    "p2_page16_online_b2048_nop2_lowwm010_out128_floor6": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_OUT128_FLOOR6_ARGS,
    "p2_page16_online_b2048_nop2_lowwm010_out128_floor8": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_OUT128_FLOOR8_ARGS,
    "p2_page16_online_b2048_nop2_lowwm010_out128_floor4": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_OUT128_FLOOR4_ARGS,
    "p2_page16_online_b2048_nop2_lowwm010_floor4": P2_PAGE16_ONLINE_B2048_NOP2_LOWWM010_FLOOR4_ARGS,
    "p2_page16_online_b2048_nop2_margin128_floor4": P2_PAGE16_ONLINE_B2048_NOP2_MARGIN128_FLOOR4_ARGS,
    "p2_page16_online_b2048_nop2_margin64_floor4": P2_PAGE16_ONLINE_B2048_NOP2_MARGIN64_FLOOR4_ARGS,
    "p2_page16_online_b2048_baseline_repro": P2_PAGE16_ONLINE_B2048_BASELINE_REPRO_ARGS,
    "p2_page16_online_b2048_margin64_nolowwm": P2_PAGE16_ONLINE_B2048_MARGIN64_NOLOWWM_ARGS,
    "p2_page16_online_b2048_no_kvadm_p2": P2_PAGE16_ONLINE_B2048_NO_KVADM_P2_ARGS,
    "p2_page16_online_b2048_wm010_p2emerg": P2_PAGE16_ONLINE_B2048_WM010_P2EMERG_ARGS,
    "p2_page16_online_b2048_wm008_p2emerg": P2_PAGE16_ONLINE_B2048_WM008_P2EMERG_ARGS,
    "p2_page16_online_b2048_wm005_p2emerg": P2_PAGE16_ONLINE_B2048_WM005_P2EMERG_ARGS,
    "p2_page16_online_b2048_wm010_nop2": P2_PAGE16_ONLINE_B2048_WM010_NOP2_ARGS,
    "p2_page16_online_b2048_wm008_nop2": P2_PAGE16_ONLINE_B2048_WM008_NOP2_ARGS,
    "p2_page16_online_b2048_wm005_nop2": P2_PAGE16_ONLINE_B2048_WM005_NOP2_ARGS,
}

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
ARCHIVE_DIR = BENCHMARKS_DIR / "archive"
