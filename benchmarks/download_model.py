#!/usr/bin/env python3
"""Download Meta-Llama-3.1-8B-Instruct to local data disk with verification."""

import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TARGET_DIR = Path("/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct")
HF_HOME = Path("/root/autodl-tmp/huggingface_cache")
META_DIR = Path("/root/autodl-tmp/.autodl/kv_cache_middleware/results/experiment_results")


def _find_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text(encoding="utf-8").strip()
    return ""


def _verify_files(base: Path) -> dict:
    required = ["config.json", "tokenizer.json"]
    missing = [name for name in required if not (base / name).exists()]
    safetensors = sorted(base.glob("model-*.safetensors"))
    return {
        "missing_required": missing,
        "safetensors_count": len(safetensors),
        "ok": (len(missing) == 0 and len(safetensors) > 0),
    }


def main() -> int:
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    HF_HOME.mkdir(parents=True, exist_ok=True)
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    token = _find_token()
    if not token:
        print("ERROR: HF_TOKEN is not set and no local Hugging Face token was found.")
        print("Please run: export HF_TOKEN=<your_token> && /root/miniconda3/bin/python benchmarks/download_model.py")
        return 2

    api = HfApi(token=token)
    try:
        info = api.model_info(REPO_ID)
        revision = str(info.sha)
    except Exception as exc:
        print(f"ERROR: cannot access model info for {REPO_ID}: {exc}")
        print("Hint: confirm you accepted the model license and token has read access.")
        return 3

    print(f"Downloading {REPO_ID}")
    print(f"Target dir: {TARGET_DIR}")
    print(f"Revision: {revision}")

    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(TARGET_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
        )
    except Exception as exc:
        print(f"ERROR: download failed: {exc}")
        return 4

    verify = _verify_files(TARGET_DIR)
    if not verify["ok"]:
        print(f"ERROR: downloaded files incomplete: {verify}")
        return 5

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "repo_id": REPO_ID,
        "target_dir": str(TARGET_DIR),
        "revision": revision,
        "verify": verify,
    }
    meta_path = META_DIR / "model_fetch_meta_llama31_8b_instruct.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("OK: model download and verification completed.")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
