# Cleanup Policy

This project keeps the working tree clean for reproducible experiments.

## Rules
- Do not keep ad-hoc backup files (`*.bak_*`) in the project root.
- Keep only the latest 2 experiment rounds under `experiment_results/paper_unified_*`.
- Keep only the latest 2 benchmark logs (`*.log`) in the project root.
- Keep model cache (`huggingface_cache/`) intact unless explicitly requested.

## Notes
- If you need archival artifacts, move them to a dedicated archive location in a follow-up task.
- `__pycache__` can be cleared at any time; it will be rebuilt automatically.
