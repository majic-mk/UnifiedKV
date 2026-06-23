#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

from transformers import AutoTokenizer, GenerationConfig


ALL_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]
RAW_TASKS = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--longbench-root",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/LongBench"),
    )
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--gate-size", type=int, default=20)
    return parser.parse_args()


def hash_token_ids(token_ids):
    payload = b"".join(struct.pack("<I", int(token_id)) for token_id in token_ids)
    return hashlib.sha256(payload).hexdigest()


def middle_truncate(token_ids, limit):
    if len(token_ids) <= limit:
        return list(token_ids), 0
    left = limit // 2
    right = limit - left
    return list(token_ids[:left]) + list(token_ids[-right:]), len(token_ids)


def generation_eos_ids(model_path, tokenizer):
    config = GenerationConfig.from_pretrained(model_path)
    eos_ids = config.eos_token_id
    if eos_ids is None:
        eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    return list(dict.fromkeys(int(token_id) for token_id in (eos_ids or [])))


def encode_prompt(tokenizer, task, prompt, is_qwen):
    if task in RAW_TASKS:
        return tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
        ).input_ids
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if is_qwen:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **kwargs,
    )


def stratified_gate_indices(rows, gate_size):
    if gate_size <= 0 or len(rows) <= gate_size:
        return {int(row["row_index"]) for row in rows}
    ranked = sorted(rows, key=lambda row: (int(row["input_tokens"]), int(row["row_index"])))
    positions = {
        round(index * (len(ranked) - 1) / (gate_size - 1))
        for index in range(gate_size)
    }
    return {int(ranked[position]["row_index"]) for position in sorted(positions)}


def main():
    args = parse_args()
    repo = args.longbench_root / "repo" / "LongBench"
    data_root = args.longbench_root / "data" / "data"
    prompts = json.loads(
        (repo / "config" / "dataset2prompt.json").read_text(encoding="utf-8")
    )
    maxgens = json.loads(
        (repo / "config" / "dataset2maxlen.json").read_text(encoding="utf-8")
    )
    tasks = ALL_TASKS if args.tasks == "all" else args.tasks.split(",")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_type = str(getattr(tokenizer, "name_or_path", args.model)).lower()
    is_qwen = "qwen" in model_type
    base_eos_ids = generation_eos_ids(args.model, tokenizer)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol_version": "longbench-frozen-v1",
        "model": str(args.model),
        "tasks": tasks,
        "raw_tasks": sorted(RAW_TASKS),
        "chat_tasks": [task for task in tasks if task not in RAW_TASKS],
        "chat_template": "model-native",
        "qwen_enable_thinking": False if is_qwen else None,
        "custom_system_prompt": False,
        "max_input_tokens_after_template": int(args.max_input_tokens),
        "truncation": "middle-on-final-token-ids",
        "base_eos_token_ids": base_eos_ids,
        "samsum_extra_eos_token_id": int(newline_ids[-1]),
        "dtype": "float16",
        "greedy": True,
        "num_beams": 1,
        "min_new_tokens": 1,
        "task_files": {},
    }

    for task in tasks:
        source = data_root / f"{task}.jsonl"
        rows = []
        for row_index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = prompts[task].format(**record)
            original_ids = encode_prompt(tokenizer, task, prompt, is_qwen)
            input_ids, truncated_from = middle_truncate(
                original_ids,
                int(args.max_input_tokens),
            )
            eos_ids = list(base_eos_ids)
            if task == "samsum":
                eos_ids = list(dict.fromkeys(eos_ids + [int(newline_ids[-1])]))
            rows.append(
                {
                    "task": task,
                    "row_index": row_index,
                    "input_ids": input_ids,
                    "input_tokens": len(input_ids),
                    "truncated_from_tokens": truncated_from,
                    "input_sha256": hash_token_ids(input_ids),
                    "max_new_tokens": int(maxgens[task]),
                    "eos_token_ids": eos_ids,
                    "prompt_mode": "raw" if task in RAW_TASKS else "chat_template",
                    "answers": record["answers"],
                    "all_classes": record.get("all_classes", []),
                    "official_length": record.get("length"),
                }
            )
        gate_indices = stratified_gate_indices(rows, int(args.gate_size))
        for row in rows:
            row["gate_selected"] = int(row["row_index"]) in gate_indices
        output_path = args.output_dir / f"{task}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metadata["task_files"][task] = {
            "path": str(output_path),
            "rows": len(rows),
            "gate_rows": len(gate_indices),
            "min_input_tokens": min(row["input_tokens"] for row in rows),
            "max_input_tokens": max(row["input_tokens"] for row in rows),
        }
        print(
            f"{task}: rows={len(rows)} gate={len(gate_indices)} "
            f"tokens={metadata['task_files'][task]['min_input_tokens']}-"
            f"{metadata['task_files'][task]['max_input_tokens']}",
            flush=True,
        )

    (args.output_dir / "manifest_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
