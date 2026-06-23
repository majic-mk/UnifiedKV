#!/usr/bin/env python3
import argparse, json, math, random
from pathlib import Path
from statistics import mean, median

TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]
METHODS = ["hf_vanilla", "off_compress_page16_b1024", "off_compress_page16_b2048", "off_compress_page16_b4096"]
LABELS = {
    "hf_vanilla": "HF vanilla",
    "off_compress_page16_b1024": "BP-KV-Compress 1024",
    "off_compress_page16_b2048": "BP-KV-Compress 2048",
    "off_compress_page16_b4096": "BP-KV-Compress 4096",
}
CATEGORIES = {
    "narrativeqa": "Single-Doc QA", "qasper": "Single-Doc QA", "multifieldqa_en": "Single-Doc QA",
    "hotpotqa": "Multi-Doc QA", "2wikimqa": "Multi-Doc QA", "musique": "Multi-Doc QA",
    "gov_report": "Summarization", "qmsum": "Summarization", "multi_news": "Summarization",
    "trec": "Few-shot", "triviaqa": "Few-shot", "samsum": "Few-shot",
    "passage_count": "Synthetic", "passage_retrieval_en": "Synthetic",
    "lcc": "Code", "repobench-p": "Code",
}
CAT_ORDER = ["Single-Doc QA", "Multi-Doc QA", "Summarization", "Few-shot", "Synthetic", "Code"]

def load_result(input_dir: Path, method: str, task: str):
    p = input_dir / f"longbench_full_official_{method}_{task}.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
        if not d.get("rows"):
            return None
        row = d["rows"][0]
        row["_file"] = str(p)
        return row
    except Exception as exc:
        return {"method": method, "task": task, "category": CATEGORIES.get(task, "Other"), "score": 0.0, "status": "ParseFailed", "error_reason": str(exc), "rows": []}

def is_completed(row):
    if not row:
        return False
    return float(row.get("completion_rate", 0.0) or 0.0) >= 1.0 and float(row.get("valid_completion_rate", 0.0) or 0.0) >= 1.0 and not str(row.get("error_reason", "")).strip()

def fmt(x):
    return "-" if x is None else f"{float(x):.2f}"

def main_table(results):
    out = []
    for method in METHODS:
        vals_by_cat = {c: [] for c in CAT_ORDER}
        all_scores = []
        completed = 0
        failed = 0
        for task in TASKS:
            row = results.get((method, task))
            if row and is_completed(row):
                score = float(row.get("score", 0.0) or 0.0)
                vals_by_cat[CATEGORIES[task]].append(score)
                all_scores.append(score)
                completed += 1
            else:
                failed += 1
        item = {"Method": LABELS.get(method, method)}
        for cat in CAT_ORDER:
            item[cat] = round(mean(vals_by_cat[cat]), 2) if vals_by_cat[cat] else "Failed/OOM"
        item["Avg"] = round(mean(all_scores), 2) if all_scores else "NA"
        item["Completed/Total"] = f"{completed}/{len(TASKS)}"
        item["Failed"] = failed
        out.append(item)
    return out

def per_task_table(results):
    rows = []
    for task in TASKS:
        item = {"Task": task, "Category": CATEGORIES[task]}
        for method in METHODS:
            row = results.get((method, task))
            item[LABELS.get(method, method)] = round(float(row.get("score", 0.0) or 0.0), 2) if row and is_completed(row) else "Failed/OOM"
        rows.append(item)
    return rows

def sample_rows(row):
    return list(row.get("rows", []) or []) if row else []

def paired_analysis(results, method_a="off_compress_page16_b2048", method_b="off_compress_page16_b4096", n_boot=5000, seed=20260415):
    pairs = []
    per_task = []
    for task in TASKS:
        a = results.get((method_a, task))
        b = results.get((method_b, task))
        rows_a = {int(r.get("sample_idx", -1)): r for r in sample_rows(a) if "sample_idx" in r}
        rows_b = {int(r.get("sample_idx", -1)): r for r in sample_rows(b) if "sample_idx" in r}
        common = sorted(set(rows_a) & set(rows_b))
        diffs = []
        win = tie = loss = 0
        for idx in common:
            da = float(rows_a[idx].get("score", 0.0) or 0.0)
            db = float(rows_b[idx].get("score", 0.0) or 0.0)
            diff = db - da
            diffs.append(diff)
            pairs.append({"task": task, "sample_idx": idx, "method_a": da, "method_b": db, "delta": diff})
            if diff > 1e-9: win += 1
            elif diff < -1e-9: loss += 1
            else: tie += 1
        per_task.append({
            "task": task,
            "n_pairs": len(diffs),
            "mean_delta": round(mean(diffs), 4) if diffs else None,
            "median_delta": round(median(diffs), 4) if diffs else None,
            "win/tie/loss": f"{win}/{tie}/{loss}",
        })
    deltas = [p["delta"] for p in pairs]
    rng = random.Random(seed)
    ci = None
    if deltas:
        boots = []
        n = len(deltas)
        for _ in range(n_boot):
            boots.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
        boots.sort()
        ci = [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]]
    return {
        "n_pairs": len(deltas),
        "mean_delta": round(mean(deltas), 4) if deltas else None,
        "median_delta": round(median(deltas), 4) if deltas else None,
        "bootstrap_95_ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
        "per_task": per_task,
    }

def markdown_table(rows):
    if not rows:
        return ""
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {(m, t): load_result(input_dir, m, t) for m in METHODS for t in TASKS}
    main_rows = main_table(results)
    task_rows = per_task_table(results)
    paired = paired_analysis(results)

    hf_avg = next((r["Avg"] for r in main_rows if r["Method"] == "HF vanilla"), None)
    ablation = []
    for method, budget, note in [
        ("off_compress_page16_b1024", "1024", "aggressive fixed budget"),
        ("off_compress_page16_b2048", "2048", "default fixed budget candidate"),
        ("off_compress_page16_b4096", "4096", "quality-oriented fixed budget"),
    ]:
        row = next(r for r in main_rows if r["Method"] == LABELS[method])
        pct = round(100.0 * float(row["Avg"]) / float(hf_avg), 2) if isinstance(row["Avg"], (int, float)) and isinstance(hf_avg, (int, float)) and hf_avg else "NA"
        ablation.append({"retain_budget_tokens": budget, "LongBench Avg": row["Avg"], "% HF": pct, "Completed/Total": row["Completed/Total"], "Conclusion": note})

    payload = {"main_table": main_rows, "per_task": task_rows, "fixed_budget_ablation": ablation, "paired_analysis": paired}
    (out_dir / "longbench_full_official_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# LongBench Full Official Summary")
    md.append("")
    md.append("Official LongBench prompts, metrics, max_gen, and full English task files. BP-KV rows should have `decode_rebuild_steps=0`, `decode_materialize_kv_bytes=0`, and `resident_miss_steps=0` in per-task JSONs.")
    md.append("")
    md.append("## Main Table")
    md.append(markdown_table(main_rows))
    md.append("")
    md.append("## Fixed Budget Ablation")
    md.append(markdown_table(ablation))
    md.append("")
    md.append("## Paired 4096 - 2048 Analysis")
    md.append(f"- n_pairs: {paired['n_pairs']}")
    md.append(f"- mean_delta: {paired['mean_delta']}")
    md.append(f"- median_delta: {paired['median_delta']}")
    md.append(f"- bootstrap_95_ci: {paired['bootstrap_95_ci']}")
    md.append("")
    md.append(markdown_table(paired["per_task"]))
    md.append("")
    md.append("## Per Task Scores")
    md.append(markdown_table(task_rows))
    (out_dir / "LONGBENCH_FULL_OFFICIAL_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_dir / "LONGBENCH_FULL_OFFICIAL_SUMMARY.md")

if __name__ == "__main__":
    main()