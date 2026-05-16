#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import mean

TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en",
    "lcc", "repobench-p",
]
CATEGORIES = {
    "narrativeqa": "Single-Doc QA", "qasper": "Single-Doc QA", "multifieldqa_en": "Single-Doc QA",
    "hotpotqa": "Multi-Doc QA", "2wikimqa": "Multi-Doc QA", "musique": "Multi-Doc QA",
    "gov_report": "Summarization", "qmsum": "Summarization", "multi_news": "Summarization",
    "trec": "Few-shot", "triviaqa": "Few-shot", "samsum": "Few-shot",
    "passage_count": "Synthetic", "passage_retrieval_en": "Synthetic",
    "lcc": "Code", "repobench-p": "Code",
}
CAT_ORDER = ["Single-Doc QA", "Multi-Doc QA", "Summarization", "Few-shot", "Synthetic", "Code"]
METHODS = [
    ("hf_vanilla", "HF vanilla", [Path("benchmarks/results/paper/quality_v4/longbench_fixed_budget_llama"), Path("benchmarks/results/paper/quality_v3/longbench_full_official_llama")]),
    ("snapkv_dense_b2048", "SnapKV dense 2048", [Path("benchmarks/results/paper/quality_v8/longbench_snapkv_dense_b2048_llama")]),
    ("off_compress_page16_b1024", "UnifiedKV-Compress 1024", [Path("benchmarks/results/paper/quality_v5/longbench_fixed_budget_selected_writeback_llama"), Path("benchmarks/results/paper/quality_v4/longbench_fixed_budget_llama")]),
    ("off_compress_page16_b2048", "UnifiedKV-Compress 2048", [Path("benchmarks/results/paper/quality_v5/longbench_fixed_budget_selected_writeback_llama"), Path("benchmarks/results/paper/quality_v4/longbench_fixed_budget_llama")]),
    ("off_compress_page16_b4096", "UnifiedKV-Compress 4096", [Path("benchmarks/results/paper/quality_v5/longbench_fixed_budget_selected_writeback_llama"), Path("benchmarks/results/paper/quality_v4/longbench_fixed_budget_llama")]),
]
OUT_DIR = Path("benchmarks/results/paper/quality_v8/longbench_with_snapkv_summary")

def load(method, task, dirs):
    for d in dirs:
        p=d/f"longbench_full_official_{method}_{task}.json"
        if p.exists() and p.stat().st_size > 0:
            try:
                data=json.loads(p.read_text(encoding="utf-8"))
                rows=data.get("rows") or []
                if rows:
                    row=dict(rows[0]); row["_file"]=str(p); return row
            except Exception as e:
                return {"status":"ParseFailed","error_reason":str(e),"score":0.0,"_file":str(p)}
    return None

def completed(row):
    return bool(row) and float(row.get("completion_rate",0) or 0) >= 1.0 and float(row.get("valid_completion_rate",0) or 0) >= 1.0 and not str(row.get("error_reason","")).strip()

def md_table(rows):
    if not rows: return ""
    keys=list(rows[0].keys())
    out=["| "+" | ".join(keys)+" |", "|"+"|".join(["---"]*len(keys))+"|"]
    for r in rows:
        out.append("| "+" | ".join(str(r.get(k,"")) for k in keys)+" |")
    return "\n".join(out)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results={}
    for method,label,dirs in METHODS:
        for task in TASKS:
            results[(method,task)]=load(method,task,dirs)
    main_rows=[]; task_rows=[]; runtime_rows=[]
    for method,label,dirs in METHODS:
        vals_by_cat={c:[] for c in CAT_ORDER}; all_scores=[]; done=0; failed=0; total_s=0.0
        for task in TASKS:
            row=results.get((method,task))
            if completed(row):
                score=float(row.get("score",0) or 0)
                vals_by_cat[CATEGORIES[task]].append(score); all_scores.append(score); done+=1
                total_s += float(row.get("wall_clock_s", row.get("wall_clock_total_runtime_ms",0)/1000.0) or 0.0)
            else:
                failed+=1
        item={"Method":label}
        for cat in CAT_ORDER:
            item[cat]=round(mean(vals_by_cat[cat]),2) if vals_by_cat[cat] else "Failed/OOM"
        item["Avg"]=round(mean(all_scores),2) if all_scores else "NA"
        item["Completed/Total"]=f"{done}/{len(TASKS)}"; item["Failed"]=failed; item["Runtime(h)"]=round(total_s/3600.0,2) if total_s else "NA"
        main_rows.append(item)
    for task in TASKS:
        item={"Task":task,"Category":CATEGORIES[task]}
        for method,label,dirs in METHODS:
            row=results.get((method,task))
            item[label]=round(float(row.get("score",0) or 0),2) if completed(row) else "Failed/OOM"
        task_rows.append(item)
    for method,label,dirs in METHODS:
        for task in TASKS:
            row=results.get((method,task))
            runtime_rows.append({"Method":label,"Task":task,"Status":str(row.get("status","") if row else "Missing"),"Completed": int(completed(row)),"Score": round(float(row.get("score",0) or 0),2) if completed(row) else "Failed/OOM","Runtime(s)": round(float(row.get("wall_clock_s", row.get("wall_clock_total_runtime_ms",0)/1000.0) or 0.0),2) if row else 0.0,"File": row.get("_file","") if row else ""})
    payload={"main_table":main_rows,"per_task":task_rows,"runtime":runtime_rows}
    (OUT_DIR/"longbench_with_snapkv_summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    md=["# LongBench With SnapKV Dense 2048", "", "## Main Table", md_table(main_rows), "", "## Per Task Scores", md_table(task_rows), "", "## Runtime / Sources", md_table(runtime_rows)]
    (OUT_DIR/"LONGBENCH_WITH_SNAPKV_SUMMARY.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(OUT_DIR/"LONGBENCH_WITH_SNAPKV_SUMMARY.md")
if __name__ == "__main__": main()
