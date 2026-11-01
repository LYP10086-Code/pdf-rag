# -*- coding: utf-8 -*-
"""诊断 L3 未命中题：并列展示命中块 (pages, breadcrumb) 与 targets。"""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from eval_run import strip_doc_prefix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
testset = json.loads((ROOT / "tests" / "testset.json").read_text(encoding="utf-8"))
report = json.loads((ROOT / "tests" / "eval_report.json").read_text(encoding="utf-8"))
failed_ids = [r["id"] for r in report["results"] if not r["L3"]]
qmap = {q["id"]: q for q in testset["questions"]}
DOC = report["doc_id"]


def search(query):
    body = json.dumps({"query": query, "k": 5, "merge_parents": False}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://localhost:8080/api/documents/{DOC}/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


for qid in failed_ids:
    q = qmap[qid]
    r = search(q["question"])
    doc_name = r["results"][0]["doc_name"] if r["results"] else ""
    print(f"\n### {qid} [{q['type']}] {q['question'][:30]}")
    print("  命中:")
    for h in r["results"]:
        print(f"    ({h['pages']}, {strip_doc_prefix(h['breadcrumb'], doc_name)}) rerank={round(h.get('rerank_score') or 0, 3)}")
    print("  targets:")
    for t in q["targets"]:
        print(f"    ({t['pages']}, {t['breadcrumb']})")
