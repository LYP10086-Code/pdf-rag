# -*- coding: utf-8 -*-
"""78 题检索评测基线 runner（tests/testset.json）。

用法：
    .venv/Scripts/python.exe tests/eval_run.py [--api http://localhost:8080] [--k 5] [--doc-id d_xxx]

判定口径（见 testset.json meta.判定口径）：
- L1 章节命中：top-k 中至少一个命中块的 section 与题目 section 一致
- L2 页码命中：L1 且该块的 pages 与题目 page_start~page_end 区间有重叠
- L3 目标块精确命中：命中块的 (pages, breadcrumb) 与 targets 任一完全相等
  （breadcrumb 剥去文档标题前缀后比较）
输出：tests/eval_report.json + tests/eval_report.md + 控制台摘要。
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = ROOT / "tests" / "testset.json"
REPORT_JSON = ROOT / "tests" / "eval_report.json"
REPORT_MD = ROOT / "tests" / "eval_report.md"


def http_post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_doc_id(api: str, keyword: str) -> str:
    data = http_get_json(f"{api}/api/documents")
    for item in data["items"]:
        if keyword in item["filename"]:
            return item["id"]
    raise SystemExit(f"未找到文件名包含 {keyword!r} 的文档，请用 --doc-id 指定")


def page_overlap(pages_str: str, page_start: int, page_end: int) -> bool:
    try:
        if "-" in pages_str:
            lo, hi = pages_str.split("-", 1)
            return int(lo) <= page_end and int(hi) >= page_start
        value = int(pages_str)
        return page_start <= value <= page_end
    except (ValueError, AttributeError, TypeError):
        return False


def strip_doc_prefix(breadcrumb: str, doc_name: str) -> str:
    prefix = f"{doc_name} > "
    return breadcrumb[len(prefix):] if breadcrumb.startswith(prefix) else breadcrumb


def judge(question: dict, hits: list[dict]) -> dict:
    l1_hits = [h for h in hits if h.get("section") == question["section"]]
    l2_pass = any(
        page_overlap(h.get("pages"), question["page_start"], question["page_end"])
        for h in l1_hits
    )
    target_set = {(t["pages"], t["breadcrumb"]) for t in question["targets"]}
    l3_pass = any(
        (h.get("pages"), strip_doc_prefix(h.get("breadcrumb", ""), h.get("doc_name", "")))
        in target_set
        for h in hits
    )
    return {
        "id": question["id"],
        "type": question.get("type", ""),
        "L1": bool(l1_hits),
        "L2": l2_pass,
        "L3": l3_pass,
        "top1": {
            "kind": hits[0].get("kind"),
            "pages": hits[0].get("pages"),
            "rerank": hits[0].get("rerank_score"),
        }
        if hits
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="78 题检索评测基线")
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--keyword", default="秦PLUS")
    args = parser.parse_args()

    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    questions = testset["questions"]
    doc_id = args.doc_id or resolve_doc_id(args.api, args.keyword)
    print(f"评测文档: {doc_id}，共 {len(questions)} 题，k={args.k}")

    results = []
    failures = []
    for index, question in enumerate(questions, 1):
        try:
            payload = {
                "query": question["question"],
                "k": args.k,
                "merge_parents": False,
            }
            response = http_post_json(
                f"{args.api}/api/documents/{doc_id}/search", payload
            )
            judged = judge(question, response["results"])
            judged["elapsed_ms"] = response.get("elapsed_ms")
        except Exception as exc:  # noqa: BLE001 单题失败不中断评测
            judged = {
                "id": question["id"],
                "type": question.get("type", ""),
                "L1": False,
                "L2": False,
                "L3": False,
                "error": str(exc),
            }
            failures.append(question["id"])
        results.append(judged)
        marks = f"L1={int(judged['L1'])} L2={int(judged['L2'])} L3={int(judged['L3'])}"
        print(f"[{index:>2}/{len(questions)}] {question['id']} {marks}")

    total = len(results)

    def rate(key: str) -> tuple[int, float]:
        passed = sum(1 for r in results if r[key])
        return passed, passed / total * 100

    n_l1, p_l1 = rate("L1")
    n_l2, p_l2 = rate("L2")
    n_l3, p_l3 = rate("L3")
    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_type.setdefault(r["type"], {"total": 0, "L1": 0, "L2": 0, "L3": 0})
        bucket["total"] += 1
        for key in ("L1", "L2", "L3"):
            bucket[key] += int(r[key])

    summary = {
        "doc_id": doc_id,
        "k": args.k,
        "total": total,
        "L1": {"passed": n_l1, "rate": round(p_l1, 1)},
        "L2": {"passed": n_l2, "rate": round(p_l2, 1)},
        "L3": {"passed": n_l3, "rate": round(p_l3, 1)},
        "by_type": by_type,
        "failed_requests": failures,
        "avg_elapsed_ms": round(
            sum(r.get("elapsed_ms") or 0 for r in results) / total
        ),
        "results": results,
    }
    REPORT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 78 题检索评测基线报告",
        "",
        f"- 文档：`{doc_id}`　k={args.k}　merge_parents=false（原始子块命中）",
        f"- **L1 章节命中：{n_l1}/{total}（{p_l1:.1f}%）**",
        f"- **L2 页码命中：{n_l2}/{total}（{p_l2:.1f}%）**",
        f"- **L3 目标块精确命中：{n_l3}/{total}（{p_l3:.1f}%）**",
        f"- 平均检索耗时：{summary['avg_elapsed_ms']}ms"
        + (f"　请求失败：{failures}" if failures else ""),
        "",
        "## 按题型分布",
        "",
        "| 题型 | 总数 | L1 | L2 | L3 |",
        "|---|---|---|---|---|",
    ]
    for qtype, bucket in sorted(by_type.items()):
        lines.append(
            f"| {qtype} | {bucket['total']} | {bucket['L1']} | {bucket['L2']} | {bucket['L3']} |"
        )
    lines += ["", "## 逐题结果", "", "| 题号 | 题型 | L1 | L2 | L3 | Top1 类型 | Top1 页码 | 耗时ms |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        top1 = r.get("top1") or {}
        lines.append(
            f"| {r['id']} | {r['type']} | {int(r['L1'])} | {int(r['L2'])} | {int(r['L3'])} "
            f"| {top1.get('kind') or '-'} | {top1.get('pages') or '-'} | {r.get('elapsed_ms', '-')} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"\n基线：L1 {n_l1}/{total}({p_l1:.1f}%)  L2 {n_l2}/{total}({p_l2:.1f}%)  "
        f"L3 {n_l3}/{total}({p_l3:.1f}%)"
    )
    print(f"报告：{REPORT_JSON}\n      {REPORT_MD}")


if __name__ == "__main__":
    main()
