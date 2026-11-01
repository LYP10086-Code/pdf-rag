# -*- coding: utf-8 -*-
"""78 题检索消融对比：rerank 开关 × 父块归并开关。

用法：
    .venv/Scripts/python.exe tests/eval_compare.py [--api http://localhost:8080] [--k 5]

四种配置：
    baseline   直接检索（无重排、无归并）
    rerank     仅重排
    merge      仅父块归并
    full       重排 + 归并（当前默认）
判定口径与 eval_run.py 一致（L1 章节 / L2 页码 / L3 目标块精确命中）。
输出：tests/eval_compare_report.json + tests/eval_compare_report.md + 控制台对比表。
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from eval_run import judge

ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = ROOT / "tests" / "testset.json"
REPORT_JSON = ROOT / "tests" / "eval_compare_report.json"
REPORT_MD = ROOT / "tests" / "eval_compare_report.md"

CONFIGS = [
    ("baseline", "直接检索（无重排/无归并）", False, False),
    ("rerank", "仅重排", False, True),
    ("merge", "仅父块归并", True, False),
    ("full", "重排+归并（当前默认）", True, True),
]


def http_post_json(url: str, payload: dict, timeout: int = 180) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_doc_id(api: str, keyword: str) -> str:
    with urllib.request.urlopen(f"{api}/api/documents", timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for item in data["items"]:
        if keyword in item["filename"]:
            return item["id"]
    raise SystemExit(f"未找到文件名包含 {keyword!r} 的文档")


def run_config(
    api: str, doc_id: str, questions: list[dict], k: int, merge: bool, rerank: bool
) -> dict:
    results = []
    for index, question in enumerate(questions, 1):
        payload = {
            "query": question["question"],
            "k": k,
            "merge_parents": merge,
            "use_reranker": rerank,
        }
        try:
            response = http_post_json(f"{api}/api/documents/{doc_id}/search", payload)
            judged = judge(question, response["results"])
            judged["elapsed_ms"] = response.get("elapsed_ms")
        except Exception as exc:  # noqa: BLE001 单题失败不中断
            judged = {
                "id": question["id"],
                "type": question.get("type", ""),
                "L1": False,
                "L2": False,
                "L3": False,
                "error": str(exc),
                "elapsed_ms": None,
            }
        results.append(judged)
        if index % 10 == 0 or index == len(questions):
            print(f"    [{index:>2}/{len(questions)}]")
    total = len(results)

    def rate(key: str) -> dict:
        passed = sum(1 for r in results if r[key])
        return {"passed": passed, "rate": round(passed / total * 100, 1)}

    by_type: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_type.setdefault(r["type"], {"total": 0, "L1": 0, "L2": 0, "L3": 0})
        bucket["total"] += 1
        for key in ("L1", "L2", "L3"):
            bucket[key] += int(r[key])

    elapsed = [r["elapsed_ms"] for r in results if r.get("elapsed_ms")]
    return {
        "total": total,
        "L1": rate("L1"),
        "L2": rate("L2"),
        "L3": rate("L3"),
        "by_type": by_type,
        "avg_elapsed_ms": round(sum(elapsed) / len(elapsed)) if elapsed else None,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="78 题检索消融对比")
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--keyword", default="秦PLUS")
    args = parser.parse_args()

    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    questions = testset["questions"]
    doc_id = resolve_doc_id(args.api, args.keyword)
    print(f"评测文档: {doc_id}，共 {len(questions)} 题，k={args.k}，4 种配置")

    summary: dict[str, dict] = {}
    for key, label, merge, rerank in CONFIGS:
        print(f"[{key}] merge_parents={merge} use_reranker={rerank}")
        started = time.time()
        summary[key] = run_config(args.api, doc_id, questions, args.k, merge, rerank)
        summary[key]["label"] = label
        summary[key]["merge_parents"] = merge
        summary[key]["use_reranker"] = rerank
        print(
            f"    完成，用时 {time.time() - started:.0f}s："
            f"L1 {summary[key]['L1']['passed']}/{summary[key]['total']} "
            f"L2 {summary[key]['L2']['passed']}/{summary[key]['total']} "
            f"L3 {summary[key]['L3']['passed']}/{summary[key]['total']}"
        )

    REPORT_JSON.write_text(
        json.dumps(
            {"doc_id": doc_id, "k": args.k, "configs": summary},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# 78 题检索消融对比报告",
        "",
        f"- 文档：`{doc_id}`　k={args.k}　候选深度：{testset.get('meta', {}).get('note', 'RETRIEVAL_CANDIDATE_DEPTH=20')}",
        "",
        "| 配置 | L1 章节命中 | L2 页码命中 | L3 目标块命中 | 平均耗时ms |",
        "|---|---|---|---|---|",
    ]
    for key, label, _, _ in CONFIGS:
        s = summary[key]
        lines.append(
            f"| {label} | {s['L1']['passed']}/{s['total']}（{s['L1']['rate']}%） "
            f"| {s['L2']['passed']}/{s['total']}（{s['L2']['rate']}%） "
            f"| {s['L3']['passed']}/{s['total']}（{s['L3']['rate']}%） "
            f"| {s['avg_elapsed_ms']} |"
        )

    lines += ["", "## 按题型分布", ""]
    for key, label, _, _ in CONFIGS:
        lines += [f"### {label}", "", "| 题型 | 总数 | L1 | L2 | L3 |", "|---|---|---|---|---|"]
        for qtype, bucket in sorted(summary[key]["by_type"].items()):
            lines.append(
                f"| {qtype} | {bucket['total']} | {bucket['L1']} | {bucket['L2']} | {bucket['L3']} |"
            )
        lines.append("")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines[6:14]))
    print(f"报告：{REPORT_JSON}\n      {REPORT_MD}")


if __name__ == "__main__":
    main()
