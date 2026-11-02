# -*- coding: utf-8 -*-
"""检索/生成指标评测：Recall@K、MRR、nDCG、引用准确率、Faithfulness 代理。

两种检索模式各测一遍：
- child（merge_parents=false）：相关判定 = (pages, breadcrumb) 与校准后 targets
  完全相等；Recall 分母 = 该题 distinct targets 数
- parent（merge_parents=true，父块归并）：targets 是子块粒度无法精确相等，
  相关判定 = evidence 证据覆盖率 ≥0.4（evidence 锚定，与父块粒度兼容）

生成指标（--phase chat，走 /api/chat 生产路径，父块归并 + 服务端引用）：
- 引用准确率：citations 中证据覆盖率 ≥0.4 的占比（citation 与父块检索
  top-CHUNK_K 结果按 (pages, breadcrumb) 一一对应取全文算覆盖）
- Faithfulness 代理①证据入上下文率：至少一个引用块覆盖证据的题目占比
- Faithfulness 代理②证据入答案率：答案文本对 evidence 的滑窗覆盖率 ≥0.3

输出：tests/eval_metrics.json + tests/eval_metrics.md（增量保存，可分阶段跑）。
"""

import argparse
import json
import math
import re
import time
from pathlib import Path

from eval_run import http_get_json, http_post_json, page_overlap, strip_doc_prefix

ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = ROOT / "tests" / "testset.json"
OUT_JSON = ROOT / "tests" / "eval_metrics.json"
OUT_MD = ROOT / "tests" / "eval_metrics.md"

K_LIST = (1, 3, 5, 10)
MRR_K = 10
CHUNK_K = 5           # chat 生产路径的检索 k（settings.retrieval_default_k）
COVERAGE_THRESHOLD = 0.4
ANSWER_COVERAGE_THRESHOLD = 0.3

_NORM = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]")


def norm(s: str) -> str:
    return _NORM.sub("", s or "")


def coverage(text: str, evidence: str) -> float:
    hit_norm, ev_norm = norm(text), norm(evidence)
    if not hit_norm or not ev_norm:
        return 0.0
    windows = [ev_norm[i : i + 12] for i in range(0, max(1, len(ev_norm) - 11), 6)]
    matched = sum(1 for w in windows if w in hit_norm)
    return matched / len(windows) if windows else 0.0


def ndcg(rel_flags: list[bool], k: int, ideal_count: int) -> float:
    dcg = sum(
        (1.0 / math.log2(i + 2))
        for i, rel in enumerate(rel_flags[:k])
        if rel
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(ideal_count, k)))
    return dcg / ideal if ideal else 0.0


def load_partial() -> dict:
    if OUT_JSON.exists():
        return json.loads(OUT_JSON.read_text(encoding="utf-8"))
    return {}


def save_partial(data: dict) -> None:
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_retrieval(api: str, doc_id: str, questions: list[dict]) -> dict:
    out = {"child": {}, "parent": {}}
    for index, q in enumerate(questions, 1):
        targets = {(t["pages"], t["breadcrumb"]) for t in q["targets"]}
        started = time.time()

        child = http_post_json(
            f"{api}/api/documents/{doc_id}/search",
            {"query": q["question"], "k": MRR_K, "merge_parents": False},
        )
        parent = http_post_json(
            f"{api}/api/documents/{doc_id}/search",
            {"query": q["question"], "k": MRR_K, "merge_parents": True},
        )

        child_flags, child_ranks = [], []
        child_cov_flags = []
        for hit in child["results"]:
            bc = strip_doc_prefix(hit.get("breadcrumb", ""), hit.get("doc_name", ""))
            child_flags.append((hit.get("pages"), bc) in targets)
            child_cov_flags.append(
                coverage(hit.get("text", ""), q.get("evidence", "")) >= COVERAGE_THRESHOLD
            )
        parent_flags, parent_loose = [], []
        for hit in parent["results"]:
            parent_flags.append(
                coverage(hit.get("text", ""), q.get("evidence", "")) >= COVERAGE_THRESHOLD
            )
            parent_loose.append(
                hit.get("section") == q["section"]
                and page_overlap(hit.get("pages"), q["page_start"], q["page_end"])
            )

        first_child = next((i + 1 for i, f in enumerate(child_flags) if f), None)
        first_parent = next((i + 1 for i, f in enumerate(parent_flags) if f), None)
        first_loose = next((i + 1 for i, f in enumerate(parent_loose) if f), None)
        first_child_cov = next((i + 1 for i, f in enumerate(child_cov_flags) if f), None)

        for key, flags, first in (
            ("child", child_flags, first_child),
            ("child_evidence", child_cov_flags, first_child_cov),
            ("parent", parent_flags, first_parent),
            ("parent_loose", parent_loose, first_loose),
        ):
            # 理想增益：target 精确判定按该题 targets 数；evidence 判定按
            # 本次检索到的相关块数（保证 nDCG ≤ 1）
            if key == "child":
                ideal = len(targets)
            else:
                ideal = max(1, sum(1 for f in flags if f))
            bucket = out.setdefault(key, {})
            for k in K_LIST:
                bucket.setdefault(f"hit@{k}", 0)
                bucket[f"hit@{k}"] += int(any(flags[:k]))
            bucket["mrr_sum"] = bucket.get("mrr_sum", 0.0) + (
                1.0 / first if first and first <= MRR_K else 0.0
            )
            bucket["ndcg_sum"] = bucket.get("ndcg_sum", 0.0) + ndcg(flags, MRR_K, ideal)

        # target 级 Recall（child 模式）：被 top-K 覆盖的 target 占比
        for k in (5, 10):
            matched = {
                t
                for hit in child["results"][:k]
                for t in targets
                if t == (hit.get("pages"), strip_doc_prefix(hit.get("breadcrumb", ""), hit.get("doc_name", "")))
            }
            out["child"][f"target_recall_sum@{k}"] = out["child"].get(
                f"target_recall_sum@{k}", 0.0
            ) + (len(matched) / len(targets) if targets else 0.0)

        out["child"].setdefault("per_question", {})[q["id"]] = {
            "first_target_rank": first_child,
        }
        out["parent"].setdefault("per_question", {})[q["id"]] = {
            "first_relevant_rank": first_parent,
            "first_loose_rank": first_loose,
        }
        print(
            f"[{index:>2}/{len(questions)}] {q['id']} "
            f"child_first={first_child} parent_first={first_parent} "
            f"({time.time() - started:.1f}s)"
        )
    return out


def run_chat(api: str, doc_id: str, questions: list[dict]) -> dict:
    out = {
        "citation_precision_sum": 0.0,
        "citation_blocks": 0,
        "evidence_in_context": 0,
        "evidence_in_answer": 0,
        "per_question": {},
    }
    for index, q in enumerate(questions, 1):
        started = time.time()
        chat = http_post_json(
            f"{api}/api/chat",
            {"question": q["question"], "k": CHUNK_K, "mode": "rag"},
            timeout=180,
        )
        # 引用块与父块检索 top-CHUNK_K 一一对应，取全文算证据覆盖
        parent = http_post_json(
            f"{api}/api/documents/{doc_id}/search",
            {"query": q["question"], "k": CHUNK_K, "merge_parents": True},
        )
        hit_by_key = {
            (h.get("pages"), strip_doc_prefix(h.get("breadcrumb", ""), h.get("doc_name", ""))): h
            for h in parent["results"]
        }
        covs = []
        for c in chat.get("citations", []):
            hit = hit_by_key.get((c.get("pages"), strip_doc_prefix(c.get("breadcrumb", ""), c.get("doc_name", ""))))
            text = hit.get("text", "") if hit else c.get("snippet", "")
            covs.append(coverage(text, q.get("evidence", "")) >= COVERAGE_THRESHOLD)
        precision = (sum(covs) / len(covs)) if covs else 0.0
        in_context = any(covs)
        in_answer = coverage(chat.get("answer", ""), q.get("evidence", "")) >= ANSWER_COVERAGE_THRESHOLD

        out["citation_precision_sum"] += precision
        out["citation_blocks"] += len(covs)
        out["evidence_in_context"] += int(in_context)
        out["evidence_in_answer"] += int(in_answer)
        out["per_question"][q["id"]] = {
            "citation_precision": round(precision, 3),
            "citations": len(covs),
            "evidence_in_context": in_context,
            "evidence_in_answer": in_answer,
            "answer_len": len(chat.get("answer", "")),
        }
        print(
            f"[{index:>2}/{len(questions)}] {q['id']} cite_prec={precision:.2f} "
            f"in_ctx={int(in_context)} in_ans={int(in_answer)} ({time.time() - started:.1f}s)"
        )
    return out


def render(data: dict, total: int) -> None:
    lines = ["# 检索/生成指标评测", ""]
    for mode, label in (
        ("child", "子块模式（merge_parents=false，target 精确判定）"),
        ("child_evidence", "子块模式·evidence 判定（与父块模式同口径，用于横比）"),
        ("parent", "父块归并模式（merge_parents=true，evidence 覆盖判定）"),
        ("parent_loose", "父块归并模式·宽松判定（section 一致且页码重叠）"),
    ):
        bucket = data.get(mode)
        if not bucket:
            continue
        lines += [f"## {label}", "", "| 指标 | 值 |", "|---|---|"]
        for k in K_LIST:
            hits = bucket.get(f"hit@{k}", 0)
            lines.append(f"| Hit@{k} | {hits}/{total}（{hits / total * 100:.1f}%） |")
        lines.append(f"| MRR@{MRR_K} | {bucket.get('mrr_sum', 0) / total:.3f} |")
        if mode == "child":
            # target 精确判定：理想相关块数 = 该题 distinct targets
            lines.append(f"| nDCG@{MRR_K} | {bucket.get('ndcg_sum', 0) / total:.3f} |")
        elif mode in ("child_evidence", "parent"):
            # evidence 判定：理想增益按检索到的相关块数，单题 nDCG 上限为 1
            lines.append(f"| nDCG@{MRR_K} | {bucket.get('ndcg_sum', 0) / total:.3f} |")
        for k in (5, 10):
            if f"target_recall_sum@{k}" in bucket:
                lines.append(
                    f"| Target-Recall@{k} | {bucket[f'target_recall_sum@{k}'] / total:.3f} |"
                )
        lines.append("")
    chat = data.get("chat")
    if chat:
        mean_precision = chat["citation_precision_sum"] / total
        lines += [
            "## 生成指标（/api/chat 生产路径）",
            "",
            f"- 引用准确率（每题均值：引用块中含答案证据的占比）：{mean_precision * 100:.1f}%"
            f"——服务端把 top-5 检索块全部列为引用，通常 1/5 为证据块、其余为上下文支撑块",
            f"- 引用召回（证据块被引用率）：{chat['evidence_in_context']}/{total}"
            f"（{chat['evidence_in_context'] / total * 100:.1f}%）",
            f"- Faithfulness 代理②证据入答案率：{chat['evidence_in_answer']}/{total}"
            f"（{chat['evidence_in_answer'] / total * 100:.1f}%）",
            "",
            "> 引用准确率口径说明：citations 为服务端从检索结果直接生成的引用列表（非模型自选），"
            "因此该指标衡量的是「top-5 检索块中证据块的占比」；证据是否被引用见引用召回。",
            "> 严格 Faithfulness 需 LLM-as-judge，未实现。",
        ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall/MRR/nDCG/引用/Faithfulness 评测")
    parser.add_argument("--phase", choices=["retrieval", "chat"], default="retrieval")
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--doc-id", default=None)
    parser.add_argument("--keyword", default="秦PLUS")
    args = parser.parse_args()

    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    questions = testset["questions"]
    doc_id = args.doc_id
    if not doc_id:
        items = http_get_json(f"{args.api}/api/documents")["items"]
        doc_id = next(i["id"] for i in items if args.keyword in i["filename"])

    data = load_partial()
    total = len(questions)
    if args.phase == "retrieval":
        data.update(run_retrieval(args.api, doc_id, questions))
        data["total"] = total
        save_partial(data)
        render(data, total)
    else:
        data["chat"] = run_chat(args.api, doc_id, questions)
        data["total"] = total
        save_partial(data)
        render(data, total)
    print(f"报告：{OUT_JSON}\n      {OUT_MD}")


if __name__ == "__main__":
    main()
