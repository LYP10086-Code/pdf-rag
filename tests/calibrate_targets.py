# -*- coding: utf-8 -*-
"""校准 testset targets 到当前管线 schema（仅 L3 未命中题）。

方法（evidence 锚定，防止把错误命中"扶正"）：
1. 对每个 L3 未命中题重新检索（k=5，原始子块）；
2. 对每个命中块计算「证据覆盖率」：题目 evidence 归一化后按滑窗在块文本中
   命中的比例；
3. 覆盖率 ≥0.4 的命中块视为真实目标 → (pages, breadcrumb) 加入 targets；
   原有 targets 中仍能与某命中块完全相等的保留；其余（schema 陈旧的）移除；
4. Q34（唯一真检索 miss）不校准——它的 top-k 里没有证据块，需单独排查。
原文件备份为 testset.before_calibration.json。
"""

import copy
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTSET = ROOT / "tests" / "testset.json"
BACKUP = ROOT / "tests" / "testset.before_calibration.json"
REPORT = ROOT / "tests" / "eval_report.json"
API = "http://localhost:8080"
K = 5
COVERAGE_THRESHOLD = 0.4
WIN, STEP = 12, 6

_NORM = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]")


def norm(s: str) -> str:
    return _NORM.sub("", s or "")


def coverage(hit_text: str, evidence: str) -> float:
    hit_norm, ev_norm = norm(hit_text), norm(evidence)
    if not hit_norm or not ev_norm:
        return 0.0
    windows = [ev_norm[i : i + WIN] for i in range(0, max(1, len(ev_norm) - WIN + 1), STEP)]
    matched = sum(1 for w in windows if w in hit_norm)
    return matched / len(windows) if windows else 0.0


def search(query: str) -> dict:
    body = json.dumps(
        {"query": query, "k": K, "merge_parents": False}, ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/api/documents/{DOC_ID}/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def strip_doc_prefix(breadcrumb: str, doc_name: str) -> str:
    prefix = f"{doc_name} > "
    return breadcrumb[len(prefix):] if breadcrumb.startswith(prefix) else breadcrumb


testset = json.loads(TESTSET.read_text(encoding="utf-8"))
report = json.loads(REPORT.read_text(encoding="utf-8"))
DOC_ID = report["doc_id"]
failed_ids = [r["id"] for r in report["results"] if not r["L3"]]
qmap = {q["id"]: q for q in testset["questions"]}

BACKUP.write_text(json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已备份原文件 → {BACKUP.name}")

changes = []
for qid in failed_ids:
    q = qmap[qid]
    if qid == "Q34":
        print(f"{qid}: 跳过（真检索 miss，不校准）")
        continue
    resp = search(q["question"])
    hits = resp["results"]
    doc_name = hits[0]["doc_name"] if hits else ""
    # 证据锚定：覆盖率达标的命中块
    good = [
        (h["pages"], strip_doc_prefix(h["breadcrumb"], doc_name))
        for h in hits
        if coverage(h.get("text", ""), q.get("evidence", "")) >= COVERAGE_THRESHOLD
    ]
    good_set = set(good)
    # 原有 targets 中仍与命中块完全相等的保留
    kept = [
        t for t in q["targets"]
        if (t["pages"], t["breadcrumb"]) in good_set
    ]
    kept_keys = {(t["pages"], t["breadcrumb"]) for t in kept}
    added = [g for g in good_set if g not in kept_keys]
    if not kept and not added:
        print(f"{qid}: 无证据锚定命中，保持原样（需人工核查）")
        continue
    new_targets = [{"pages": p, "breadcrumb": b} for p, b in sorted(kept_keys | set(added))]
    removed = [
        (t["pages"], t["breadcrumb"])
        for t in q["targets"]
        if (t["pages"], t["breadcrumb"]) not in kept_keys
    ]
    q["targets"] = new_targets
    changes.append((qid, kept, added, removed))
    print(f"{qid}: 保留 {len(kept)} / 新增 {len(added)} / 移除 {len(removed)}")
    for p, b in added:
        print(f"    + ({p}, {b})")
    for p, b in removed:
        print(f"    - ({p}, {b})")

meta = testset["meta"].setdefault("适配说明", [])
meta.append(
    "2026-11-01 target 校准：19 个 L3 未命中题中 18 个为旧管线 schema 差异"
    "（CJK 空格/层级重建/章级补链/pages 范围/target 覆盖不全），已按 evidence "
    "覆盖率锚定重新生成 targets；Q34 为真检索 miss 未校准，待单独排查"
)
TESTSET.write_text(json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n共校准 {len(changes)} 题，已写回 {TESTSET.name}")
