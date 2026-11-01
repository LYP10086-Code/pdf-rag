# -*- coding: utf-8 -*-
"""诊断：父块归并配置下，目标子块是否被返回的父块覆盖。

覆盖判定 L3cover：命中块的 pages 区间完全覆盖任一 target 的 pages，
且 target 的 breadcrumb 以命中块 breadcrumb 为前缀（即命中块是目标的祖先或自身）。
与 L3（与子块目标逐字相等）不同，L3cover 在父块粒度下才可比。
"""

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTSET_PATH = ROOT / "tests" / "testset.json"


def post(url: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def page_range(pages_str: str) -> tuple[int, int] | None:
    try:
        if "-" in pages_str:
            lo, hi = pages_str.split("-", 1)
            return int(lo), int(hi)
        return int(pages_str), int(pages_str)
    except (ValueError, AttributeError, TypeError):
        return None


def covers_pages(hit: dict, target: dict) -> bool:
    hr = page_range(hit.get("pages"))
    tr = page_range(target["pages"])
    return bool(hr and tr and hr[0] <= tr[0] and hr[1] >= tr[1])


def covers(hit: dict, target: dict, doc_name: str) -> bool:
    hr = page_range(hit.get("pages"))
    tr = page_range(target["pages"])
    if not hr or not tr:
        return False
    if not (hr[0] <= tr[0] and hr[1] >= tr[1]):
        return False
    bc = hit.get("breadcrumb", "")
    prefix = f"{doc_name} > "
    if bc.startswith(prefix):
        bc = bc[len(prefix):]
    return target["breadcrumb"].startswith(bc)


def main() -> None:
    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    questions = testset["questions"]
    doc_id = "d_44a8d134e8394cb3"

    for rerank in (False, True):
        pages_only = 0
        full_cover = 0
        for question in questions:
            response = post(
                f"http://localhost:8080/api/documents/{doc_id}/search",
                {
                    "query": question["question"],
                    "k": 5,
                    "merge_parents": True,
                    "use_reranker": rerank,
                },
            )
            hits = response["results"]
            doc_name = hits[0].get("doc_name", "") if hits else ""
            hit_pages = any(
                covers_pages(h, t) for h in hits for t in question["targets"]
            )
            hit_full = any(
                covers(h, t, doc_name) for h in hits for t in question["targets"]
            )
            pages_only += int(hit_pages)
            full_cover += int(hit_full)
        print(
            f"merge(use_reranker={rerank}): 页码覆盖 {pages_only}/{len(questions)}，"
            f"页码+breadcrumb 覆盖 {full_cover}/{len(questions)}"
        )


if __name__ == "__main__":
    main()
