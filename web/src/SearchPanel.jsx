import React, { useState } from "react";
import { api } from "./api.js";

export default function SearchPanel({ docs, onError }) {
  const readyDocs = docs.filter((d) => d.status === "ready");
  const [docId, setDocId] = useState("__all__");
  const [query, setQuery] = useState("首次保养在什么时候进行");
  const [k, setK] = useState(5);
  const [mergeParents, setMergeParents] = useState(true);
  const [debug, setDebug] = useState(false);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);

  async function run(e) {
    e?.preventDefault?.();
    if (!query.trim()) return;
    setBusy(true);
    setRes(null);
    try {
      const payload = {
        query: query.trim(),
        k: Number(k),
        merge_parents: mergeParents,
      };
      if (docId !== "__all__") payload.document_ids = [docId];
      const data = await api.searchAll(payload, debug);
      setRes(data);
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2>检索测试</h2>
          <p className="muted">
            混合检索（BM25 + 向量 RRF）→ 可选父块归并 → BGE Reranker 精排；只检索不生成答案。
          </p>
        </div>
      </div>

      <form className="search-form" onSubmit={run}>
        <label className="field">
          <span>范围</span>
          <select value={docId} onChange={(e) => setDocId(e.target.value)}>
            <option value="__all__">全库（{readyDocs.length} 份文档）</option>
            {readyDocs.map((d) => (
              <option key={d.id} value={d.id}>
                {d.filename}（{d.chunk_count} 块）
              </option>
            ))}
          </select>
        </label>

        <label className="field grow">
          <span>问题</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="例如：首次保养在什么时候进行"
          />
        </label>

        <label className="field small">
          <span>条数 k</span>
          <input
            type="number"
            min="1"
            max="20"
            value={k}
            onChange={(e) => setK(e.target.value)}
          />
        </label>

        <label className="field">
          <span>返回粒度</span>
          <select
            value={mergeParents ? "parent" : "child"}
            onChange={(e) => setMergeParents(e.target.value === "parent")}
          >
            <option value="parent">父块（归并）</option>
            <option value="child">子块（原始命中）</option>
          </select>
        </label>

        <label className="field check">
          <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} />
          <span>debug 明细</span>
        </label>

        <button className="btn primary" disabled={busy}>
          {busy ? "检索中…" : "检索"}
        </button>
      </form>

      {res && (
        <div className="search-meta">
          <span>「{res.query}」</span>
          <span className="muted">
            耗时 {res.elapsed_ms} ms · 返回 {res.results.length} 个
            {res.merge_parents ? "父块" : "子块"}
          </span>
        </div>
      )}

      {res && res.results.length === 0 && <div className="empty">没有命中结果。</div>}

      {res &&
        res.results.map((r) => (
          <article key={r.rank} className="hit">
            <div className="hit-head">
              <span className="rank">#{r.rank}</span>
              <span className="doc">{r.doc_name}</span>
              <span className="pages">p{r.pages}</span>
              <span className="bc">{r.breadcrumb}</span>
              <span className="muted right">
                {res.merge_parents
                  ? `归并 ${r.matched_children ?? 1} 块 · score ${r.score}`
                  : `${r.kind} · #${r.chunk_index} · score ${r.score}`}
                {r.rerank_score !== null && r.rerank_score !== undefined
                  ? ` · rerank ${Number(r.rerank_score).toFixed(3)}`
                  : ""}
              </span>
            </div>
            <pre className="hit-text">{r.text}</pre>
            {debug && r.children && (
              <div className="children">
                {r.children.map((c) => (
                  <span key={c.chunk_index} className="child">
                    #{c.chunk_index} {c.kind} · rank {c.rank} · dense {c.dense_score} · bm25 {c.bm25_score}
                  </span>
                ))}
              </div>
            )}
          </article>
        ))}
    </section>
  );
}
