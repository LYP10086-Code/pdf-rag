import React, { useState } from "react";
import { api } from "./api.js";

export default function AgentPanel({ docs, onError, onNotice }) {
  const readyDocs = docs.filter((d) => d.status === "ready");
  const [docId, setDocId] = useState("__all__");
  const [question, setQuestion] = useState("首次保养在什么时候进行？");
  const [k, setK] = useState(5);
  const [mergeParents, setMergeParents] = useState(true);
  const [debug, setDebug] = useState(false);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);

  async function run(e) {
    e?.preventDefault?.();
    if (!question.trim()) return;
    setBusy(true);
    setRes(null);
    try {
      const payload = {
        question: question.trim(),
        k: Number(k),
        mode: "rag",
        merge_parents: mergeParents,
        debug,
      };
      if (docId !== "__all__") payload.document_ids = [docId];
      const data = await api.chat(payload);
      setRes(data);
      if (data.answer === "知识库中未找到相关内容。") {
        onNotice("检索没有返回可用依据，模型按拒答策略返回。");
      }
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (readyDocs.length === 0) {
    return (
      <section className="panel">
        <div className="panel-head">
          <div>
            <h2>Agent 问答测试</h2>
            <p className="muted">固定单轮链路：检索 → 父块 → 系统提示词 → Qwen3 生成。</p>
          </div>
        </div>
        <div className="empty">还没有就绪的文档，请先在「文档管理」中上传并等待入库完成。</div>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2>Agent 问答测试</h2>
          <p className="muted">
            固定单轮 RAG 链：先混合检索 + 重排，取 Top-K 父块拼进系统提示词，再由 Qwen3 生成答案；
            不启用多步工具调用。
          </p>
        </div>
      </div>

      <form className="search-form" onSubmit={run}>
        <label className="field">
          <span>文档范围</span>
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
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="例如：首次保养在什么时候进行？"
          />
        </label>

        <label className="field small">
          <span>上下文 k</span>
          <input
            type="number"
            min="1"
            max="20"
            value={k}
            onChange={(e) => setK(e.target.value)}
          />
        </label>

        <label className="field">
          <span>上下文粒度</span>
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
          <span>显示检索明细</span>
        </label>

        <button className="btn primary" disabled={busy}>
          {busy ? "生成中…" : "提问"}
        </button>
      </form>

      {res && (
        <>
          <div className="agent-meta">
            <span>模式：{res.mode === "rag" ? "固定 RAG 链" : res.mode}</span>
            <span className="muted">
              检索 {res.retrieval?.elapsed_ms ?? "-"} ms · 总耗时 {res.elapsed_ms} ms
            </span>
          </div>

          <div className="answer">
            <div className="answer-title">模型回复</div>
            <div className="answer-text">{res.answer}</div>
          </div>

          <div className="answer-section-title">
            检索依据（{res.citations?.length ?? 0} 条）
          </div>
          {res.citations && res.citations.length > 0 ? (
            <div className="citations">
              {res.citations.map((c) => (
                <article key={c.id} className="citation">
                  <div className="hit-head">
                    <span className="rank">片段{c.id}</span>
                    <span className="doc">{c.doc_name}</span>
                    <span className="pages">p{c.pages}</span>
                    <span className="bc">{c.breadcrumb}</span>
                  </div>
                  <div className="citation-text">{c.snippet}</div>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty">没有检索到依据。</div>
          )}

          {debug && res.retrieval && (
            <>
              <div className="answer-section-title">检索参数 / 明细</div>
              <pre className="debug-json">{JSON.stringify(res.retrieval, null, 2)}</pre>
            </>
          )}
        </>
      )}
    </section>
  );
}
