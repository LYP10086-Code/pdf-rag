import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import DocumentsPanel from "./DocumentsPanel.jsx";
import SearchPanel from "./SearchPanel.jsx";
import AgentPanel from "./AgentPanel.jsx";

const ACTIVE = ["pending", "parsing", "chunking", "indexing"];

export default function App() {
  const [tab, setTab] = useState("docs");
  const [docs, setDocs] = useState([]);
  const [health, setHealth] = useState(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  async function refresh() {
    try {
      const [list, h] = await Promise.all([api.listDocuments(), api.health()]);
      setDocs(list.items || []);
      setHealth(h);
      setError("");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  // 有文档在入库时自动轮询
  useEffect(() => {
    const active = docs.some((d) => ACTIVE.includes(d.status));
    if (!active) return undefined;
    const t = setTimeout(refresh, 3000);
    return () => clearTimeout(t);
  }, [docs]);

  const dbOk = health?.database?.ok;
  const qdrantOk = health?.vector_store?.ok;
  const points = health?.vector_store?.points;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>PDF-RAG 知识库平台</h1>
          <span className="muted">自研解析 · BGE-M3 · Qdrant · Qwen3</span>
        </div>
        <div className="health">
          {health ? (
            <>
              <span className={`dot ${dbOk && qdrantOk ? "ok" : "bad"}`} />
              数据库 {dbOk ? "正常" : "异常"} · 向量 {points ?? "-"} 条 ·{" "}
              {health.models?.llm || "LLM 未配置"}
            </>
          ) : (
            <span className="muted">连接中…</span>
          )}
        </div>
      </header>

      <nav className="tabs">
        <button className={tab === "docs" ? "tab active" : "tab"} onClick={() => setTab("docs")}>
          文档管理
        </button>
        <button className={tab === "search" ? "tab active" : "tab"} onClick={() => setTab("search")}>
          检索测试
        </button>
        <button className={tab === "agent" ? "tab active" : "tab"} onClick={() => setTab("agent")}>
          Agent 问答
        </button>
      </nav>

      {(notice || error) && (
        <div className={error ? "banner error" : "banner ok"}>
          <span>{error || notice}</span>
          <button
            className="link"
            onClick={() => {
              setError("");
              setNotice("");
            }}
          >
            关闭
          </button>
        </div>
      )}

      {loading ? (
        <div className="empty">加载中…</div>
      ) : tab === "docs" ? (
        <DocumentsPanel docs={docs} onRefresh={refresh} onNotice={setNotice} onError={setError} />
      ) : tab === "search" ? (
        <SearchPanel docs={docs} onError={setError} />
      ) : (
        <AgentPanel docs={docs} onError={setError} onNotice={setNotice} />
      )}
    </div>
  );
}
