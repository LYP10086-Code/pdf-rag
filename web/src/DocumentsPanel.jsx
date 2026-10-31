import React, { useRef, useState } from "react";
import { api, STATUS_LABEL, fmtSize, fmtTime } from "./api.js";

export default function DocumentsPanel({ docs, onRefresh, onNotice, onError }) {
  const fileRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const [detail, setDetail] = useState(null);

  async function handlePick(e) {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      onError("仅支持 PDF 文件");
      return;
    }
    setBusy(true);
    try {
      const res = await api.upload(file);
      onNotice(`已提交入库：${res.filename}（任务 ${res.job_id}，状态会自动刷新）`);
      onRefresh();
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(d) {
    if (!window.confirm(`确认删除「${d.filename}」？原始文件、分块与向量都会删除。`)) return;
    try {
      await api.deleteDocument(d.id);
      onNotice(`已删除：${d.filename}`);
      onRefresh();
    } catch (e) {
      onError(e.message);
    }
  }

  async function handleReingest(d) {
    try {
      await api.reingest(d.id);
      onNotice(`已重新提交入库：${d.filename}`);
      onRefresh();
    } catch (e) {
      onError(e.message);
    }
  }

  async function showDetail(d) {
    try {
      setDetail(await api.getDocument(d.id));
    } catch (e) {
      onError(e.message);
    }
  }

  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <h2>文档</h2>
          <p className="muted">
            上传 PDF 后自动解析、分块并向量化；同名文件会自动加时间后缀，不会覆盖已有文档。
          </p>
        </div>
        <div className="actions">
          <input ref={fileRef} type="file" accept="application/pdf" hidden onChange={handlePick} />
          <button className="btn primary" disabled={busy} onClick={() => fileRef.current.click()}>
            {busy ? "上传中…" : "上传 PDF"}
          </button>
        </div>
      </div>

      {docs.length === 0 ? (
        <div className="empty">还没有文档，先上传一份 PDF。</div>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>名称</th>
              <th>状态</th>
              <th>页数</th>
              <th>块 / 父块</th>
              <th>大小</th>
              <th>创建时间</th>
              <th className="right">操作</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((d) => (
              <tr key={d.id}>
                <td>
                  <button className="link strong" onClick={() => showDetail(d)}>
                    {d.filename}
                  </button>
                  {d.error && <div className="err-text">{d.error}</div>}
                </td>
                <td>
                  <span className={`chip ${d.status}`}>{STATUS_LABEL[d.status] || d.status}</span>
                </td>
                <td>{d.page_count ?? "-"}</td>
                <td>
                  {d.chunk_count ?? 0} / {d.parent_count ?? 0}
                </td>
                <td>{fmtSize(d.size_mb)}</td>
                <td className="muted">{fmtTime(d.created_at)}</td>
                <td className="right">
                  <button className="link" onClick={() => handleReingest(d)}>重入库</button>
                  <button className="link danger" onClick={() => handleDelete(d)}>删除</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {detail && (
        <div className="modal-mask" onClick={() => setDetail(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>{detail.filename}</h3>
            <dl className="kv">
              <dt>ID</dt><dd className="mono">{detail.id}</dd>
              <dt>状态</dt><dd>{STATUS_LABEL[detail.status] || detail.status}</dd>
              <dt>页数</dt><dd>{detail.page_count}</dd>
              <dt>分块</dt><dd>{detail.chunk_count} 块 / {detail.parent_count} 父块</dd>
              <dt>大小</dt><dd>{fmtSize(detail.size_mb)}</dd>
              <dt>SHA256</dt><dd className="mono">{detail.sha256}</dd>
              <dt>存储路径</dt><dd className="mono">{detail.storage_path || "-"}</dd>
              <dt>索引版本</dt><dd>{detail.index_version || "-"}</dd>
              <dt>创建 / 更新</dt>
              <dd>{fmtTime(detail.created_at)} / {fmtTime(detail.updated_at)}</dd>
              {detail.error && (
                <>
                  <dt>错误</dt>
                  <dd className="err-text">{detail.error}</dd>
                </>
              )}
            </dl>
            <div className="modal-actions">
              <button className="btn" onClick={() => setDetail(null)}>关闭</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
