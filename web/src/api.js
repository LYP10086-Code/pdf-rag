const BASE = "/api";

async function jsonFetch(url, options = {}) {
  const resp = await fetch(url, options);
  const text = await resp.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { raw: text };
  }
  if (!resp.ok) {
    const detail = data && data.detail !== undefined ? data.detail : data;
    const err = new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail)
    );
    err.status = resp.status;
    err.detail = detail;
    throw err;
  }
  return data;
}

export const api = {
  health: () => jsonFetch(`${BASE}/health`),

  listDocuments: (params = {}) => {
    const q = new URLSearchParams(params).toString();
    return jsonFetch(`${BASE}/documents${q ? "?" + q : ""}`);
  },
  getDocument: (id) => jsonFetch(`${BASE}/documents/${id}`),
  deleteDocument: (id) =>
    jsonFetch(`${BASE}/documents/${id}`, { method: "DELETE" }),
  reingest: (id) =>
    jsonFetch(`${BASE}/documents/${id}/reingest`, { method: "POST" }),
  getJob: (id) => jsonFetch(`${BASE}/jobs/${id}`),

  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return jsonFetch(`${BASE}/documents`, { method: "POST", body: form });
  },

  searchInDocument: (id, payload, debug) =>
    jsonFetch(`${BASE}/documents/${id}/search?debug=${debug ? 1 : 0}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  searchAll: (payload, debug) =>
    jsonFetch(`${BASE}/search?debug=${debug ? 1 : 0}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  chat: (payload) =>
    jsonFetch(`${BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};

export const STATUS_LABEL = {
  pending: "排队中",
  parsing: "解析中",
  chunking: "分块中",
  indexing: "向量化中",
  ready: "就绪",
  failed: "失败",
};

export function fmtSize(mb) {
  if (mb === undefined || mb === null) return "-";
  return mb < 1 ? `${Math.round(mb * 1024)} KB` : `${mb.toFixed(1)} MB`;
}

export function fmtTime(iso) {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("zh-CN", { hour12: false });
}
