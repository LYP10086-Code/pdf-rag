# PDF-RAG 本地知识库

一套可在单机运行的 PDF 知识库问答系统：上传 PDF → 自研版面解析 → 结构化分块 → BGE-M3 向量化 → Qdrant 检索 → BGE 重排 → Qwen3 生成带页码引用的答案。

项目面向 8GB 显存的单卡环境设计：LLM 放在 GPU，Embedding、Reranker 和 PDF 解析放在 CPU，避免多个模型争抢显存。

## 功能

- PDF 上传与异步入库，状态机可观测：`pending → parsing → chunking → indexing → ready / failed`
- 自研 PDF 解析链路：样式签名标题探测、表格结构重建、伪影清洗、目录页剔除
- 结构化分块：标题 fragment 作为父块，500/50 子块作为检索粒度，表格按完整行切分
- 混合检索：BM25（CJK 二元组）+ Qdrant 向量，加权 RRF 融合，BGE Reranker 精排
- 父块归并：子块命中后按 `parent_key` 取回完整父块，兼顾检索精度与生成上下文
- RAG 问答：固定单轮链路，答案由服务端统一生成引用（文档名、章节、页码）
- Agent 问答页：检索 → 拼系统提示词 → Qwen3 单次生成，不做多步工具调用
- 前端页面：文档管理、检索测试、Agent 问答，React + Vite 构建后由 FastAPI 托管
- 独立测试：分块、页码标记、RRF、提示词、文件名清洗等纯函数测试

## 技术栈

| 模块 | 选型 |
|---|---|
| API | FastAPI + Uvicorn |
| 业务数据库 | PostgreSQL 16 + SQLAlchemy + psycopg |
| 向量库 | Qdrant |
| LLM | Qwen3-4B-AWQ，vLLM OpenAI 兼容接口 |
| Embedding | BAAI/bge-m3（CPU，1024 维） |
| Reranker | BAAI/bge-reranker-base（CPU） |
| PDF 解析 | 自研链路（pymupdf4llm 文本底座 + 物理信号标题/表格重建） |
| 分块 | LangChain Text Splitters |
| 前端 | React 18 + Vite 5 |

## 架构

```text
Windows 11
├── NVIDIA Windows 驱动
├── Docker Desktop（WSL2 后端）
│   ├── rag-postgres   PostgreSQL
│   ├── rag-qdrant     Qdrant
│   └── rag-vllm       Qwen3-4B-AWQ
└── WSL2 Ubuntu 24.04
    └── Python 3.12 venv
        ├── FastAPI
        ├── 自研 PDF 解析
        ├── BGE-M3 Embedding（CPU）
        ├── BGE Reranker（CPU）
        └── React/Vite 前端（构建产物由 FastAPI 托管）

入库：PDF → 解析 → 分块 → Embedding → Qdrant
问答：问题 → 混合检索 → RRF → 重排 → 父块归并 → 提示词 → Qwen3 → 答案 + 引用
```

## 目录结构

```text
pdf-rag/
├── app/
│   ├── main.py                 # FastAPI 入口、健康检查、前端静态挂载
│   ├── api/
│   │   ├── documents.py        # 上传、列表、详情、删除、重入库、检索
│   │   └── chat.py             # RAG / Agent 问答接口
│   ├── core/
│   │   ├── config.py           # .env 配置
│   │   ├── database.py         # SQLAlchemy 引擎与会话
│   │   ├── models.py           # documents / chunks / parent_chunks / ingest_jobs
│   │   ├── embeddings.py       # BGE-M3 单例
│   │   └── qdrant.py           # Qdrant 访问层
│   ├── ingestion/
│   │   ├── parser.py           # 自研解析入口
│   │   ├── custom_parse/       # 标题物理探测、表格重建、清洗
│   │   ├── chunker.py          # 父块/子块、页尾标记、表格切分
│   │   ├── cleaner.py          # Markdown 伪影清洗
│   │   └── indexer.py          # 入库状态机编排
│   ├── retrieval/
│   │   ├── retriever.py        # 检索 + 重排 + 父块归并
│   │   ├── hybrid.py           # BM25 + RRF
│   │   └── reranker.py         # BGE CrossEncoder
│   ├── rag/
│   │   ├── chain.py            # 提示词 + Qwen3 调用
│   │   ├── prompts.py          # RAG / Agent 提示词
│   │   └── agent.py            # 工具调用 Agent（可选模式）
│   ├── services/
│   │   ├── document_service.py # 上传、任务、列表、删除
│   │   └── chat_service.py     # 检索 → 生成编排
│   └── static/                 # 前端构建产物（npm run build 生成）
├── tests/                      # 纯函数测试
├── web/                        # React + Vite 前端源码
├── storage/                    # 原始 PDF、解析/分块产物、图片
├── docker-compose.yml          # PostgreSQL / Qdrant / vLLM
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

### 1. 前置条件

- Windows 11 + NVIDIA 驱动（支持 WSL CUDA）
- WSL2 + Ubuntu 24.04
- Docker Desktop（启用 WSL2 后端与 Ubuntu 集成）
- Python 3.12、Node.js 18+
- 显存 8GB 左右（示例配置为 RTX 4060 Laptop）

### 2. 启动基础服务

```bash
cd ~/projects/pdf-rag
cp .env.example .env
# 编辑 .env：修改 POSTGRES_PASSWORD，并同步 DATABASE_URL 中的密码
docker compose up -d postgres qdrant
docker compose ps
```

确认数据库与向量库就绪后，再启动 vLLM：

```bash
docker compose up -d vllm
docker compose logs -f vllm
curl http://localhost:8000/v1/models
```

首次启动需要下载模型。如果显存不足，按以下顺序调整：先把 `--max-model-len` 降到 2048，再把 `--gpu-memory-utilization` 降到 0.75，并确认没有其他程序占用显存。

### 3. 安装 Python 依赖

```bash
cd ~/projects/pdf-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# 仅 CPU 运行 Embedding/Reranker 时，建议先装 CPU 版 torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Ubuntu 如果提示缺少 `ensurepip`，先安装系统包：

```bash
sudo apt install -y python3.12-venv
```

### 4. 启动 API

```bash
cd ~/projects/pdf-rag
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

- 前端：http://localhost:8080/
- 接口文档：http://localhost:8080/docs
- 健康检查：http://localhost:8080/api/health

首次检索会加载 BGE-M3 与 Reranker；模型会缓存到 `~/.cache/huggingface`。

### 5. 构建前端

```bash
cd ~/projects/pdf-rag/web
npm install
npm run build
```

构建产物输出到 `app/static`，由 FastAPI 在 `/` 挂载。前端开发模式：

```bash
cd ~/projects/pdf-rag/web
npm run dev
```

Vite 会把 `/api` 代理到 `http://127.0.0.1:8080`。

## 配置

配置统一放在项目根 `.env`，示例见 `.env.example`。关键项：

| 变量 | 说明 | 默认 |
|---|---|---|
| `DATABASE_URL` | PostgreSQL 连接串，密码需与 `POSTGRES_PASSWORD` 一致 | `postgresql+psycopg://...` |
| `QDRANT_URL` | Qdrant 地址 | `http://localhost:6333` |
| `QDRANT_COLLECTION` | 集合名 | `pdf_knowledge` |
| `LLM_BASE_URL` | vLLM OpenAI 兼容地址 | `http://localhost:8000/v1` |
| `LLM_MODEL` | 模型名 | `Qwen/Qwen3-4B-AWQ` |
| `EMBEDDING_MODEL` | Embedding 模型 | `BAAI/bge-m3` |
| `EMBEDDING_DEVICE` | Embedding 运行设备 | `cpu` |
| `RERANKER_MODEL` | 重排模型 | `BAAI/bge-reranker-base` |
| `RERANKER_ENABLED` | 是否启用重排 | `true` |
| `PDF_PARSER` | 解析后端 | `custom` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 子块大小 / 重叠 | `500 / 50` |
| `RETRIEVAL_CANDIDATE_DEPTH` | 召回深度 | `20` |
| `RETRIEVAL_DEFAULT_K` | 默认返回条数 | `5` |
| `HYBRID_BM25_WEIGHT` | BM25 在 RRF 中的权重 | `0.2` |
| `CONTEXT_BUDGET_CHARS` | 生成上下文预算 | `2500` |
| `HF_ENDPOINT` | Hugging Face 镜像 | `https://hf-mirror.com` |

不要提交真实 `.env`，仓库只保留 `.env.example`。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 数据库、Qdrant、模型配置与存储状态 |
| POST | `/api/documents` | 上传 PDF，返回 `202 + document_id + job_id` |
| GET | `/api/documents` | 文档列表，支持 `status`、`q`、`page`、`size` |
| GET | `/api/documents/{id}` | 文档详情 |
| DELETE | `/api/documents/{id}` | 级联删除 Qdrant → PostgreSQL → 文件 |
| POST | `/api/documents/{id}/reingest` | 重新入库 |
| GET | `/api/jobs/{job_id}` | 入库进度与状态 |
| POST | `/api/search` | 全库 / 指定文档检索 |
| POST | `/api/documents/{id}/search` | 单文档检索 |
| POST | `/api/chat` | RAG 问答，`mode=rag` 为固定单轮链路 |

上传与问答示例：

```bash
curl -X POST -F "file=@manual.pdf" http://localhost:8080/api/documents

curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"首次保养在什么时候进行？","k":5,"mode":"rag"}'
```

问答响应包含 `answer`、`citations`（文档名、章节、页码、片段摘要）与 `retrieval` 元信息。

## 处理链路

### PDF 解析

`app/ingestion/custom_parse` 是项目自研解析链路：

1. `style_headings.py` 用 PDF 物理信号探测标题层级：字号、颜色、字形轮廓路径、单元格底纹
2. `table_parse.py` 用单元格底纹矩形与文字坐标重建表格网格，支持跨页续表
3. `clean_parse.py` 剔除目录页与前部材料，清理删除线/高亮/图标残留，并按坐标注入标题
4. `parser.py` 输出逐页 Markdown，页码与 PDF 页对齐

当前解析策略针对中文手册类文档标定，阈值不是通用 OCR 方案。下一阶段计划接入 MinerU 做对比实验。

### 分块

- 标题 fragment 作为父块（生成粒度），`parent_key` 关联子块
- 子块 500/50，按段落、换行、句号逐级切分
- 表格按完整行分组，每片重复表头
- 页尾标记 `@@E{page}@@` 让页码跟随本页内容，避免标题切分导致页码漂移
- 警告/注意/温馨提示额外抽取为 `kind=box` 的检索副本

### 检索与生成

1. BM25（CJK 二元组）与 Qdrant 向量各自召回
2. 加权 RRF 融合，BM25 权重 0.2，RRF k=60
3. BGE Reranker 对候选精排
4. 按 `parent_key` 归并父块，受 `CONTEXT_BUDGET_CHARS` 预算约束
5. 拼成带编号、文档名、章节、页码的资料块，交给 Qwen3
6. Qwen3 关闭思考模式（`enable_thinking=false`），直接输出答案

## 测试

```bash
cd ~/projects/pdf-rag
source .venv/bin/activate
pytest tests -q
```

覆盖分块页码标记、跨页合并、表格切分、RRF 融合、页文本定位、提示词约束和文件名清洗。

已完成的回归验证：

- 3 页测试 PDF：2 个子块、1 个父块、2 个 Qdrant 向量，问答返回正确引用
- 302 页真实手册：252 个有效页、124,499 字符，标题层级与警告框正常注入
- 三个问答样例单次耗时约 0.8～1.2 秒（检索 + 重排 + 生成）

## 资源与限制

- RTX 4060 8GB：建议 vLLM 显存利用率 0.75～0.80，上下文 4096，并发 1
- Embedding / Reranker 默认 CPU，避免与 LLM 争显存
- 本项目是单机 Demo，未实现多用户鉴权、集群、高可用与备份恢复
- 不要把未鉴权的 vLLM、PostgreSQL、Qdrant 端口暴露到公网
- 同名 PDF 会自动加时间后缀，不会覆盖已有文档
- 更换 Embedding 模型时需要重建 Qdrant 索引

## Roadmap

- 接入 MinerU 并对比解析质量、页码准确率和入库耗时
- PDF 解析阈值 profile 化，支持更多版式的车型手册
- 文档增量更新与幂等重建
- 检索与生成的固定评测集：Recall@K、MRR、nDCG、Faithfulness、引用准确率
- 任务队列化，解析不阻塞 API
