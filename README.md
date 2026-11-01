# PDF-RAG 本地知识库

一套可在单机运行的 PDF 知识库问答系统：上传 PDF → 自研版面解析 → 结构化分块 → BGE-M3 向量化 → Qdrant 检索 → 可选 BGE 重排 → Qwen3 生成带页码引用的答案。

项目面向 8GB 显存的单卡环境设计：LLM 放在 GPU，Embedding、Reranker 和 PDF 解析放在 CPU，避免多个模型争抢显存。

## 功能

- PDF 上传与异步入库，状态机可观测：`pending → parsing → chunking → indexing → ready / failed`
- 任务队列化：上传只写 Postgres 任务表，解析/分块/向量化由独立 worker 容器执行，不阻塞 API
- 自研 PDF 解析链路：样式签名标题探测、表格结构重建、伪影清洗、目录页剔除
- 结构化分块：标题 fragment 作为父块，500/50 子块作为检索粒度，表格按完整行切分
- 混合检索：BM25（CJK 二元组）+ Qdrant 向量，加权 RRF 融合，可选 BGE Reranker 精排（默认关闭，78 题消融净效果≈0）
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
| 任务队列 | PostgreSQL `FOR UPDATE SKIP LOCKED` + 独立 worker 容器 |
| 向量库 | Qdrant |
| LLM | Qwen3-4B-AWQ，vLLM OpenAI 兼容接口 |
| Embedding | BAAI/bge-m3（独立 embedder 容器，CPU，1024 维） |
| Reranker | BAAI/bge-reranker-base（独立 reranker 容器，CPU，默认关闭） |
| PDF 解析 | 自研链路（pymupdf4llm 文本底座 + 物理信号标题/表格重建），在 worker 容器执行 |
| 分块 | LangChain Text Splitters |
| 前端 | React 18 + Vite 5 |

## 架构

```text
Windows 11
├── NVIDIA Windows 驱动
├── Docker Desktop（WSL2 后端）
│   ├── rag-postgres   PostgreSQL
│   ├── rag-qdrant     Qdrant
│   ├── rag-vllm       Qwen3-4B-AWQ
│   ├── rag-embedder   BGE-M3 Embedding（CPU，OpenAI 兼容接口）
│   ├── rag-reranker   BGE Reranker（CPU，Cohere 兼容 /v1/rerank）
│   └── rag-worker     入库任务队列消费者（解析/分块/向量化）
└── WSL2 Ubuntu 24.04
    └── Python 3.12 venv
        ├── FastAPI
        ├── 自研 PDF 解析
        └── React/Vite 前端（构建产物由 FastAPI 托管）

入库：PDF → 解析 → 分块 → Embedding → Qdrant（worker 容器从 Postgres 队列认领执行）
问答：问题 → 混合检索 → RRF → 重排（可选，默认关）→ 父块归并 → 提示词 → Qwen3 → 答案 + 引用
```

## 目录结构

```text
pdf-rag/
├── app/
│   ├── main.py                 # FastAPI 入口、健康检查、前端静态挂载
│   ├── embedder_server.py      # 独立嵌入服务（OpenAI 兼容 /v1/embeddings）
│   ├── reranker_server.py      # 独立重排服务（Cohere 兼容 /v1/rerank）
│   ├── worker.py               # 入库队列消费者（SKIP LOCKED 认领、租约心跳）
│   ├── api/
│   │   ├── documents.py        # 上传、列表、详情、删除、重入库、检索
│   │   └── chat.py             # RAG / Agent 问答接口
│   ├── core/
│   │   ├── config.py           # .env 配置
│   │   ├── database.py         # SQLAlchemy 引擎与会话
│   │   ├── models.py           # documents / chunks / parent_chunks / ingest_jobs
│   │   ├── embeddings.py       # 嵌入 HTTP 客户端（调用 embedder 服务）
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
│   │   └── reranker.py         # 重排 HTTP 客户端（调用 reranker 服务）
│   ├── rag/
│   │   ├── chain.py            # 提示词 + Qwen3 调用
│   │   ├── prompts.py          # RAG / Agent 提示词
│   │   └── agent.py            # 工具调用 Agent（可选模式）
│   ├── services/
│   │   ├── document_service.py # 上传、任务、列表、删除
│   │   └── chat_service.py     # 检索 → 生成编排
│   └── static/                 # 前端构建产物（npm run build 生成）
├── tests/                      # 纯函数测试 + 78 题检索评测（eval_run / eval_compare）
├── web/                        # React + Vite 前端源码
├── storage/                    # 原始 PDF、解析/分块产物、图片
├── Dockerfile.api              # API 业务镜像：前端构建 + 运行时（无 torch）
├── Dockerfile.model            # 模型镜像：embedder / reranker（torch CPU）
├── .dockerignore
├── docker-compose.yml          # PostgreSQL / Qdrant / vLLM / embedder / reranker / worker / API
├── requirements.txt            # API 业务依赖
├── requirements-model.txt      # 模型服务依赖（torch / sentence-transformers）
├── .env.example
└── README.md
```

## 部署

支持两种部署路径，每种路径都提供两种镜像来源：

| | Windows + WSL2 | 原生 Linux 服务器 |
|---|---|---|
| Docker | Docker Desktop（WSL2 后端） | Docker Engine + NVIDIA Container Toolkit |
| 适用场景 | Windows 个人机跑全套（当前项目环境） | 独立服务器 / 云主机 |
| 命令执行位置 | WSL 终端 | 服务器终端 |

- **方式 A：从 GHCR 镜像仓库拉取** —— 不编译，最快跑通；前提是目标 commit 的镜像已由 CI 推送
- **方式 B：本地构建镜像** —— 不依赖 CI，改完代码即可构建；首次构建需下载依赖

本项目自身镜像有两个，CI 在 push `main` 后以同一 commit SHA 一起构建（保证嵌入/重排客户端与服务端版本一致）：

- `ghcr.io/lyp10086-code/pdf-rag-api`：API 业务镜像（无 torch），api 与 worker 共用
- `ghcr.io/lyp10086-code/pdf-rag-model`：模型镜像（embedder / reranker，torch CPU）

PostgreSQL、Qdrant、vLLM 使用官方镜像，`docker compose up -d` 时自动拉取。

### 1. 通用前置步骤

#### 准备机器

- Linux x86_64（示例 Ubuntu 24.04）或 Windows 11 + WSL2 Ubuntu 24.04
- 磁盘建议 40GB 以上（vLLM 镜像 + Qwen 权重较大）
- 问答功能需要 NVIDIA GPU + 驱动；没有 GPU 仍可用前端、入库和检索，见「没有 GPU 时」

#### 安装 Docker

Linux 服务器：

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker
```

`newgrp docker` 之后当前终端会进入 docker 组；如果仍提示权限不足，重新登录服务器再试。

WSL：在 Windows 上安装 Docker Desktop，并在 Settings → Resources → WSL Integration 勾选对应 Ubuntu 发行版（GPU 由 Docker Desktop 透传，Windows 侧装好 NVIDIA 驱动即可，无需在 WSL 内装 toolkit）。

两种环境都验证：

```bash
docker --version
docker compose version
```

GPU 服务器另需安装 NVIDIA Container Toolkit，并用 `nvidia-smi` 确认驱动可用。

#### 获取代码

在 WSL 或 Linux 终端执行（WSL 下不要把仓库放在 `/mnt/c/...`，IO 慢且权限易出问题）：

```bash
git clone https://github.com/LYP10086-Code/pdf-rag.git
cd pdf-rag
```

如果没有 git：`sudo apt update && sudo apt install -y git`；也可以从 GitHub 页面下载 ZIP 解压。

#### 配置 `.env`

```bash
cp .env.example .env
nano .env
```

至少修改数据库密码，并保持 `DATABASE_URL` 中的密码一致：

```env
POSTGRES_USER=raguser
POSTGRES_PASSWORD=your-strong-password
POSTGRES_DB=ragdb
DATABASE_URL=postgresql+psycopg://raguser:your-strong-password@localhost:5432/ragdb

API_PORT=8080
STORAGE_HOST_DIR=./storage
HF_CACHE_DIR=./data/huggingface
```

`RAG_API_IMAGE` / `RAG_MODEL_IMAGE` 决定镜像来源：方式 B 保持默认值（`pdf-rag-api:local`、`pdf-rag-model:local`），方式 A 改成 GHCR 地址（见下文）。

创建数据目录（容器内使用 UID 1000，目录需对其可写）：

```bash
mkdir -p storage data/huggingface
sudo chown -R 1000:1000 storage data/huggingface
```

> 项目内路径适合快速跑通和日常使用。长期运行的服务器建议把数据挪到独立目录（如 `/var/lib/pdf-rag/storage`、`/var/lib/pdf-rag/models`），避免清理仓库时误删数据：改 `.env` 里这两个变量、`mkdir -p` 并 `chown -R 1000:1000` 即可，其余步骤不变。不要把这两个目录放在 NFS 挂载或 WSL 的 `/mnt/c/...` 下。

### 2. 方式 A：从 GHCR 拉取镜像（WSL 与 Linux 命令相同）

先确认 CI 已为目标 commit 构建镜像：push `main` 后在 GitHub Actions 的 build-images 工作流查看（GHCR 包当前可匿名拉取，私有化后先 `docker login ghcr.io`，PAT 需带 `read:packages`）。

```bash
# latest 跟随 main 最新；生产建议固定 commit SHA，便于回滚
export RAG_API_IMAGE=ghcr.io/lyp10086-code/pdf-rag-api:latest
export RAG_MODEL_IMAGE=ghcr.io/lyp10086-code/pdf-rag-model:latest

docker compose pull api embedder reranker worker
docker compose up -d
```

两个变量也可以直接写进 `.env`（compose 中 shell 环境变量优先于 `.env`，临时验证用 export、长期固定写 `.env`）。

### 3. 方式 B：本地构建镜像（WSL 与 Linux 命令相同）

```bash
docker compose build      # 构建两个镜像：pdf-rag-api（业务）+ pdf-rag-model（模型）
docker compose up -d
```

- `Dockerfile.api` 构建业务镜像（前端构建 + Python 依赖，无 torch）
- `Dockerfile.model` 构建模型镜像（CPU 版 torch + sentence-transformers）
- 首次构建需访问 PyPI / npm / PyTorch 源，国内环境建议配置镜像加速
- 更新代码后重新 `build` + `up` 即可；Compose 对同 tag 新镜像不一定判定为配置变化，重建容器时加 `--force-recreate`

### 4. 启动后验证（两种方式通用）

```bash
docker compose ps
curl http://localhost:8080/api/health
```

- 等待 `rag-api`、`rag-embedder`、`rag-reranker`、`rag-worker` 四个容器显示 `healthy`（worker 靠心跳文件探活）；vLLM 首次启动要下载 Qwen 权重（约数 GB），比其他服务慢，`docker compose logs -f vllm` 看进度
- 嵌入 / 重排模型（bge-m3、bge-reranker-base）在首次入库/检索时自动下载到 `HF_CACHE_DIR`，`HF_ENDPOINT` 默认已指向 hf-mirror
- 打开前端：`http://localhost:8080/`（接口文档 `/docs`）
- 从其他机器访问：`http://服务器IP:8080/`，或先建 SSH 隧道 `ssh -L 8080:localhost:8080 user@server` 后访问 `http://localhost:8080/`；WSL2 会把 localhost 转发到 Windows，直接在 Windows 浏览器打开 `http://localhost:8080/`

### 5. 第一次使用

1. 打开前端「文档管理」，上传一个 PDF
2. 等待文档状态变成 `ready`
3. 打开「检索测试」或「Agent 问答」，输入问题
4. 首次检索会下载 BGE-M3 和 Reranker，需要网络

### 6. 没有 GPU 时

vLLM 需要 NVIDIA GPU。没有 GPU 时跳过 vLLM，启动其余服务（embedder / reranker 必须带上，api 依赖它们健康）：

```bash
docker compose up -d --build postgres qdrant embedder reranker worker api
```

此时前端、文档入库和检索可用；问答需要外部 LLM：

编辑 `docker-compose.yml`，把 `api.environment.LLM_BASE_URL` 改成外部 OpenAI 兼容服务地址。`.env` 里的 `LLM_BASE_URL` 会被 compose 覆盖，所以改 `.env` 不生效。

### 7. 停止、更新与重新启动

```bash
docker compose stop       # 停止全部容器
docker compose down       # 停止并删除容器，保留数据卷
docker compose down -v    # 连数据卷一起删除，慎用
```

更新代码后更新镜像与容器：

```bash
git pull

# 拉取方式
docker compose pull api embedder reranker worker
docker compose up -d --force-recreate api embedder reranker worker

# 构建方式
docker compose build
docker compose up -d --force-recreate
```

只更新 api 业务代码：

```bash
docker compose build api && docker compose up -d --no-deps --force-recreate api
```

### 8. 常见问题

- `permission denied`：重新登录服务器，或临时用 `sudo docker compose ...`
- 8080 被占用：修改 `.env` 的 `API_PORT`
- `rag-api` 一直不 healthy：看 `docker compose logs api`
- vLLM 启动失败：检查 `nvidia-smi` 和 NVIDIA Container Toolkit
- 模型下载慢：设置 `HF_ENDPOINT`，或提前拷贝模型缓存
- 端口不要直接暴露公网：API 目前没有鉴权，建议用 SSH 隧道或反向代理

## 快速开始

> 以下是不使用 Docker 的本地开发流程；日常启动请优先使用上面的「部署」。

### 1. 前置条件

- Windows 11 + NVIDIA 驱动（支持 WSL CUDA）
- WSL2 + Ubuntu 24.04
- Docker Desktop（启用 WSL2 后端与 Ubuntu 集成）
- Python 3.12、Node.js 18+
- 显存 8GB 左右（示例配置为 RTX 4060 Laptop）

### 2. 启动基础服务

```bash
cd ~/projects/pdf-rag
# 仅首次运行：已有 .env 时跳过，避免覆盖密码
[ -f .env ] || cp .env.example .env
# 编辑 .env：修改 POSTGRES_PASSWORD，并同步 DATABASE_URL 中的密码
docker compose up -d postgres qdrant embedder reranker worker
docker compose ps
```

embedder / reranker 容器发布在宿主机 8001 / 8002 端口，供本地直跑的 API 调用；worker 容器消费入库队列（本地 API 上传后任务由它执行）。

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

# API 业务依赖（API 不含模型代码，不需要 torch）
pip install -r requirements.txt

# 仅本地直跑 embedder / reranker 服务时才需要模型依赖：
# pip install torch --index-url https://download.pytorch.org/whl/cpu
# pip install -r requirements-model.txt
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

本地 API 通过 `EMBEDDING_API_URL` / `RERANKER_API_URL`（默认 `localhost:8001` / `localhost:8002`）调用 embedder / reranker 容器。调试模型服务本身时也可以本地直跑（需 `requirements-model.txt`）：

```bash
uvicorn app.embedder_server:app --host 0.0.0.0 --port 8001
uvicorn app.reranker_server:app --host 0.0.0.0 --port 8002
```

此方式下模型缓存到 `~/.cache/huggingface`。

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

## WSL 部署与运行方式

当前实际部署环境是 Windows 11 + WSL2 Ubuntu-24.04，全部服务以 Docker 容器运行：

- Windows 源码目录：`c:\Users\l1y2p\Desktop\pdf-rag`
- WSL 运行目录：`/home/lyp/projects/pdf-rag`
- `rag-api` / `rag-worker` 用 `Dockerfile.api` 构建（业务镜像，worker 换启动命令）、`rag-embedder` / `rag-reranker` 用 `Dockerfile.model` 构建（模型镜像）；PostgreSQL、Qdrant、vLLM 使用官方镜像
- Windows 仓库通过 `wsl` 远程推送到 WSL 内的裸仓库，再由钩子同步代码、拉取（或回退本地构建）镜像并重启容器

### 启动顺序

1. 启动 Docker Desktop，确认已开启 WSL 集成（Ubuntu-24.04）。
2. 在 WSL 项目目录启动全部服务：

```bash
cd ~/projects/pdf-rag
docker compose up -d
docker compose ps
```

只启动或更新 API：

```bash
docker compose up -d --no-deps api
```

3. 检查服务：

```bash
curl http://localhost:8080/api/health
```

Windows 侧可直接访问 `http://localhost:8080/`，WSL2 会把 localhost 转发到 Linux。

### 容器配置

`.env` 中与容器相关的变量：

| 变量 | 说明 | 当前 WSL 值 |
|---|---|---|
| `RAG_API_IMAGE` | API 业务镜像名（无 torch） | `pdf-rag-api:local` |
| `RAG_MODEL_IMAGE` | 模型镜像名（embedder/reranker 共用） | `pdf-rag-model:local` |
| `API_PORT` | 宿主机端口 | `8080` |
| `EMBEDDER_PORT` | embedder 服务宿主机端口 | `8001` |
| `RERANKER_PORT` | reranker 服务宿主机端口 | `8002` |
| `STORAGE_HOST_DIR` | 宿主机 storage 目录 | `/home/lyp/projects/pdf-rag/storage` |
| `HF_CACHE_DIR` | Embedding/Reranker 模型缓存 | `/home/lyp/.cache/huggingface` |
| `VLLM_CACHE_DIR` | vLLM 模型缓存 | `./data/vllm` |

容器内通过 compose 服务名访问基础服务：`postgres:5432`、`qdrant:6333`、`vllm:8000`、`embedder:8001`、`reranker:8002`。storage 挂载为 `/data/storage`，模型缓存挂载为 `/models`，模型权重不打进镜像；PDF 与模型缓存均落在宿主机目录，不使用 Docker 命名卷。

镜像构建与更新的完整命令见上文「部署」章节的方式 A / 方式 B；日常更新也可直接 `git push wsl main`，由部署钩子自动完成（见下）。

### 从 Windows 推送更新

Windows 仓库已配置指向 WSL 裸仓库的 `wsl` 远程：

```powershell
git push wsl main
```

首次配置时可执行：

```powershell
git remote add wsl '\\wsl.localhost\Ubuntu-24.04\home\lyp\projects\pdf-rag.git'
```

推送钩子（`~/projects/pdf-rag.git/hooks/post-receive`）会调用 WSL 内的部署脚本，依次完成：

1. 将提交内容同步到 `/home/lyp/projects/pdf-rag`，保留 `.env`、`.venv`、`storage`、`node_modules`
2. 优先拉取与提交 SHA 对应的 GHCR 镜像 `ghcr.io/lyp10086-code/pdf-rag-api:<commit-sha>` 与 `ghcr.io/lyp10086-code/pdf-rag-model:<commit-sha>`
3. 镜像尚未生成或拉取失败时，回退到 `docker compose build` 本地构建
4. 执行 `docker compose up -d --force-recreate` 重建 api / embedder / reranker 容器
5. 请求 `/api/health` 做健康检查

部署脚本位于 `~/projects/pdf-rag.git/hooks/deploy-in-wsl.sh`，只存在于 WSL，不在 Git 仓库内。钩子会先检查提交里是否包含 `Dockerfile.api`、`Dockerfile.model` 和带 `api` 服务的 compose 文件，缺失时直接失败并保留当前工作区，避免把容器配置回退掉。只推送已提交的内容，Windows 侧未提交的改动不会同步。

### GitHub Actions 发布镜像

`.github/workflows/build-image.yml` 在 push `main` 时自动构建并推送镜像：

- 镜像：`ghcr.io/lyp10086-code/pdf-rag-api` 与 `ghcr.io/lyp10086-code/pdf-rag-model`
- tag：同一 commit 的完整 SHA 和 `latest`（两个镜像同 commit 构建，保证嵌入/重排客户端与服务端版本一致）
- 使用 Buildx 的 GHA 缓存

GHCR 可选部署方式：

```bash
docker login ghcr.io
export RAG_API_IMAGE=ghcr.io/lyp10086-code/pdf-rag-api:<commit-sha>
export RAG_MODEL_IMAGE=ghcr.io/lyp10086-code/pdf-rag-model:<commit-sha>
docker compose pull api embedder reranker worker
docker compose up -d --no-deps --force-recreate api embedder reranker worker
```

当前 GHCR 包可匿名拉取，不需要 `docker login`；如果以后改为私有，PAT 需要带 `read:packages`。WSL 钩子优先拉取 GHCR 镜像，拉取失败时回退本地构建。

### 查看与停止

```bash
docker compose ps
docker compose logs -f api
tail -f ~/projects/pdf-rag/logs/deploy.log
docker compose restart api
docker compose stop api
```

### 注意

- 所有容器都设置了 `restart: unless-stopped`，Docker Desktop 启动后会自动恢复；API 不再需要手动运行 uvicorn。
- 全量启动时如果 vLLM 因 `--enable-auto-tool-choice` / `--tool-call-parser` 启动失败，按 `docker-compose.yml` 里的注释删掉这两行再启动；RAG 模式不需要。
- 开发调试仍可使用 `.venv` + uvicorn，但需要先 `docker compose stop api`，或换一个端口，避免和容器争用 8080。
- 端口：API `8080`，vLLM `8000`，embedder `8001`，reranker `8002`，PostgreSQL `5432`，Qdrant `6333/6334`。

## 配置

配置统一放在项目根 `.env`，示例见 `.env.example`。关键项：

| 变量 | 说明 | 默认 |
|---|---|---|
| `DATABASE_URL` | PostgreSQL 连接串，密码需与 `POSTGRES_PASSWORD` 一致 | `postgresql+psycopg://...` |
| `QDRANT_URL` | Qdrant 地址 | `http://localhost:6333` |
| `QDRANT_COLLECTION` | 集合名 | `pdf_knowledge` |
| `LLM_BASE_URL` | vLLM OpenAI 兼容地址 | `http://localhost:8000/v1` |
| `LLM_MODEL` | 模型名 | `Qwen/Qwen3-4B-AWQ` |
| `EMBEDDING_MODEL` | Embedding 模型（embedder 服务） | `BAAI/bge-m3` |
| `EMBEDDING_API_URL` | embedder 服务地址（容器内由 compose 覆盖） | `http://localhost:8001/v1` |
| `EMBEDDING_API_KEY` | 嵌入服务 key（本地留空，云端必填） | 空 |
| `EMBEDDING_DEVICE` | Embedding 运行设备 | `cpu` |
| `RERANKER_MODEL` | 重排模型（reranker 服务） | `BAAI/bge-reranker-base` |
| `RERANKER_API_URL` | reranker 服务地址（容器内由 compose 覆盖） | `http://localhost:8002/v1` |
| `RERANKER_API_KEY` | reranker 服务 key（本地留空，云端必填） | 空 |
| `RERANKER_ENABLED` | 是否启用重排（默认关，检索页可勾选；78 题消融净效果≈0） | `false` |
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
3. （可选）BGE Reranker 对候选精排——默认关闭，检索测试页可勾选；78 题消融净效果≈0
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

检索评测脚本（需先启动 api）：

```bash
python tests/eval_run.py        # 78 题基线（L1 章节命中 / L2 页码命中 / L3 目标块精确命中）
python tests/eval_compare.py    # rerank × 父块归并 消融对比
```

报告写入 `tests/eval_report.md` 与 `tests/eval_compare_report.md`。

已完成的回归验证：

- 3 页测试 PDF：2 个子块、1 个父块、2 个 Qdrant 向量，问答返回正确引用
- 302 页真实手册：252 个有效页、124,499 字符，标题层级与警告框正常注入
- 78 题消融：直接检索 L3 96.2%，加 rerank 97.4%（净 +1 题、+3.6s/查询），父块归并下目标页码覆盖 100% —— rerank 默认关闭
- 问答端到端耗时：检索 ~0.1s（rerank 关）+ Qwen3 生成；开启 rerank 时检索段 +3.6s

## 资源与限制

- RTX 4060 8GB：建议 vLLM 显存利用率 0.75～0.80，上下文 4096，并发 1
- Embedding / Reranker 均为独立容器（CPU），api 容器只保留业务逻辑，避免与 LLM 争显存
- PDF 解析在独立 worker 容器（4g/4CPU）执行，api 只保留在线请求（2g/2CPU）
- 本项目是单机 Demo，未实现多用户鉴权、集群、高可用与备份恢复
- 不要把未鉴权的 vLLM、PostgreSQL、Qdrant 端口暴露到公网
- 同名 PDF 会自动加时间后缀，不会覆盖已有文档
- 更换 Embedding 模型时需要重建 Qdrant 索引

## Roadmap

- 接入 MinerU 并对比解析质量、页码准确率和入库耗时
- PDF 解析阈值 profile 化，支持更多版式的车型手册
- 文档增量更新与幂等重建
- 检索与生成的固定评测集：Recall@K、MRR、nDCG、Faithfulness、引用准确率
- ✅ 任务队列化，解析不阻塞 API（已落地：Postgres `SKIP LOCKED` 队列 + 独立 worker 容器，支持租约重排与多 worker 水平扩展）
