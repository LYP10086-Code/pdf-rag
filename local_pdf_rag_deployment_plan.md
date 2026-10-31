# 本地化 PDF-RAG 知识库项目部署计划

> **硬件环境**：NVIDIA RTX 4060 8GB、32GB 系统内存、约 1TB 可用硬盘空间\
> **技术栈**：WSL2 + Ubuntu 24.04 + Docker Compose + vLLM + LangChain +
> FastAPI + PostgreSQL + Qdrant\
> **项目目标**：搭建一套可在单机运行、具备工程化结构的 PDF-RAG 知识库
> Demo。

------------------------------------------------------------------------

## 1. 项目目标与总体方案

本项目实现从 PDF
上传、解析、文本切分、向量化、向量检索、重排序到大模型生成答案的完整流程，并返回答案引用来源、文档名称和页码。

考虑到 RTX 4060 只有 8GB 显存，部署策略是：

-   GPU 优先用于 LLM 推理。
-   Embedding 和 Reranker 初期放在 CPU 上运行，避免多个模型争抢显存。
-   PostgreSQL、Qdrant 和 vLLM 使用 Docker 部署。
-   FastAPI、LangChain 先在 WSL2 的 Python
    虚拟环境中运行，便于开发调试。
-   PDF 解析先采用 CPU 优先的方式，后续根据实际速度和资源占用优化。

### 1.1 总体架构

``` text
Windows 11
├── NVIDIA Windows 驱动
├── Docker Desktop（启用 WSL2 后端）
└── WSL2
    └── Ubuntu 24.04
        ├── Python 虚拟环境
        │   ├── FastAPI
        │   ├── LangChain
        │   ├── BGE-M3 Embedding（CPU）
        │   ├── BGE Reranker（CPU）
        │   └── MinerU（CPU 优先）
        │
        └── Docker Compose
            ├── vLLM：Qwen3-4B-AWQ（GPU）
            ├── PostgreSQL：业务数据
            └── Qdrant：向量数据
```

### 1.2 核心数据流

**文档入库流程**

``` text
用户上传 PDF
    |
    v
FastAPI 接收文件并保存原始 PDF
    |
    v
PostgreSQL 创建文档记录
    |
    v
后台任务调用 MinerU 解析 PDF
    |
    v
生成 Markdown、表格内容和图片资源
    |
    v
文本清洗、结构化和 Chunk 切分
    |
    v
BGE-M3 生成 Embedding
    |
    v
写入 Qdrant
    |
    v
PostgreSQL 更新文档状态和统计信息
```

**问答流程**

``` text
用户问题
    |
    v
FastAPI
    |
    v
问题 Embedding
    |
    v
Qdrant 向量召回 Top-K
    |
    v
Reranker 重排序
    |
    v
选取相关 Chunk 并组织 Prompt
    |
    v
vLLM / Qwen3-4B-AWQ
    |
    v
生成答案、引用文档和页码
```

------------------------------------------------------------------------

## 2. 模型选型

  模块        推荐模型 / 组件            初始运行位置   用途
  ----------- -------------------------- -------------- ---------------------------------
  LLM         `Qwen/Qwen3-4B-AWQ`        vLLM + GPU     RAG 答案生成、工具调用实验
  Embedding   `BAAI/bge-m3`              CPU            中文、多语言文本向量化
  Reranker    `BAAI/bge-reranker-base`   CPU            对召回结果进行相关性重排
  PDF 解析    MinerU                     CPU 优先       PDF 转 Markdown、提取表格和图片

### 2.1 LLM：Qwen3-4B-AWQ

-   模型仓库：`Qwen/Qwen3-4B-AWQ`
-   选择原因：
    -   4B 级别模型更适合 8GB 显存的单卡环境。
    -   AWQ 4-bit 量化降低权重显存占用。
    -   中文和多语言能力适合中文知识库问答。
    -   可通过 vLLM 提供 OpenAI 兼容 API，方便 LangChain 接入。
-   初始推理参数建议：
    -   上下文长度：4096 tokens。
    -   最大并发序列数：1。
    -   GPU 显存利用率：先从 0.75～0.80 试起。
    -   生成长度：按任务设置，例如 512～1024 tokens。

模型权重文件大小不等于实际推理显存占用。实际还需要为 KV
Cache、运行时激活和 CUDA 开销留出空间。首次启动如遇到
OOM，优先降低上下文长度和并发数，再调整显存利用率。

模型页面：https://huggingface.co/Qwen/Qwen3-4B-AWQ

### 2.2 Embedding：BGE-M3

-   模型仓库：`BAAI/bge-m3`
-   向量维度：1024。
-   支持中文及多语言。
-   支持 Dense、Sparse 和 Multi-Vector 等检索方式。
-   适合后续扩展 Hybrid Search、BM25 / Sparse Retrieval 和 RRF。

初期放在 CPU 上运行，批量大小从 4 或 8 开始测试，再根据内存和速度调整。

模型页面：https://huggingface.co/BAAI/bge-m3

### 2.3 Reranker：BGE Reranker Base

-   模型仓库：`BAAI/bge-reranker-base`
-   初期放在 CPU 上运行。
-   典型流程：向量召回 Top 20，再由 Reranker 重排，最后取 Top 5 作为 LLM
    上下文。
-   后续可评估 `BAAI/bge-reranker-v2-m3`，但应先比较 CPU
    推理耗时和检索质量。

模型页面：https://huggingface.co/BAAI/bge-reranker-base

### 2.4 模型部署原则

**初期不要让 LLM、Embedding、Reranker 和 PDF 解析同时使用 GPU。**

建议先固定为：

-   LLM：GPU。
-   Embedding：CPU。
-   Reranker：CPU。
-   MinerU：CPU 优先。

如果 CPU Embedding 速度不足，可以在文档入库阶段暂时停止 vLLM，再让
Embedding 使用 GPU；或者通过任务队列安排 GPU 任务，避免资源争抢。

------------------------------------------------------------------------

## 3. WSL2 环境搭建

### 3.1 安装 WSL2 和 Ubuntu 24.04

在 Windows PowerShell（管理员）中执行：

``` powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl --set-default-version 2
```

查看发行版状态：

``` powershell
wsl -l -v
```

确保 Ubuntu 的 VERSION 为 `2`。首次启动 Ubuntu 时，按提示创建 Linux
用户。

### 3.2 配置 WSL2 资源

在 Windows 用户目录下创建或编辑 `.wslconfig`：

``` ini
[wsl2]
memory=20GB
processors=8
swap=8GB
```

说明：

-   `memory=20GB`：给 WSL2 设置内存上限。32GB 物理内存的电脑可先从 20GB
    开始。
-   `processors=8`：示例值，应根据实际 CPU
    核心数调整，不要超过机器实际可用核心数。
-   `swap=8GB`：设置交换空间，避免偶发内存峰值直接导致进程退出。

修改后，在 PowerShell 中执行：

``` powershell
wsl --shutdown
```

然后重新打开 Ubuntu。

### 3.3 配置 NVIDIA CUDA for WSL

1.  在 Windows 安装支持 WSL CUDA 的 NVIDIA 显卡驱动。
2.  进入 Ubuntu，执行：

``` bash
nvidia-smi
```

如果能看到 RTX 4060 和显存信息，说明 WSL 能访问 GPU。

**注意：不要在 WSL2 中安装独立的 Linux NVIDIA 内核显卡驱动。** WSL 使用
Windows 侧驱动提供的 GPU 支持。

### 3.4 安装 Docker Desktop

安装 Docker Desktop 后：

1.  General：开启 `Use the WSL 2 based engine`。
2.  Resources → WSL Integration：开启 Ubuntu-24.04。
3.  确保 Docker Desktop 使用 Linux containers。

在 Ubuntu 中验证：

``` bash
docker --version
docker compose version
```

验证 Docker 容器是否能够访问 GPU：

``` bash
docker run --rm --gpus all \
  nvidia/cuda:12.4.1-base-ubuntu22.04 \
  nvidia-smi
```

如果容器中也能看到 RTX 4060，说明 Docker GPU 环境基本打通。

------------------------------------------------------------------------

## 4. 项目目录与 Docker 部署

### 4.1 创建项目目录

建议将项目放在 WSL2 的 Linux 文件系统中，例如：

``` bash
mkdir -p ~/projects/pdf-rag
cd ~/projects/pdf-rag

mkdir -p storage/originals
mkdir -p storage/markdown
mkdir -p storage/images
mkdir -p data/huggingface
```

不要把 Qdrant 等数据库的数据目录直接放在 `/mnt/c/` 等 Windows
挂载路径下。优先使用 Docker 命名卷，或使用 WSL2 Linux
文件系统中的目录，以减少文件系统兼容性和数据持久化风险。

### 4.2 创建 `.env`

在项目根目录创建 `.env`：

``` env
POSTGRES_USER=raguser
POSTGRES_PASSWORD=请替换为强密码
POSTGRES_DB=ragdb
```

不要将包含真实密码的 `.env` 文件提交到 Git。应在 `.gitignore` 中加入：

``` gitignore
.env
.venv/
__pycache__/
```

### 4.3 `docker-compose.yml`

以下是开发阶段的起始配置。首次运行建议先启动数据库，再启动
vLLM，便于分别排查问题。

``` yaml
services:
  postgres:
    image: postgres:16
    container_name: rag-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  vllm:
    image: vllm/vllm-openai:latest
    container_name: rag-vllm
    restart: unless-stopped
    ipc: host
    ports:
      - "8000:8000"
    volumes:
      - huggingface_cache:/root/.cache/huggingface
    command:
      - --model
      - Qwen/Qwen3-4B-AWQ
      - --quantization
      - awq
      - --dtype
      - half
      - --max-model-len
      - "4096"
      - --gpu-memory-utilization
      - "0.80"
      - --max-num-seqs
      - "1"
      - --host
      - 0.0.0.0
      - --port
      - "8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  postgres_data:
  qdrant_data:
  huggingface_cache:
```

> 版本提示：`latest`
> 便于初次验证，但不利于长期复现。第一次成功运行后，应记录实际镜像版本并固定
> tag。vLLM 的模型参数和量化支持也应以所固定版本的官方文档为准。

### 4.4 启动和检查服务

``` bash
docker compose up -d postgres qdrant
docker compose ps

# 确认数据库启动后，再启动模型服务
docker compose up -d vllm

# 查看模型服务日志
docker compose logs -f vllm
```

检查 vLLM 模型列表：

``` bash
curl http://localhost:8000/v1/models
```

测试对话接口：

``` bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-AWQ",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "max_tokens": 256
  }'
```

如果启动 OOM，可按以下顺序排查：

1.  将 `--max-model-len` 从 4096 降到 2048。
2.  将 `--gpu-memory-utilization` 从 0.80 降到 0.75。
3.  确认没有其他程序占用大量显存。
4.  检查 vLLM 版本与模型量化格式是否兼容。

------------------------------------------------------------------------

## 5. Python 与 LangChain 环境

### 5.1 创建虚拟环境

在 WSL Ubuntu 中执行：

``` bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git

cd ~/projects/pdf-rag
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
```

### 5.2 安装基础依赖

``` bash
pip install \
  langchain \
  langchain-openai \
  langchain-qdrant \
  langchain-text-splitters \
  langchain-community \
  fastapi \
  uvicorn \
  sqlalchemy \
  "psycopg[binary]" \
  qdrant-client \
  sentence-transformers \
  FlagEmbedding \
  python-multipart \
  pydantic-settings
```

建议后续将依赖版本锁定到 `requirements.txt` 或
`pyproject.toml`，保证环境可复现。MinerU 可以先独立安装和测试，确认其
Python、PyTorch 和系统依赖后再集成，避免与其他包产生版本冲突。

### 5.3 加载 BGE-M3

``` python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cpu",
)

texts = [
    "LangChain 是一个用于构建大语言模型应用的框架。",
    "Qdrant 是一个向量数据库。",
]

vectors = embedding_model.encode(
    texts,
    batch_size=8,
    normalize_embeddings=True,
)

print(vectors.shape)
```

BGE-M3 输出 1024 维向量。创建 Qdrant Collection 时，向量维度需要与
Embedding 模型一致。

### 5.4 创建 Qdrant Collection

``` python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="pdf_knowledge",
    vectors_config=models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
    ),
)
```

不同 Embedding 模型生成的向量不能直接混用。更换 Embedding
模型时，应重新向量化并重建对应索引，或使用独立 Collection。

### 5.5 通过 LangChain 调用 vLLM

``` python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Qwen/Qwen3-4B-AWQ",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.1,
    max_tokens=1024,
)

response = llm.invoke(
    "请介绍一下 RAG 的基本工作流程。"
)

print(response.content)
```

`api_key="EMPTY"` 仅适用于当前未配置鉴权的本地开发服务。正式部署时应增加
API 鉴权，不要将推理服务直接暴露到公网。

------------------------------------------------------------------------

## 6. 项目代码结构

建议按业务职责拆分模块：

``` text
pdf-rag/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── documents.py
│   │   └── chat.py
│   ├── core/
│   │   ├── config.py
│   │   └── database.py
│   ├── ingestion/
│   │   ├── parser.py
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   └── indexer.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── hybrid.py
│   ├── rag/
│   │   ├── chain.py
│   │   └── prompts.py
│   └── services/
│       ├── document_service.py
│       └── chat_service.py
├── storage/
│   ├── originals/
│   ├── markdown/
│   └── images/
├── tests/
│   ├── test_retrieval.py
│   └── test_rag.py
├── docker-compose.yml
├── .env
├── .gitignore
└── requirements.txt
```

### 6.1 数据职责划分

  -----------------------------------------------------------------------------------
  存储组件                            保存内容
  ----------------------------------- -----------------------------------------------
  PostgreSQL                          用户、知识库、文档记录、任务状态、权限、Chunk
                                      元数据和关联关系

  Qdrant                              向量、Chunk 检索文本及 Payload

  文件存储                            原始 PDF、解析后的 Markdown、图片等文件
  -----------------------------------------------------------------------------------

PostgreSQL 的文档记录可以包含：

-   `id`
-   `filename`
-   `file_type`
-   `storage_path`
-   `status`
-   `created_at`

Chunk 记录可以包含：

-   `id`
-   `document_id`
-   `chunk_index`
-   `page_number`
-   `content_hash`
-   `qdrant_point_id`

Qdrant Payload 可以包含：

-   `user_id`
-   `knowledge_base_id`
-   `document_id`
-   `chunk_id`
-   `page_number`
-   `content`
-   `section`

后续应在检索时根据用户和知识库 ID 做 Metadata
Filter，避免跨用户或跨知识库检索到不应访问的数据。

------------------------------------------------------------------------

## 7. 资源规划

### 7.1 GPU 显存

RTX 4060 8GB 建议专门承担 LLM 推理。AWQ
权重文件大小不代表完整推理显存需求，实际显存还包括 KV
Cache、运行时激活、CUDA 上下文等。

初始设置：

  参数                 初始建议
  -------------------- -----------------
  LLM                  Qwen3-4B-AWQ
  量化                 AWQ 4-bit
  上下文长度           4096 tokens
  最大并发序列数       1
  GPU 显存利用率参数   0.75～0.80 起步
  Embedding            CPU
  Reranker             CPU
  PDF 解析             CPU 优先

运行时通过 `nvidia-smi`
观察实际显存占用，再决定是否增加上下文长度或并发数。

### 7.2 系统内存

32GB 内存可以先采用以下预算：

  使用场景                  建议预算
  -------------------- -------------
  Windows 宿主机          约 8～10GB
  WSL2 / Docker 服务     约 12～16GB
  其他余量                约 6～10GB

这只是初始预算，实际需要根据 PDF 解析峰值、Embedding
批量大小和数据库缓存情况调整。

### 7.3 硬盘空间

约 1TB 可用空间足以开始项目。需要预留空间给：

-   Docker 镜像和容器卷。
-   Hugging Face 模型缓存。
-   原始 PDF 和解析产物。
-   Qdrant 向量索引。
-   PostgreSQL 数据。
-   日志、备份和测试数据集。

建议定期清理无用 Docker 镜像和模型缓存，并对数据库与原始文档做独立备份。

------------------------------------------------------------------------

## 8. 分阶段实施计划与验收标准

### 阶段一：基础环境

任务：

-   安装 WSL2 和 Ubuntu 24.04。
-   安装 Docker Desktop 并启用 WSL Integration。
-   验证 WSL 中的 `nvidia-smi`。
-   验证 Docker GPU 容器。

验收标准：

-   Ubuntu 运行在 WSL2。
-   WSL 中可以识别 RTX 4060。
-   Docker 容器中可以执行 `nvidia-smi`。

### 阶段二：本地模型服务

任务：

-   部署 vLLM。
-   下载 Qwen3-4B-AWQ。
-   测试 OpenAI 兼容 API。
-   检查显存占用和推理稳定性。

验收标准：

-   `/v1/models` 能返回模型信息。
-   `/v1/chat/completions` 能正常生成回答。
-   连续多次请求没有显存溢出。

### 阶段三：数据库和向量检索

任务：

-   部署 PostgreSQL 和 Qdrant。
-   加载 BGE-M3。
-   将测试文本转换为向量。
-   实现向量入库和 Top-K 检索。

验收标准：

-   PostgreSQL 可以创建和查询业务记录。
-   Qdrant 可以写入和检索向量。
-   对测试问题能够召回预期文本。

### 阶段四：完整 PDF-RAG

任务：

-   实现 PDF 上传接口。
-   使用 MinerU 解析 PDF。
-   实现 Markdown 清洗和 Chunk 切分。
-   生成 Embedding 并写入 Qdrant。
-   使用 LangChain 完成检索、重排序和答案生成。
-   返回答案及来源文档、页码。

验收标准：

-   用户上传 PDF 后可以完成解析和索引。
-   用户可以针对文档提问。
-   答案能够提供可追溯的来源信息。

### 阶段五：工程化升级

任务：

-   Redis + Worker 异步文档解析。
-   文档增量更新、删除和幂等处理。
-   用户权限和知识库隔离。
-   RAG 检索与生成质量评估。
-   日志、异常处理、性能监控。
-   数据库备份与恢复测试。

验收标准：

-   文档处理不阻塞问答 API。
-   文档更新和删除能同步处理 PostgreSQL、Qdrant 和文件存储。
-   有固定测试集和可重复的质量评估流程。
-   能展示关键日志、任务状态和异常处理过程。

------------------------------------------------------------------------

## 9. 项目评估建议

项目跑通后，不要只用"回答看起来不错"判断效果。建议建立一套固定的测试集，至少包含：

1.  能在文档中直接找到答案的问题。
2.  需要跨多个段落综合回答的问题。
3.  涉及复杂表格的问题。
4.  涉及图片或图表的问题。
5.  文档中没有答案、模型应该拒答的问题。

检索层可以评估：

-   Recall@K：正确证据是否进入 Top-K。
-   MRR：第一个相关结果的排名。
-   nDCG：考虑相关性等级的排序质量。

生成层可以评估：

-   答案正确性。
-   忠实度 / Faithfulness：是否忠于检索到的证据。
-   引用准确性：引用的文档和页码是否真正支持答案。
-   无答案问题的拒答能力。

每次调整 Chunk 策略、Embedding、混合检索、Reranker 或
Prompt，都使用同一测试集对比，避免凭主观感受判断优化效果。

------------------------------------------------------------------------

## 10. 重要注意事项

1.  **显存限制**：8GB 显存优先运行一个 4B 量化模型。不要一开始就同时在
    GPU 上运行 LLM、Embedding 和 Reranker。
2.  **WSL 驱动**：Windows 安装支持 WSL 的 NVIDIA 驱动；不要在 WSL
    中安装独立 Linux NVIDIA 内核驱动。
3.  **数据卷位置**：数据库持久化数据优先放在 WSL2 Linux 文件系统或
    Docker 命名卷中，避免直接使用 `/mnt/c/` 下的数据库数据目录。
4.  **模型与镜像版本**：验证成功后固定模型版本和容器镜像 tag，避免
    `latest` 更新造成环境变化。
5.  **安全性**：不要将未鉴权的 vLLM、PostgreSQL 或 Qdrant
    端口直接暴露到公网。
6.  **备份**：数据库卷、原始 PDF、Markdown
    和图片资源都需要纳入备份策略。
7.  **项目定位**：这是一套单机工程化
    Demo，可用于学习和项目展示，不等同于具备高可用、容灾和多节点扩展能力的生产集群。

------------------------------------------------------------------------

## 11. 官方资料

-   vLLM 文档：https://docs.vllm.ai/
-   vLLM Docker 部署：https://docs.vllm.ai/en/latest/deployment/docker/
-   Qwen3-4B-AWQ：https://huggingface.co/Qwen/Qwen3-4B-AWQ
-   BGE-M3：https://huggingface.co/BAAI/bge-m3
-   BGE Reranker Base：https://huggingface.co/BAAI/bge-reranker-base
-   Qdrant 文档：https://qdrant.tech/documentation/
-   PostgreSQL 文档：https://www.postgresql.org/docs/
-   Microsoft WSL 文档：https://learn.microsoft.com/windows/wsl/
-   NVIDIA CUDA on WSL：https://docs.nvidia.com/cuda/wsl-user-guide/

------------------------------------------------------------------------

**推荐的第一步：** 先完成 WSL2、Docker GPU 验证和 vLLM 模型启动。确认
RTX 4060 8GB 能稳定运行 Qwen3-4B-AWQ 后，再部署 PostgreSQL、Qdrant 和
LangChain RAG 流程。
