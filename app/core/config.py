"""应用配置：统一从项目根 `.env` 读取，避免散落的 os.environ 访问。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "pdf-rag"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # PostgreSQL：业务唯一权威（文档、分块、任务状态）
    database_url: str = (
        "postgresql+psycopg://raguser:change-me-strong-password@localhost:5432/ragdb"
    )
    db_echo: bool = False

    # Qdrant：只存 child 向量与定位 payload
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "pdf_knowledge"
    qdrant_vector_size: int = 1024

    # vLLM / Qwen3-4B-AWQ（OpenAI 兼容接口）
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3-4B-AWQ"
    llm_api_key: str = "EMPTY"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Embedding / Reranker 均为独立服务（compose 内覆盖为 embedder:8001 / reranker:8002），
    # 模型加载与推理由服务端负责；api 容器只保留业务逻辑
    hf_endpoint: str = "https://hf-mirror.com"
    hf_hub_offline: int = 0
    embedding_model: str = "BAAI/bge-m3"
    embedding_api_url: str = "http://localhost:8001/v1"
    # 本地 embedder 留空；切换云端 OpenAI 兼容服务时填 key
    embedding_api_key: str = ""
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8

    # 78 题消融结论：hybrid 检索已把目标排进 top-5，rerank 净效果≈0（+1 题）
    # 而单次精排 ~3.6s；默认关闭，需要时经 .env 或前端勾选开启
    reranker_enabled: bool = False
    reranker_api_url: str = "http://localhost:8002/v1"
    # 本地 reranker 留空；切换云端 Cohere 兼容服务时填 key
    reranker_api_key: str = ""
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    reranker_max_length: int = 512

    # 检索参数：沿用现有 78 题评测结论；上下文预算需与 vLLM 4096 上下文匹配
    retrieval_candidate_depth: int = 20
    retrieval_default_k: int = 5
    hybrid_bm25_weight: float = 0.2
    rrf_k: int = 60
    context_budget_chars: int = 2500
    merge_parents: bool = True

    # 分块参数
    chunk_size: int = 500
    chunk_overlap: int = 50
    # 提示框补充块超长时的再切分上限
    supplement_max_chars: int = 6000

    # 父块 floor-merge 下限聚合：小碎片父块（<floor）与同章同节内相邻碎片
    # 合并，合并后不超 ceiling；breadcrumb 取组内首个 fragment 的具体层级。
    # 默认关闭（=父块切分后不做合并）：实测合并对检索指标影响很小，
    # 关闭可保留每个小节的独立父块，引用溯源更精确。
    parent_merge_enabled: bool = False
    parent_merge_floor_chars: int = 300
    parent_merge_ceiling_chars: int = 1000

    # 提示框几何检测（物理签名，按 PDF 模板调整；默认值为秦PLUS EV 手册实测，
    # 全书 302 页验证：184 个标题条 100% 命中、0 例框内表格、2 例跨页框）
    # 关闭后提示框补充块退回文本启发式定界（旧行为）
    box_detect_enabled: bool = True
    box_bar_fill_hex: str = "#ffe3bf"
    box_color_tolerance: float = 0.02
    box_bar_min_width_pt: int = 300
    box_bar_max_width_pt: int = 380
    box_bar_min_height_pt: int = 18
    box_bar_max_height_pt: int = 32
    box_bottom_search_pt: int = 400
    box_x_tolerance_pt: int = 5

    # PDF 解析：项目自研解析链路（标题物理探测 + 表格重建 + 清洗）
    pdf_parser: str = "custom"

    # 存储与业务默认值
    max_upload_mb: int = 200
    storage_dir: str = "storage"
    default_user_id: str = "local"
    default_kb_id: str = "default"
    chat_mode: str = "rag"
    agent_max_iterations: int = 3

    # MCP 端点（/mcp/）的 Host 允许列表：逗号分隔，支持 host:* 通配端口；
    # 留空关闭 DNS 重绑定校验（内网 demo 默认），公网部署建议填实际域名
    mcp_allowed_hosts: str = ""

    @property
    def storage_path(self) -> Path:
        path = Path(self.storage_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def originals_dir(self) -> Path:
        return self.storage_path / "originals"

    @property
    def markdown_dir(self) -> Path:
        return self.storage_path / "markdown"

    @property
    def images_dir(self) -> Path:
        return self.storage_path / "images"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for path in (self.originals_dir, self.markdown_dir, self.images_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
