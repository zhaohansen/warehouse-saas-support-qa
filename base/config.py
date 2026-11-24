"""配置加载：读取 config.ini 与环境变量，并把相对路径解析为绝对路径。"""

from configparser import ConfigParser
from pathlib import Path
from urllib.parse import urlsplit
import os

config_file = Path(os.getenv("QA_CONFIG_PATH", str(Path(__file__).with_name("config.ini"))))
parser = ConfigParser(interpolation=None)
if not parser.read(config_file, encoding="utf-8"):
    raise FileNotFoundError(f"配置文件不存在: {config_file}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_ROOT = PROJECT_ROOT / "storage" / "private_knowledge"


def _split_csv(raw):
    return [item.strip() for item in (raw or '').split(',') if item.strip()]


def _parse_pairs(raw):
    pairs = {}
    for item in _split_csv(raw):
        if ':' in item:
            key, value = item.split(':', 1)
            if key.strip() and value.strip():
                pairs[key.strip()] = value.strip()
    return pairs


class Config:
    mysql_host = parser.get("mysql", "host")
    mysql_user = parser.get("mysql", "user")
    mysql_password = parser.get("mysql", "password")
    mysql_db = parser.get("mysql", "database")
    redis_host = parser.get("redis", "host")
    redis_port = parser.getint("redis", "port")
    redis_password = parser.get("redis", "password")
    redis_db = parser.getint("redis", "db")
    log_file = parser.get("logger", "log_file", fallback="logs/app.log")
    milvus_host = parser.get("milvus", "host")
    milvus_port = parser.getint("milvus", "port")
    milvus_database = parser.get("milvus", "database")
    milvus_collection = parser.get("milvus", "collection")
    # 可选：显式连接串。留空时使用 http://host:port 连接 Milvus 服务；
    # 填本地文件路径（如 ./storage/milvus_lite.db）时使用 Milvus Lite，便于无 Docker 演示。
    # 注意：pymilvus 也会读取环境变量 MILVUS_URI 且只接受服务地址，
    # 因此 Milvus Lite 文件路径请用 QA_MILVUS_URI，避免与 pymilvus 的同名变量冲突。
    qa_milvus_uri = os.getenv("QA_MILVUS_URI", "").strip()
    server_uri = os.getenv("MILVUS_URI", "").strip()
    if server_uri and not server_uri.startswith(("http://", "https://", "tcp://")):
        server_uri = ""
    milvus_uri = qa_milvus_uri or server_uri or parser.get("milvus", "uri", fallback="").strip()
    embedding_model_path = parser.get("embedding", "model_path")
    embedding_device = parser.get("embedding", "device")
    embedding_use_fp16 = parser.getboolean("embedding", "use_fp16")
    reranker_model_path = parser.get("reranker", "model_path")
    reranker_device = parser.get("reranker", "device")
    data_dir = parser.get("document", "data_dir")
    parent_chunk_size = parser.getint("document", "parent_chunk_size")
    child_chunk_size = parser.getint("document", "child_chunk_size")
    chunk_overlap = parser.getint("document", "chunk_overlap")
    batch_size = parser.getint("document", "batch_size", fallback=32)
    retrieval_k = parser.getint("retrieval", "retrieval_k")
    candidate_m = parser.getint("retrieval", "candidate_m")
    dense_weight = parser.getfloat("retrieval", "dense_weight")
    sparse_weight = parser.getfloat("retrieval", "sparse_weight")
    nprobe = parser.getint("retrieval", "nprobe")
    min_score = parser.getfloat("retrieval", "min_score")
    LLM_BASE_URL = parser.get(
        "llm",
        "base_url",
        fallback="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    if urlsplit(LLM_BASE_URL).hostname == "api.deepseek.com":
        LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
    else:
        LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip() or parser.get("llm", "api_key", fallback="").strip()
    LLM_MODEL = parser.get("llm", "model", fallback="qwen-plus")
    VALID_SOURCES = _split_csv(parser.get("app", "valid_sources", fallback="ai"))
    SOURCE_LABELS = _parse_pairs(parser.get("app", "source_labels", fallback=""))
    for _source in VALID_SOURCES:
        SOURCE_LABELS.setdefault(_source, _source)
    FILTER_PRODUCTS = _split_csv(parser.get("app", "filter_products", fallback=""))
    FILTER_PLATFORMS = _split_csv(parser.get("app", "filter_platforms", fallback=""))
    FILTER_VERSIONS = _split_csv(parser.get("app", "filter_versions", fallback=""))
    FAQ_THRESHOLD = parser.getfloat("app", "faq_threshold", fallback=0.85)
    if not 0 <= FAQ_THRESHOLD <= 1:
        raise ValueError("faq_threshold 必须在 0 到 1 之间")
    DATA_DIR = str((PROJECT_ROOT / data_dir).resolve())
    MODELS_DIR = str((PROJECT_ROOT / "rag_qa" / "models").resolve())
    # 可用 QA_HISTORY_PATH 指向独立会话库，便于隔离验证，避免读写正式历史记录。
    HISTORY_PATH = os.getenv("QA_HISTORY_PATH", "").strip() or str(
        PROJECT_ROOT / "storage" / "warehouse_history.sqlite3")
    MAX_OUTPUT_TOKENS = parser.getint("llm", "max_output_tokens", fallback=4096)
    MAX_CONTEXT_CHARS = parser.getint("retrieval", "max_context_chars", fallback=12000)
    MAX_HISTORY_CHARS = parser.getint("app", "max_history_chars", fallback=12000)
    MILVUS_HOST = milvus_host
    MILVUS_PORT = milvus_port
    MILVUS_DATABASE_NAME = milvus_database
    MILVUS_COLLECTION_NAME = milvus_collection
    MILVUS_URI = milvus_uri
    PARENT_CHUNK_SIZE = parent_chunk_size
    CHILD_CHUNK_SIZE = child_chunk_size
    CHUNK_OVERLAP = chunk_overlap
    BATCH_SIZE = batch_size
    RETRIEVAL_K = retrieval_k
    RECALL_CANDIDATES = max(retrieval_k, min(100, parser.getint("retrieval", "recall_candidates", fallback=20)))
    CANDIDATE_M = candidate_m
    DENSE_WEIGHT = dense_weight
    SPARSE_WEIGHT = sparse_weight
    NPROBE = nprobe
    MIN_SCORE = min_score
    MIN_ANSWER_SCORE = parser.getfloat("retrieval", "min_answer_score", fallback=0.35)
    EMBEDDING_MODEL_PATH = embedding_model_path
    EMBEDDING_DEVICE = embedding_device
    EMBEDDING_USE_FP16 = embedding_use_fp16
    RERANKER_MODEL_PATH = reranker_model_path
    RERANKER_DEVICE = reranker_device

    PRIVATE_MODE = os.getenv("QA_PRIVATE_MODE", "").strip() == "1"
    if PRIVATE_MODE:
        # Keep the data, conversation history and index distinct from the public demo.
        DATA_DIR = str(PRIVATE_ROOT / "documents")
        HISTORY_PATH = str(PRIVATE_ROOT / "history.sqlite3")
        log_file = str(PRIVATE_ROOT / "app.log")
        MILVUS_HOST = "127.0.0.1"
        MILVUS_URI = f"http://127.0.0.1:{MILVUS_PORT}"
        MILVUS_COLLECTION_NAME = "warehouse_private_kb"
        VALID_SOURCES = ["internal_docs"]
        SOURCE_LABELS = {"internal_docs": "内部资料"}
        FILTER_PRODUCTS = []
        FILTER_PLATFORMS = []
        FILTER_VERSIONS = []
        LLM_BASE_URL = "http://127.0.0.1:11434/v1"
        LLM_MODEL = "qwen2.5:7b"
        LLM_API_KEY = "ollama"


def _validate_config():
    positive = ('BATCH_SIZE', 'RETRIEVAL_K', 'CANDIDATE_M', 'NPROBE', 'MAX_OUTPUT_TOKENS',
                'MAX_CONTEXT_CHARS', 'MAX_HISTORY_CHARS', 'PARENT_CHUNK_SIZE', 'CHILD_CHUNK_SIZE')
    for name in positive:
        if getattr(Config, name) <= 0:
            raise ValueError(f'配置项 {name} 必须大于 0')
    if Config.CHUNK_OVERLAP < 0 or Config.CHUNK_OVERLAP >= min(Config.PARENT_CHUNK_SIZE,
                                                                Config.CHILD_CHUNK_SIZE):
        raise ValueError('chunk_overlap 必须大于等于 0 且小于父块、子块长度')
    if not 0 <= Config.MIN_SCORE <= 1:
        raise ValueError('min_score 必须在 0 到 1 之间')
    if not 0 <= Config.MIN_ANSWER_SCORE <= 1:
        raise ValueError('min_answer_score 必须在 0 到 1 之间')
    if Config.DENSE_WEIGHT < 0 or Config.SPARSE_WEIGHT < 0 or (
            Config.DENSE_WEIGHT + Config.SPARSE_WEIGHT <= 0):
        raise ValueError('检索权重不能为负，且不能同时为 0')
    if not Config.VALID_SOURCES:
        raise ValueError('valid_sources 至少配置一个资料分类')


_validate_config()

# 兼容两种用法: Config.xxx (类属性) 和 config.xxx (单例)
config = Config()
