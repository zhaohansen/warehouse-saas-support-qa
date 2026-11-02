"""配置加载：读取 config.ini 与环境变量，并把相对路径解析为绝对路径。"""

from configparser import ConfigParser
from pathlib import Path
import os

config_file = Path(__file__).with_name("config.ini")
parser = ConfigParser()
if not parser.read(config_file, encoding="utf-8"):
    raise FileNotFoundError(f"配置文件不存在: {config_file}")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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
    DASHSCOPE_API_KEY = parser.get(
        "llm", "api_key", fallback=os.getenv("DASHSCOPE_API_KEY", "")
    )
    DASHSCOPE_BASE_URL = parser.get(
        "llm",
        "base_url",
        fallback="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    LLM_MODEL = parser.get("llm", "model", fallback="qwen-plus")
    VALID_SOURCES = [
        item.strip()
        for item in parser.get("app", "valid_sources", fallback="ai").split(",")
        if item.strip()
    ]
    CUSTOMER_SERVICE_PHONE = parser.get("app", "customer_service_phone", fallback="")
    DATA_DIR = str((PROJECT_ROOT / data_dir).resolve())
    MODELS_DIR = str((PROJECT_ROOT / "models").resolve())
    MILVUS_HOST = milvus_host
    MILVUS_PORT = milvus_port
    MILVUS_DATABASE_NAME = milvus_database
    MILVUS_COLLECTION_NAME = milvus_collection
    PARENT_CHUNK_SIZE = parent_chunk_size
    CHILD_CHUNK_SIZE = child_chunk_size
    CHUNK_OVERLAP = chunk_overlap
    BATCH_SIZE = batch_size
    RETRIEVAL_K = retrieval_k
    CANDIDATE_M = candidate_m
    DENSE_WEIGHT = dense_weight
    SPARSE_WEIGHT = sparse_weight
    NPROBE = nprobe
    MIN_SCORE = min_score
    EMBEDDING_MODEL_PATH = embedding_model_path
    EMBEDDING_DEVICE = embedding_device
    EMBEDDING_USE_FP16 = embedding_use_fp16
    RERANKER_MODEL_PATH = reranker_model_path
    RERANKER_DEVICE = reranker_device

# 兼容两种用法: Config.xxx (类属性) 和 config.xxx (单例)
config = Config()
