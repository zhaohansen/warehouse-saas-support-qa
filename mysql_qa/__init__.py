"""FAQ 通道入口：按需暴露 MySQL 客户端、Redis 客户端与 BM25 检索。"""

__all__ = ["MySQLClient", "RedisClient", "BM25Search"]


def __getattr__(name):
    if name == "MySQLClient":
        from mysql_qa.db.mysql_client import MySQLClient

        return MySQLClient

    if name == "RedisClient":
        from mysql_qa.cache.redis_client import RedisClient

        return RedisClient

    if name == "BM25Search":
        from mysql_qa.retrieval.bm25search import BM25Search

        return BM25Search

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
