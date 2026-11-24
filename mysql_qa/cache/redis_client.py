"""带容量与过期限制的缓存；Redis 运行时不可用时回退到内存缓存。"""
import json
import threading
import time
from collections import OrderedDict
import redis
from base import Config
from base.logger import logger


class RedisClient:
    def __init__(self):
        self.client = None
        self._memory = OrderedDict()
        self._lock = threading.RLock()
        config = Config()
        try:
            self.client = redis.Redis(host=config.redis_host, port=config.redis_port, db=config.redis_db,
                                      password=config.redis_password, decode_responses=True, socket_connect_timeout=2,
                                      socket_timeout=2)
            self.client.ping()
        except redis.RedisError:
            self.client = None
            logger.warning('Redis 不可用，使用有容量限制的本地缓存')

    def set_data(self, key, value, ttl=600):
        with self._lock:
            self._memory[key] = (time.monotonic() + ttl, value)
            self._memory.move_to_end(key)
            while len(self._memory) > 1000:
                self._memory.popitem(last=False)
            if self.client is not None:
                try:
                    self.client.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
                except redis.RedisError:
                    self.client = None
                    logger.warning('Redis 连接中断，已切换至本地缓存')

    def get_data(self, key):
        with self._lock:
            if self.client is not None:
                try:
                    value = self.client.get(key)
                    if value is not None:
                        decoded = json.loads(value)
                        if decoded:
                            logger.info("Redis 缓存命中，返回答案")
                        return decoded
                except (redis.RedisError, ValueError, TypeError):
                    self.client = None
            record = self._memory.get(key)
            if record is None:
                return None
            expiry, value = record
            if expiry <= time.monotonic():
                del self._memory[key]
                return None
            self._memory.move_to_end(key)
            if value:
                logger.info("本地缓存命中，返回答案")
            return value

    def get_answer(self, query):
        return self.get_data('answer:' + query)

    def close(self):
        if self.client is not None:
            self.client.close()
