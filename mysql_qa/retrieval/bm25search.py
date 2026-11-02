"""FAQ 检索：jieba 分词 + BM25，相似度超过阈值时直接返回标准答案。"""

import numpy as np
from numpy import argmax
from base import logger
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.utils.preprocess import preprocess_text
from rank_bm25 import BM25Okapi as BM25


class BM25Search:
    def __init__(self, redis_client=None, mysql_client=None):
        self.logger = logger
        # 允许上层注入已经创建的客户端，避免 V1/V2 同时重复建立连接。
        # 不传参数时仍保留原来的自建方式，兼容直接运行本文件的用法。
        self.redis_client = redis_client or RedisClient()
        self.mysql_client = mysql_client or MySQLClient()
        self.questions = None
        self.question_texts = None
        self.tokenized_questions = None
        self.bm25_model = None
        self.load_questions()

    def load_questions(self):
        original_key = "original_data"
        tokenizer_key = "tokenizer_data"
        self.question_texts = self.redis_client.get_data(original_key)
        self.tokenized_questions = self.redis_client.get_data(tokenizer_key)

        if self.question_texts and self.tokenized_questions and len(self.question_texts) == len(self.tokenized_questions):
            self.logger.info("问题库缓存命中")
            self.bm25_model = BM25(self.tokenized_questions)
            self.logger.info("BM25 检索模型初始化完成")
            return self.question_texts, self.tokenized_questions

        try:
            self.question_texts = self.mysql_client.fetch_questions()
            if self.question_texts:
                self.logger.info("已从 MySQL 加载问题库")
            else:
                self.logger.info("MySQL 中没有可用的问题数据")
                # 明确清空模型状态，避免空数据时 search() 访问半初始化对象。
                self.question_texts = []
                self.tokenized_questions = []
                self.bm25_model = None
                return [], []
        except Exception as e:
            self.logger.error(f"问题库加载失败：{e}")
            raise

        self.question_texts = [row[0] for row in self.question_texts]
        self.tokenized_questions = [preprocess_text(question) for question in self.question_texts]
        self.redis_client.set_data(original_key, self.question_texts)
        self.redis_client.set_data(tokenizer_key, self.tokenized_questions)
        self.logger.info("问题库已写入 Redis 缓存")
        # 只有存在语料时才创建 BM25，避免 rank_bm25 对空列表报错。
        self.bm25_model = BM25(self.tokenized_questions) if self.tokenized_questions else None
        self.logger.info("BM25 检索模型初始化完成")
        return self.question_texts, self.tokenized_questions

    def _softmax(self, scores):
        if len(scores) == 0:
            return np.array([], dtype=float)
        scores_exp = np.exp(scores - np.max(scores))
        total = np.sum(scores_exp)
        return scores_exp / total if total else np.zeros_like(scores_exp)

    def search(self, query, threshold=0.85):
        # 把只包含空白的输入视为无效查询，避免无意义的向量化和检索。
        if not isinstance(query, str) or not query.strip():
            self.logger.error("问题不能为空，且必须是字符串")
            raise ValueError("query 必须是非空字符串")
        # 统一去除首尾空白，使同一个问题的缓存键保持一致。
        normalized_query = query.strip()
        answer = self.redis_client.get_answer(normalized_query)
        if answer:
            self.logger.info("已从 Redis 缓存中获取答案")
            return answer
        try:
            # 离线 CSV 为空或数据库暂无数据时，直接回退 RAG，避免 NoneType 异常。
            if self.bm25_model is None or not self.question_texts:
                self.logger.info("BM25 问题库为空，跳过 FAQ 检索")
                return None
            tokenizer_question = preprocess_text(normalized_query)
            scores = self.bm25_model.get_scores(tokenizer_question)
            softmax_scores = self._softmax(scores)
            best_idx = argmax(softmax_scores)
            best_scores = softmax_scores[best_idx]
            best_text = self.question_texts[best_idx]
            print(f"最相似的问题：{best_text}")
            print(f"相似度：{best_scores}")

            if best_scores > threshold:
                answer = self.mysql_client.answer(best_text)
                if answer:
                    self.redis_client.set_data(f"answer:{normalized_query}", answer)
                    self.logger.info(f"答案匹配成功，相似度：{best_scores},答案已写入 Redis 缓存")
                    return answer
            else:
                self.logger.info(f"未找到可靠答案，最高相似度：{best_scores}<阈值：{threshold}")
                return None
        except Exception as e:
            self.logger.error(f"答案检索失败，数据库或检索模块异常：{e}")
            # 显式返回 None，调用方可稳定地回退到 RAG。
            return None

if __name__ == "__main__":
    file = BM25Search()
    question_texts, tokenized_questions = file.load_questions()
    file.search("什么是Python的列表推导式")
