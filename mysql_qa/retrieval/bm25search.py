"""FAQ 检索：使用精确匹配、保守的词汇相似度评分和学科隔离。"""
import hashlib
import json
import re
from difflib import SequenceMatcher
from rank_bm25 import BM25Okapi
from base.logger import logger
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.utils.preprocess import preprocess_text


def normalize_question(text):
    return re.sub(r'[\s?？!！。]+', '', text).casefold()


def lexical_similarity(query, question):
    left, right = normalize_question(query), normalize_question(question)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if bool(re.search(r'不能|不是|不会|不支持|不可以|不要|不可|不应', left)) != bool(
            re.search(r'不能|不是|不会|不支持|不可以|不要|不可|不应', right)):
        return 0.0
    a = {t.casefold() for t in preprocess_text(query) if re.search(r'\w', t)}
    b = {t.casefold() for t in preprocess_text(question) if re.search(r'\w', t)}
    overlap = len(a & b)
    if not overlap: return 0.0
    f1 = 2 * overlap / (len(a) + len(b))
    return 0.6 * SequenceMatcher(None, left, right).ratio() + 0.4 * f1


class BM25Search:
    def __init__(self, redis_client=None, mysql_client=None):
        self.redis_client = redis_client or RedisClient()
        self.mysql_client = mysql_client or MySQLClient()
        self.load_questions()

    def load_questions(self):
        self.records = self.mysql_client.fetch_faq() or []
        self.faq_source = getattr(self.mysql_client, "faq_source", None)
        self.revision = hashlib.sha256(
            json.dumps(self.records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        self.question_texts = [r['question'] for r in self.records]
        self.tokenized_questions = [preprocess_text(q) or ['__empty__'] for q in self.question_texts]
        self.bm25_model = BM25Okapi(self.tokenized_questions) if self.records else None
        return self.question_texts, self.tokenized_questions

    def search(self, query, threshold=0.85, source_filter=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError('问题不能为空')
        candidates = [r for r in self.records if not source_filter or r['source'] == source_filter]
        if not candidates:
            return None
        key_data = [self.revision, source_filter, threshold, normalize_question(query)]
        key = 'faq:v3:' + hashlib.sha256(json.dumps(key_data, ensure_ascii=False).encode()).hexdigest()
        cached = self.redis_client.get_data(key)
        if cached: return cached
        exact = [r for r in candidates if normalize_question(r['question']) == normalize_question(query)]
        if exact:
            # 候选答案冲突时存在歧义，交由后续检索处理。
            answers = {r['answer'] for r in exact}
            if len(answers) != 1: return None
            answer = exact[0]['answer']
        else:
            if self.bm25_model is not None and len(candidates) > 20:
                raw_scores = self.bm25_model.get_scores(preprocess_text(query))
                allowed = {id(r) for r in candidates}
                candidates = [r for _, r in
                              sorted(zip(raw_scores, self.records), key=lambda pair: pair[0], reverse=True)
                              if id(r) in allowed][:20]
            ranked = sorted(((lexical_similarity(query, r['question']), r) for r in candidates),
                            key=lambda pair: pair[0], reverse=True)
            score, record = ranked[0]
            if score < threshold: return None
            # 所有与最高分接近的候选中只要存在冲突答案，就应交给知识检索消歧。
            ambiguous = [r for candidate_score, r in ranked if score - candidate_score < 0.03]
            if len({r['answer'] for r in ambiguous}) > 1:
                return None
            answer = record['answer']
        self.redis_client.set_data(key, answer, ttl=600)
        source = getattr(self, "faq_source", None)
        if source:
            logger.info("FAQ 题库匹配成功，返回答案（题库来源：%s）", source)
        else:
            logger.info("FAQ 题库匹配成功，返回答案")
        return answer
