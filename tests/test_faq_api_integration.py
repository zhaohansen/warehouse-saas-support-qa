"""应用接口层验证：FAQ 命中、FAQ 未命中走 RAG、CSV 回退与多轮追问。"""

import os
import tempfile
import threading
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import rag_qa

# 接口测试不加载向量库和嵌入模型。
rag_qa.VectorStore = object
rag_qa.RAGSystem = object

from fastapi.testclient import TestClient  # noqa: E402

import app as web  # noqa: E402
from base.config import Config  # noqa: E402
from base.conversation_store import ConversationStore  # noqa: E402
from cli import IntegratedQASystem  # noqa: E402
from mysql_qa.retrieval.bm25search import BM25Search  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeRedis:
    def get_data(self, key):
        return None

    def set_data(self, key, value, ttl=None):
        pass


class FakeRAG:
    def __init__(self, answer='RAG 模拟回答'):
        self.answer = answer
        self.calls = []

    def generate_answer_stream(self, query, source_filter=None, history=None, on_context=None,
                               cancel_event=None, filters=None):
        self.calls.append(dict(query=query, source_filter=source_filter, history=list(history or [])))
        if on_context:
            on_context([])
        yield self.answer


class FakeImporter:
    def delete_file(self, source, document_id):
        return {'status': 'deleted'}

    def list_documents(self, source):
        return []


def make_search(source='MySQL'):
    search = BM25Search.__new__(BM25Search)
    search.records = [record for record in _csv_records()]
    search.revision = 1
    search.bm25_model = None
    search.faq_source = source
    search.redis_client = FakeRedis()
    return search


def _csv_records():
    from mysql_qa.db.mysql_client import MySQLClient
    client = MySQLClient.__new__(MySQLClient)
    client.csv_path = ROOT / 'mysql_qa/db/faq_knowledge_base.csv'
    return client._csv_records()


def make_qa(search, rag):
    directory = tempfile.TemporaryDirectory()
    qa = IntegratedQASystem.__new__(IntegratedQASystem)
    qa._session_locks = weakref.WeakValueDictionary()
    qa._session_guard = threading.Lock()
    qa.config = types.SimpleNamespace(VALID_SOURCES=Config.VALID_SOURCES, FAQ_THRESHOLD=0.85)
    qa.logger = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None,
                                      warning=lambda *a, **k: None, exception=lambda *a, **k: None)
    qa.mysql_client = None
    qa.redis_client = None
    qa.bm25_search = search
    qa.history = ConversationStore(os.path.join(directory.name, 'history.sqlite3'))
    qa.client = None
    qa.vector_store = None
    qa.rag_system = rag
    qa._temp_directory = directory
    return qa


class FaqApiIntegrationTests(unittest.TestCase):
    def make_client(self, faq_source='MySQL'):
        rag = FakeRAG()
        qa = make_qa(make_search(faq_source), rag)
        self.addCleanup(qa._temp_directory.cleanup)
        env = patch.dict(os.environ, {'QA_ACCESS_TOKEN': 'test-token'})
        env.start()
        self.addCleanup(env.stop)
        client = TestClient(web.create_app(qa_factory=lambda: qa,
                                           importer_factory=lambda _: FakeImporter()))
        self.addCleanup(client.close)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        client.headers['Authorization'] = 'Bearer test-token'
        return client, rag

    def test_faq_hit_returns_standard_answer_and_source_document(self):
        client, rag = self.make_client()
        response = client.post('/api/query', json={'query': '抖店电子面单打印数据用哪个接口？'})
        data = response.json()
        self.assertIn('/logistics/waybillApply', data['answer'])
        self.assertEqual(data['sources'][0]['title'], '标准问答库 · 02-抖店-电子面单取号与打印.txt')
        self.assertEqual(data['sources'][0]['source_doc'], '02-抖店-电子面单取号与打印.txt')
        self.assertEqual(rag.calls, [])

    def test_faq_miss_continues_to_rag(self):
        client, rag = self.make_client()
        response = client.post('/api/query', json={'query': '旺店通客户端 V2.4.4 能用 trade_push.php 吗？'})
        data = response.json()
        self.assertEqual(data['answer'], 'RAG 模拟回答')
        self.assertEqual(data['sources'], [])
        self.assertEqual(len(rag.calls), 1)
        self.assertEqual(rag.calls[0]['query'], '旺店通客户端 V2.4.4 能用 trade_push.php 吗？')

    def test_general_question_uses_rag_without_sources(self):
        client, rag = self.make_client()
        data = client.post('/api/query', json={'query': '什么是电子面单？'}).json()
        self.assertEqual(data['answer'], 'RAG 模拟回答')
        self.assertEqual(data['sources'], [])
        self.assertEqual(len(rag.calls), 1)

    def test_faq_still_answers_when_search_uses_csv_fallback(self):
        client, rag = self.make_client(faq_source='CSV')
        data = client.post('/api/query', json={'query': '抖店电子面单打印数据用哪个接口？'}).json()
        self.assertIn('/logistics/waybillApply', data['answer'])
        self.assertEqual(data['sources'][0]['title'], '标准问答库 · 02-抖店-电子面单取号与打印.txt')
        self.assertEqual(rag.calls, [])

    def test_follow_up_carries_previous_question_into_rag(self):
        client, rag = self.make_client()
        session_id = '11111111-1111-4111-8111-111111111111'
        first = client.post('/api/query', json={'query': '抖店电子面单打印出来是空白，可能是什么原因？',
                                                'session_id': session_id}).json()
        second = client.post('/api/query', json={'query': '那库存回写呢？',
                                                 'session_id': session_id}).json()
        self.assertEqual(first['answer'], 'RAG 模拟回答')
        self.assertEqual(second['answer'], 'RAG 模拟回答')
        self.assertEqual(rag.calls[1]['history'][0]['question'],
                         '抖店电子面单打印出来是空白，可能是什么原因？')

    def test_faq_hit_then_follow_up_reads_same_history_key(self):
        client, rag = self.make_client()
        session_id = '22222222-2222-4222-8222-222222222222'
        first = client.post('/api/query', json={'query': '抖店电子面单打印数据用哪个接口？',
                                                'session_id': session_id,
                                                'source_filter': 'platform_docs'}).json()
        self.assertIn('/logistics/waybillApply', first['answer'])
        self.assertEqual(rag.calls, [])
        second = client.post('/api/query', json={'query': '那打印组件呢？',
                                                 'session_id': session_id,
                                                 'source_filter': 'platform_docs'}).json()
        self.assertEqual(second['answer'], 'RAG 模拟回答')
        self.assertEqual(len(rag.calls), 1)
        self.assertEqual(rag.calls[0]['history'][0]['question'],
                         '抖店电子面单打印数据用哪个接口？')

    def test_switching_category_does_not_mix_old_history(self):
        client, rag = self.make_client()
        session_id = '33333333-3333-4333-8333-333333333333'
        first = client.post('/api/query', json={'query': '抖店电子面单打印数据用哪个接口？',
                                                'session_id': session_id,
                                                'source_filter': 'platform_docs'}).json()
        self.assertIn('/logistics/waybillApply', first['answer'])
        second = client.post('/api/query', json={'query': '那打印组件呢？',
                                                 'session_id': session_id,
                                                 'source_filter': 'product_docs'}).json()
        self.assertEqual(second['answer'], 'RAG 模拟回答')
        self.assertEqual(rag.calls[0]['history'], [])

    def test_switching_product_or_version_does_not_mix_old_history(self):
        client, rag = self.make_client()
        session_id = '44444444-4444-4444-8444-444444444444'
        client.post('/api/query', json={'query': '旺店通客户端 V2.4.4 能用创建原始订单接口吗？',
                                        'session_id': session_id,
                                        'product': '旺店通ERP'}).json()
        self.assertEqual(len(rag.calls), 1)
        client.post('/api/query', json={'query': '那升级后呢？',
                                        'session_id': session_id,
                                        'product': '旺店通ERP',
                                        'version': '客户端 V2.4.5.0'}).json()
        self.assertEqual(len(rag.calls), 2)
        self.assertEqual(rag.calls[1]['history'], [])
        client.post('/api/query', json={'query': '那平台侧呢？',
                                        'session_id': session_id,
                                        'product': '旺店通ERP',
                                        'platform': '抖音电商'}).json()
        self.assertEqual(len(rag.calls), 3)
        self.assertEqual(rag.calls[2]['history'], [])


if __name__ == '__main__':
    unittest.main()
