import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

# 接口测试无需加载 GPU、Milvus 或真实问答模型。
pipeline = types.ModuleType('rag_qa.core.rag_pipeline')
pipeline.RetrievalError = type('RetrievalError', (RuntimeError,), {})
from fastapi.testclient import TestClient
with patch.dict(sys.modules, {'rag_qa.core.rag_pipeline': pipeline}):
    import app as web

class FakeQA:
    def __init__(self, ready=True):
        self.vector_store = object() if ready else None
        self.config = types.SimpleNamespace(VALID_SOURCES=['python'])
    def _build_filters(self, product, platform, version):
        return {key: value for key, value in
                (('product', product), ('platform', platform), ('version', version)) if value}
    def list_sessions(self, limit): return []
    def query(self, query, **kwargs):
        yield '模拟回答', True
    def close(self): pass

class FakeImporter:
    def delete_file(self, source, document_id):
        if document_id == 'missing': raise FileNotFoundError('资料不存在')
        return {'status': 'deleted'}
    def list_documents(self, source): return [{'id': 'example', 'source': 'python'}]

class WebTests(unittest.TestCase):
    def make_client(self, ready=True, qa_factory=None):
        env = patch.dict(os.environ, {'QA_ACCESS_TOKEN': 'test-token'})
        env.start()
        self.addCleanup(env.stop)
        client = TestClient(web.create_app(qa_factory=qa_factory or (lambda: FakeQA(ready)),
                                          importer_factory=lambda qa: FakeImporter()))
        self.addCleanup(client.close)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        client.headers['Authorization'] = 'Bearer test-token'
        return client

    def test_delete_reports_service_unavailable(self):
        client = self.make_client(ready=False)
        response = client.delete('/api/documents?source=python&document_id=example')
        self.assertEqual(response.status_code, 503)

    def test_delete_and_missing(self):
        client = self.make_client()
        self.assertEqual(client.delete('/api/documents?source=python&document_id=example').json()['status'], 'deleted')
        self.assertEqual(client.delete('/api/documents?source=python&document_id=missing').status_code, 404)

    def test_auth_and_cross_origin(self):
        client = self.make_client()
        self.assertEqual(client.get('/api/sources').json()['sources'], ['python'])
        self.assertEqual(client.delete('/api/documents?source=python&document_id=example', headers={'Origin':'https://example.com'}).status_code, 403)
        client.headers.pop('Authorization')
        self.assertEqual(client.get('/api/sources').status_code, 401)

    def test_websocket_query(self):
        client = self.make_client()
        with client.websocket_connect('/api/stream', headers={'Authorization': 'Bearer test-token'}) as socket:
            socket.send_json({'query': '什么是决策树', 'source_filter': 'python'})
            messages = [socket.receive_json(), socket.receive_json(), socket.receive_json()]
        self.assertEqual([message['type'] for message in messages], ['start', 'token', 'end'])
        self.assertEqual(messages[1]['token'], '模拟回答')

    def test_disconnected_stream_closes_generator_after_worker_exits(self):
        running, cancelled, release, closed = (threading.Event() for _ in range(4))

        class PausedQA(FakeQA):
            def query(self, query, cancel_event, **kwargs):
                def generate():
                    try:
                        yield '第一段', False
                        running.set()
                        if cancel_event.wait(2):
                            cancelled.set()
                        release.wait(2)
                        yield '未发送的部分', False
                    finally:
                        closed.set()
                self.generator = generate()
                return self.generator

        qa = PausedQA()
        client = self.make_client(qa_factory=lambda: qa)
        with client.websocket_connect('/api/stream', headers={'Authorization': 'Bearer test-token'}) as socket:
            socket.send_json({'query': '测试流式取消'})
            self.assertEqual(socket.receive_json()['type'], 'start')
            self.assertEqual(socket.receive_json()['type'], 'token')
            self.assertTrue(running.wait(2))
            socket.close()
            try:
                self.assertTrue(cancelled.wait(2))
                self.assertFalse(closed.is_set())
            finally:
                release.set()
            self.assertTrue(closed.wait(2))

    def test_websocket_keeps_multiple_tokens_and_sources(self):
        class ChunkedQA(FakeQA):
            def query(self, query, **kwargs):
                callback = kwargs.get('source_callback')
                if callback:
                    callback([{'index': 1, 'title': '产品手册', 'source_doc': '虚构演示-协作平台产品手册.txt'}])
                yield '第一段', False
                yield '第二段 [1]', True

        env = patch.dict(os.environ, {'QA_ACCESS_TOKEN': 'test-token'})
        env.start()
        self.addCleanup(env.stop)
        client = TestClient(web.create_app(qa_factory=lambda: ChunkedQA(),
                                          importer_factory=lambda qa: FakeImporter()))
        self.addCleanup(client.close)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        with client.websocket_connect('/api/stream', headers={'Authorization': 'Bearer test-token'}) as socket:
            socket.send_json({'query': '无法登录协作平台应该联系谁？', 'source_filter': 'python'})
            messages = [socket.receive_json() for _ in range(4)]
        self.assertEqual([message['type'] for message in messages], ['start', 'token', 'token', 'end'])
        self.assertEqual(''.join(message['token'] for message in messages if message['type'] == 'token'),
                         '第一段第二段 [1]')
        self.assertEqual(messages[-1]['sources'][0]['title'], '产品手册')

if __name__ == '__main__': unittest.main()
