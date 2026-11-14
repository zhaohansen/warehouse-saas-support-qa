"""FAQ 数据、幂等入库和 CSV 回退的回归测试（不需要真实 MySQL）。"""

import csv
import tempfile
import threading
import unittest
from pathlib import Path

from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrieval.bm25search import BM25Search

ROOT = Path(__file__).resolve().parents[1]
FAQ_CSV = ROOT / 'mysql_qa' / 'db' / 'faq_knowledge_base.csv'


class FakeCursor:
    def __init__(self):
        self.existing = set()
        self.inserted = []
        self._result = None

    def execute(self, sql, params=None):
        normalized = ' '.join(sql.split())
        if normalized.startswith('SELECT id'):
            self._result = (1,) if (params[0], params[1]) in self.existing else None
        elif normalized.startswith('INSERT'):
            self.inserted.append(params)
            self.existing.add((params[0], params[1]))
            self._result = None
        else:
            self._result = None

    def fetchone(self):
        return self._result


class FakeRedis:
    def __init__(self, cached=None):
        self.cached = cached

    def get_data(self, key):
        return self.cached

    def set_data(self, key, value, ttl=None):
        self.cached = value


def bare_client(csv_path=FAQ_CSV):
    client = MySQLClient.__new__(MySQLClient)
    client.csv_path = csv_path
    client.conn = None
    client.cursor = None
    client.lock = threading.Lock()
    client._has_source_doc = None
    client.faq_source = None
    return client


class FaqCsvTests(unittest.TestCase):
    def test_csv_contains_categories_and_source_doc(self):
        records = bare_client()._csv_records()
        self.assertEqual(len(records), 15)
        self.assertEqual({record['source'] for record in records},
                         {'product_docs', 'platform_docs', 'release_notes'})
        self.assertTrue(all(record['source_doc'] for record in records))
        self.assertTrue(all(record['question'] and record['answer'] for record in records))

    def test_old_csv_without_source_doc_column_is_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'faq.csv'
            with path.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['资料分类', '问题', '答案'])
                writer.writerow(['product_docs', '测试问题？', '测试答案。'])
            records = bare_client(path)._csv_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['source_doc'], '')

    def test_import_is_idempotent_and_never_updates_existing_rows(self):
        client = bare_client()
        client.cursor = FakeCursor()
        first = client.insert_data()
        second = client.insert_data()
        self.assertEqual((first['inserted'], first['skipped']), (15, 0))
        self.assertEqual((second['inserted'], second['skipped']), (0, 15))
        self.assertEqual(len(client.cursor.inserted), 15)

    def test_mysql_unavailable_falls_back_to_csv(self):
        client = bare_client()
        records = client.fetch_faq()
        self.assertEqual(len(records), 15)
        self.assertEqual(client.faq_source, 'CSV')


class FaqSearchTests(unittest.TestCase):
    def make_search(self, cached=None):
        search = BM25Search.__new__(BM25Search)
        search.records = bare_client()._csv_records()
        search.revision = 1
        search.bm25_model = None
        search.faq_source = 'CSV'
        search.redis_client = FakeRedis(cached)
        return search

    def test_hit_returns_answer_and_source_document(self):
        match = self.make_search().search('抖店电子面单打印数据用哪个接口？')
        self.assertIsInstance(match, dict)
        self.assertIn('/logistics/waybillApply', match['answer'])
        self.assertEqual(match['source'], 'platform_docs')
        self.assertEqual(match['source_doc'], '02-抖店-电子面单取号与打印.txt')

    def test_old_string_cache_is_upgraded_to_match_dict(self):
        match = self.make_search('旧缓存答案').search('任意问题')
        self.assertEqual(match['answer'], '旧缓存答案')
        self.assertEqual(match['source_doc'], '')


if __name__ == '__main__':
    unittest.main()
