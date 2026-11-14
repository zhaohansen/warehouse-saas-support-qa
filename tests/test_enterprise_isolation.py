"""企业场景隔离：旧学科资料、旧 FAQ 与旧对话不能混入企业问答。"""

import csv
import importlib.util
import json
import re
import tempfile
import unittest
from configparser import ConfigParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('conversation_store', ROOT / 'base/conversation_store.py')
conversation_store = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conversation_store)
ConversationStore = conversation_store.ConversationStore
ENTERPRISE_SOURCES = {'policies', 'products', 'operations', 'onboarding'}
LEGACY_TERMS = re.compile(r'Python|Java|MySQL|Redis|决策树|学科|课程|习题', re.IGNORECASE)


class EnterpriseIsolationTests(unittest.TestCase):
    def test_example_config_only_contains_enterprise_sources(self):
        parser = ConfigParser(interpolation=None)
        parser.read(ROOT / 'base/config.example.ini', encoding='utf-8')
        sources = {item.strip() for item in parser.get('app', 'valid_sources').split(',') if item.strip()}
        self.assertEqual(sources, ENTERPRISE_SOURCES)
        self.assertEqual(parser.get('milvus', 'collection'), 'enterprise_knowledge_qa')
        self.assertEqual(parser.get('milvus', 'database'), 'enterprise_knowledge')

    def test_history_uses_separate_enterprise_database(self):
        source = (ROOT / 'base/config.py').read_text(encoding='utf-8')
        self.assertIn('enterprise_history.sqlite3', source)
        self.assertNotIn('subject_history', source)

    def test_faq_csv_has_no_legacy_subject_entries(self):
        with (ROOT / 'mysql_qa/db/faq_knowledge_base.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row.get('问题', '').strip() and row.get('答案', '').strip()]
        self.assertEqual([row for row in rows if row['资料分类'] not in ENTERPRISE_SOURCES], [])

    def test_data_directory_contains_only_enterprise_categories(self):
        directories = {path.name for path in (ROOT / 'rag_qa/data').iterdir() if path.is_dir()}
        self.assertEqual(directories, {name + '_data' for name in ENTERPRISE_SOURCES})

    def test_evaluation_questions_are_enterprise_only(self):
        data = json.loads((ROOT / 'rag_qa/rag_assessment/enterprise_evaluate_data.json')
                          .read_text(encoding='utf-8'))
        self.assertTrue(data)
        for item in data:
            with self.subTest(question=item.get('question')):
                self.assertFalse(LEGACY_TERMS.search(item['question']))
                self.assertTrue(item['ground_truth'].strip())

    def test_classification_samples_use_enterprise_labels(self):
        data = json.loads((ROOT / 'rag_qa/classify_data/enterprise_scope.json').read_text(encoding='utf-8'))
        self.assertTrue(data)
        self.assertLessEqual({item['label'] for item in data}, {'通用知识', '企业资料'})

    def test_history_subject_filter_excludes_other_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / 'history.sqlite3')
            store.add('session', '旧学科问题', '旧答案', subject='python')
            store.add('session', '请假怎么申请？', '由直属负责人审批。', subject='policies')
            store.add('session', '协作平台怎么登录？', '联系 IT 服务台。', subject='products')

            self.assertEqual([row['question'] for row in store.history('session', subject='policies')],
                             ['请假怎么申请？'])
            self.assertEqual(store.history('session', subject='onboarding'), [])
            self.assertEqual(len(store.history('session')), 3)


if __name__ == '__main__':
    unittest.main()
