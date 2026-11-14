"""版本适用范围回归：具体版本、范围部分重叠、边界、尾部零和网页版。"""

import ast
import os
import re
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_qa.core.doc_metadata import (filter_documents_by_version, parse_version,
                                      parse_version_selection, version_match_detail,
                                      version_matches)

PIPELINE = ROOT / 'rag_qa/core/rag_pipeline.py'
TREE = ast.parse(PIPELINE.read_text(encoding='utf-8'))
WANTED_ASSIGNMENTS = {'_COMPANY_MARKERS', '_ENTERPRISE_TOPICS', '_TECH_SYSTEM_TERMS',
                      '_TECH_OPERATIONS', '_DEFINITION_PATTERN', '_EXPLICIT_LOCAL'}
WANTED_FUNCTIONS = {'_is_follow_up', '_requires_local_evidence',
                    '_follow_up_requires_local_evidence', '_document_score',
                    '_version_scope_notice', '_source_key', '_dedupe_source_documents'}

nodes = []
for node in TREE.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANTED_ASSIGNMENTS
                                            for t in node.targets):
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCTIONS:
        nodes.append(node)

namespace = {'re': re, 'parse_version_selection': parse_version_selection}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PIPELINE), 'exec'), namespace)
rag_class = next(node for node in TREE.body
                 if isinstance(node, ast.ClassDef) and node.name == 'RAGSystem')
prepare_method = next(node for node in rag_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == '_prepare_answer')


class FakePrompt:
    def format(self, **values):
        return values


class FakeSystem:
    rag_prompt = FakePrompt()

    def __init__(self, documents):
        self.documents = documents

    def retrieve_and_merge(self, query, source_filter=None, filters=None):
        return list(self.documents)


def document(version, status='match', text='接口说明。', title='版本说明'):
    return types.SimpleNamespace(page_content=text, metadata={
        'doc_title': title, 'source': 'release_notes', 'version': version,
        'version_match': status, 'score': 0.9})


class VersionMatchTests(unittest.TestCase):
    def test_trailing_zero_is_same_version(self):
        self.assertEqual(parse_version('V2.4.5'), parse_version('V2.4.5.0'))
        self.assertEqual(version_match_detail('客户端 V2.4.5.0 及以上', '客户端 V2.4.5')['status'],
                         'match')
        self.assertTrue(version_matches('客户端 V2.4.5.0 及以上', '客户端 V2.4.5'))

    def test_exact_version_inside_and_outside_range(self):
        self.assertEqual(version_match_detail('客户端 V2.4.5.0 及以上', '客户端 V2.4.6.6')['status'],
                         'match')
        self.assertEqual(version_match_detail('客户端 V2.4.5.0 及以上', '客户端 V2.4.4')['status'],
                         'no_match')

    def test_partial_overlap_is_not_reported_as_full_match(self):
        detail = version_match_detail('客户端 V1.3.3.1 及以上', '客户端 V2.4.4 及以下')
        self.assertEqual(detail['status'], 'partial')
        self.assertIn('部分', detail['reason'])
        # 具体版本落在资料范围内时，仍然应判为完全匹配。
        inside = version_match_detail('客户端 V1.3.3.1 及以上', '客户端 V2.4.4')
        self.assertEqual(inside['status'], 'match')

    def test_boundary_values(self):
        self.assertEqual(version_match_detail('客户端 V2.4.4 及以下', '客户端 V2.4.4')['status'],
                         'match')
        self.assertEqual(version_match_detail('客户端 V2.4.4 及以下', '客户端 V2.4.4.1')['status'],
                         'no_match')
        self.assertEqual(version_match_detail('客户端 V1.3.3.1 及以上', '客户端 V1.3.3.1')['status'],
                         'match')
        self.assertEqual(version_match_detail('客户端 V1.3.3.1 及以上', '客户端 V1.3.3')['status'],
                         'no_match')

    def test_reference_and_web_scope(self):
        self.assertEqual(version_match_detail('多版本对照', '客户端 V2.4.5.0')['status'],
                         'reference')
        self.assertFalse(version_matches('多版本对照', '客户端 V2.4.5.0'))
        self.assertEqual(version_match_detail('网页版', '网页版')['status'], 'match')
        self.assertEqual(version_match_detail('客户端 V2.4.5.0 及以上', '网页版')['status'],
                         'no_match')
        self.assertEqual(version_match_detail('网页版', '客户端 V2.4.5.0')['status'], 'no_match')
        self.assertEqual(version_match_detail('平台最新（旺店通开放平台）', '客户端 V2.4.5.0')['status'],
                         'no_match')
        self.assertEqual(version_match_detail('平台最新（旺店通开放平台）', '平台最新')['status'],
                         'match')

    def test_selection_kinds(self):
        self.assertEqual(parse_version_selection('客户端 V2.4.5.0')['kind'], 'exact')
        self.assertEqual(parse_version_selection('客户端 V2.4.5.0 及以上')['kind'], 'at_least')
        self.assertEqual(parse_version_selection('客户端 V2.4.4 及以下')['kind'], 'at_most')
        self.assertEqual(parse_version_selection('网页版')['kind'], 'web')
        self.assertEqual(parse_version_selection('平台最新')['kind'], 'platform')

    def test_filter_keeps_partial_but_drops_reference(self):
        documents = [document('客户端 V2.4.4 及以下', 'match', '低版本限制'),
                     document('客户端 V1.3.3.1 及以上', 'partial', '库存回写'),
                     document('多版本对照', 'reference', '版本对照'),
                     document('客户端 V2.4.5.0 及以上', 'no_match', '原始订单')]
        kept = filter_documents_by_version(documents, '客户端 V2.4.4 及以下')
        self.assertEqual([doc.page_content for doc in kept], ['低版本限制', '库存回写'])
        self.assertEqual([doc.metadata['version_match'] for doc in kept], ['match', 'partial'])


class VersionAnswerFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = dict(namespace)
        cls.module.update({
            'conf': types.SimpleNamespace(MIN_ANSWER_SCORE=0.35, MAX_HISTORY_CHARS=12000,
                                          MAX_CONTEXT_CHARS=12000),
            'logger': types.SimpleNamespace(info=lambda *a: None, debug=lambda *a: None,
                                            exception=lambda *a: None),
            'RetrievalError': type('RetrievalError', (RuntimeError,), {}),
            'os': os,
            'is_clarification_answer': lambda text: False,
            'is_clarification_reply': lambda text: False,
            'needs_clarification': lambda query, filters=None: None,
            'parse_version_selection': parse_version_selection,
        })
        exec(compile(ast.Module(body=[prepare_method], type_ignores=[]),
                     str(PIPELINE), 'exec'), cls.module)

    def prepare(self, system, filters):
        return self.module['_prepare_answer'](
            system, '旺店通 V2.4.4 及以下能用 trade_push.php 吗？', filters=filters)

    def test_partial_source_adds_condition_notice(self):
        system = FakeSystem([
            document('客户端 V2.4.4 及以下', 'match', 'V2.4.4 及以下不满足 trade_push.php。'),
            document('客户端 V1.3.3.1 及以上', 'partial', '库存同步回写适用 V1.3.3.1 及以上。'),
        ])
        prompt, documents, direct = self.prepare(
            system, {'version': '客户端 V2.4.4 及以下'})
        self.assertIsNone(direct)
        self.assertEqual(len(documents), 2)
        self.assertIn('版本适用范围提示', prompt['context'])
        self.assertIn('只覆盖其中一部分', prompt['context'])

    def test_all_partial_for_range_asks_full_version(self):
        system = FakeSystem([document('客户端 V1.3.3.1 及以上', 'partial', '库存同步回写说明。')])
        prompt, documents, direct = self.prepare(
            system, {'version': '客户端 V2.4.4 及以下'})
        self.assertIsNone(prompt)
        self.assertEqual(documents, [])
        self.assertIn('完整版本号', direct)

    def test_exact_version_does_not_trigger_range_follow_up(self):
        system = FakeSystem([document('客户端 V1.3.3.1 及以上', 'partial', '库存同步回写说明。')])
        prompt, documents, direct = self.prepare(system, {'version': '客户端 V2.4.4'})
        self.assertIsNone(direct)
        self.assertEqual(len(documents), 1)
        self.assertIn('版本适用范围提示', prompt['context'])

    def test_no_version_filter_adds_no_notice(self):
        system = FakeSystem([document('客户端 V1.3.3.1 及以上', 'partial', '库存同步回写说明。')])
        prompt, _, direct = self.module['_prepare_answer'](
            system, '库存同步回写接口怎么用？', filters={})
        self.assertIsNone(direct)
        self.assertNotIn('版本适用范围提示', prompt['context'])


if __name__ == '__main__':
    unittest.main()
