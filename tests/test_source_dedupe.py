"""来源去重回归测试：同一份资料按 URL/文件 ID 合并，编号与正文引用保持一致。"""

import ast
import os
import re
import sys
import types
import unittest
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1] / 'rag_qa/core/rag_pipeline.py'
if str(PIPELINE.parents[2]) not in sys.path:
    sys.path.insert(0, str(PIPELINE.parents[2]))
from rag_qa.core.doc_metadata import parse_version_selection  # noqa: E402

TREE = ast.parse(PIPELINE.read_text(encoding='utf-8'))
WANTED_ASSIGNMENTS = {'_COMPANY_MARKERS', '_ENTERPRISE_TOPICS', '_TECH_SYSTEM_TERMS',
                      '_TECH_OPERATIONS', '_DEFINITION_PATTERN', '_EXPLICIT_LOCAL'}
WANTED_FUNCTIONS = {'_is_follow_up', '_requires_local_evidence',
                    '_follow_up_requires_local_evidence', '_document_score',
                    '_version_scope_notice', '_filter_citation_markers',
                    '_source_key', '_dedupe_source_documents'}

nodes = [node for node in TREE.body
         if (isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCTIONS)
         or (isinstance(node, ast.Assign) and any(
             isinstance(target, ast.Name) and target.id in WANTED_ASSIGNMENTS
             for target in node.targets))]
namespace = {'re': re, 'parse_version_selection': parse_version_selection}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PIPELINE), 'exec'), namespace)

rag_class = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'RAGSystem')
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


def document(text, url=None, file_path=None, title='库存同步接口文档', score=0.9):
    metadata = {'filename': title, 'source': 'product_docs', 'doc_title': title, 'score': score}
    if url:
        metadata['source_url'] = url
    if file_path:
        metadata['file_path'] = file_path
    return types.SimpleNamespace(page_content=text, metadata=metadata)


class DedupeFunctionTests(unittest.TestCase):
    def test_same_url_merged_into_one_document(self):
        docs = [document('第一段：查询接口要求 V2.4.5.0。', url='https://example.com/doc/1'),
                document('第二段：回写必须以平台结果为准。', url='https://example.com/doc/1')]
        merged = namespace['_dedupe_source_documents'](docs)
        self.assertEqual(len(merged), 1)
        self.assertIn('第一段', merged[0].page_content)
        self.assertIn('第二段', merged[0].page_content)

    def test_same_title_with_different_urls_kept(self):
        docs = [document('接口 A。', url='https://example.com/doc/a'),
                document('接口 B。', url='https://example.com/doc/b')]
        self.assertEqual(len(namespace['_dedupe_source_documents'](docs)), 2)

    def test_documents_without_stable_id_kept(self):
        docs = [document('同名文档 1。'), document('同名文档 2。')]
        self.assertEqual(len(namespace['_dedupe_source_documents'](docs)), 2)

    def test_file_path_used_when_url_missing(self):
        docs = [document('第一块。', file_path='docs/stock.md'),
                document('第二块。', file_path='docs/stock.md')]
        merged = namespace['_dedupe_source_documents'](docs)
        self.assertEqual(len(merged), 1)
        self.assertIn('第二块', merged[0].page_content)


class PrepareAnswerDedupeTests(unittest.TestCase):
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
        })
        exec(compile(ast.Module(body=[prepare_method], type_ignores=[]), str(PIPELINE), 'exec'),
             cls.module)

    def test_duplicate_sources_share_one_number(self):
        system = FakeSystem([
            document('第一段：库存同步查询要求 V2.4.5.0。', url='https://example.com/doc/1'),
            document('第二段：回写必须以平台结果为准。', url='https://example.com/doc/1')])
        prompt, documents, direct = self.module['_prepare_answer'](
            system, '无法登录协作平台应该联系谁？')
        self.assertIsNone(direct)
        self.assertEqual(len(documents), 1)
        self.assertIn('[1]', prompt['context'])
        self.assertNotIn('[2]', prompt['context'])
        self.assertIn('第一段', prompt['context'])
        self.assertIn('第二段', prompt['context'])

    def test_different_docs_keep_separate_numbers(self):
        system = FakeSystem([
            document('接口 A 的说明。', url='https://example.com/doc/a'),
            document('接口 B 的说明。', url='https://example.com/doc/b')])
        prompt, documents, _ = self.module['_prepare_answer'](
            system, '无法登录协作平台应该联系谁？')
        self.assertEqual(len(documents), 2)
        self.assertIn('[1]', prompt['context'])
        self.assertIn('[2]', prompt['context'])


if __name__ == '__main__':
    unittest.main()
