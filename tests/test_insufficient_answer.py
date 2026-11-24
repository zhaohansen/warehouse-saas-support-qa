"""资料不足与追问的回归测试：不启动模型或向量库，只验证问答判断逻辑。"""

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
from rag_qa.core.doc_metadata import parse_version_selection
TREE = ast.parse(PIPELINE.read_text(encoding='utf-8'))

WANTED_ASSIGNMENTS = {'_COMPANY_MARKERS', '_ENTERPRISE_TOPICS', '_TECH_SYSTEM_TERMS',
                      '_TECH_OPERATIONS', '_DEFINITION_PATTERN', '_EXPLICIT_LOCAL'}
WANTED_FUNCTIONS = {'_is_follow_up', '_requires_local_evidence',
                    '_follow_up_requires_local_evidence', '_document_score',
                    '_version_scope_notice', '_filter_citation_markers',
                    '_source_key', '_dedupe_source_documents'}

nodes = []
for node in TREE.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANTED_ASSIGNMENTS
                                            for t in node.targets):
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCTIONS:
        nodes.append(node)

namespace = {'re': re, 'parse_version_selection': parse_version_selection}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PIPELINE), 'exec'), namespace)

rag_class = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'RAGSystem')
prepare_method = next(node for node in rag_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == '_prepare_answer')


class StreamOutputTests(unittest.TestCase):
    def setUp(self):
        method = next(node for node in rag_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'generate_answer_stream')
        local = dict(namespace)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(PIPELINE), 'exec'), local)
        self.generate = local['generate_answer_stream']

    def test_empty_model_output_does_not_emit_reference_notice(self):
        rag = types.SimpleNamespace(_prepare_answer=lambda *_: ('prompt', [], None),
                                    stream_llm=lambda _: iter([' ', '[99]']))
        stream = self.generate(rag, '什么是字典？')
        with self.assertRaisesRegex(RuntimeError, '模型未返回有效回答'):
            next(stream)

    def test_closing_answer_closes_model_stream(self):
        closed = []
        def model(_):
            try:
                yield '回答正文'
                yield '更多内容'
            finally:
                closed.append(True)
        rag = types.SimpleNamespace(_prepare_answer=lambda *_: ('prompt', [], None), stream_llm=model)
        stream = self.generate(rag, '什么是字典？')
        self.assertEqual(next(stream), '回答正文')
        stream.close()
        self.assertEqual(closed, [True])

    def test_general_answer_puts_reference_notice_at_end(self):
        rag = types.SimpleNamespace(_prepare_answer=lambda *_: ('prompt', [], None),
                                    stream_llm=lambda _: iter(['字典按键存取值。', '键必须可哈希。']))
        chunks = list(self.generate(rag, '什么是字典？'))
        self.assertEqual(chunks[:2], ['字典按键存取值。', '键必须可哈希。'])
        self.assertIn('仅供参考', chunks[-1])

    def test_non_decimal_numeric_text_in_brackets_is_preserved(self):
        self.assertEqual(''.join(namespace['_filter_citation_markers'](['型号 [²]，见 [1]'], 1)),
                         '型号 [²]，见 [1]')


class FakePrompt:
    def format(self, **values):
        return values


class FakeSystem:
    rag_prompt = FakePrompt()

    def __init__(self, documents):
        self.documents = documents
        self.retrieval_calls = []

    def retrieve_and_merge(self, query, source_filter=None, filters=None):
        self.retrieval_calls.append((query, source_filter))
        return list(self.documents)


def document(score=0.9, text='无法登录时联系 IT 服务台。'):
    return types.SimpleNamespace(page_content=text, metadata={
        'filename': '虚构演示-协作平台产品手册.txt', 'source': 'products', 'score': score})


class PrepareAnswerTests(unittest.TestCase):
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
        method = ast.Module(body=[prepare_method], type_ignores=[])
        exec(compile(method, str(PIPELINE), 'exec'), cls.module)

    def prepare_answer(self, system, query, **kwargs):
        return self.module['_prepare_answer'](system, query, **kwargs)

    def test_login_question_without_hits_reports_missing_material(self):
        system = FakeSystem([])
        prompt, documents, direct = self.prepare_answer(system, '无法登录协作平台应该联系谁？')
        self.assertIsNone(prompt)
        self.assertEqual(documents, [])
        self.assertIn('资料不足', direct)
        self.assertIn('不会根据常识猜测', direct)
        self.assertEqual(system.retrieval_calls, [('无法登录协作平台应该联系谁？', None)])

    def test_rephrased_login_question_still_requires_material(self):
        system = FakeSystem([])
        _, _, direct = self.prepare_answer(system, '协作平台登不上去，该找哪个部门？')
        self.assertIn('资料不足', direct)

    def test_low_relevance_recall_is_treated_as_missing(self):
        system = FakeSystem([document(score=0.02, text='员工请假需提交工作交接说明。')])
        _, documents, direct = self.prepare_answer(system, '节日福利金额是多少？')
        self.assertEqual(documents, [])
        self.assertIsNotNone(direct)

    def test_relevant_hit_builds_prompt_and_sources(self):
        system = FakeSystem([document(score=0.88)])
        prompt, documents, direct = self.prepare_answer(system, '无法登录协作平台应该联系谁？')
        self.assertIsNone(direct)
        self.assertEqual(len(documents), 1)
        self.assertIn('IT 服务台', prompt['context'])
        self.assertIn('[1]', prompt['context'])

    def test_general_knowledge_skips_retrieval_and_is_marked(self):
        system = FakeSystem([document(score=0.9)])
        prompt, documents, direct = self.prepare_answer(system, '什么是 RAG？')
        self.assertIsNone(direct)
        self.assertEqual(documents, [])
        self.assertEqual(system.retrieval_calls, [])
        self.assertIn('未检索知识库', prompt['context'])

    def test_follow_up_inherits_anchor_and_local_requirement(self):
        system = FakeSystem([])
        history = [{'question': '无法登录协作平台应该联系谁？', 'answer': '联系 IT 服务台。'}]
        _, _, direct = self.prepare_answer(system, '那应该联系谁？', history=history)
        self.assertIn('资料不足', direct)
        self.assertEqual(len(system.retrieval_calls), 1)
        retrieval_query = system.retrieval_calls[0][0]
        self.assertIn('无法登录协作平台应该联系谁？', retrieval_query)
        self.assertIn('后续问题', retrieval_query)

    def test_category_filter_is_forwarded(self):
        system = FakeSystem([document(score=0.9)])
        _, _, direct = self.prepare_answer(system, '协作平台如何登录？', source_filter='products')
        self.assertIsNone(direct)
        self.assertEqual(system.retrieval_calls[0][1], 'products')

    def test_company_specific_question_retrieves_even_without_known_keywords(self):
        system = FakeSystem([])
        _, documents, direct = self.prepare_answer(system, '云帆科技的餐卡怎么充值？')
        self.assertEqual(documents, [])
        self.assertIn('资料不足', direct)
        self.assertEqual(system.retrieval_calls, [('云帆科技的餐卡怎么充值？', None)])

    def test_general_tech_definition_skips_enterprise_retrieval(self):
        for query in ('VPN 是什么？', '什么是 VPN？', '解释一下 SSO 的原理'):
            with self.subTest(query=query):
                system = FakeSystem([document(score=0.9)])
                prompt, documents, direct = self.prepare_answer(system, query)
                self.assertIsNone(direct)
                self.assertEqual(documents, [])
                self.assertEqual(system.retrieval_calls, [])
                self.assertIn('未检索知识库', prompt['context'])

    def test_tech_operation_with_company_context_retrieves(self):
        system = FakeSystem([])
        _, _, direct = self.prepare_answer(system, '公司 VPN 怎么连接？')
        self.assertIn('资料不足', direct)
        self.assertEqual(len(system.retrieval_calls), 1)

    def test_category_filter_forces_retrieval_for_general_question(self):
        system = FakeSystem([])
        _, _, direct = self.prepare_answer(system, 'VPN 是什么？', source_filter='products')
        self.assertIn('资料不足', direct)
        self.assertEqual(system.retrieval_calls, [('VPN 是什么？', 'products')])

    def test_follow_up_after_general_question_stays_general(self):
        system = FakeSystem([document(score=0.9)])
        history = [{'question': '什么是 RAG？', 'answer': '检索增强生成。'}]
        prompt, documents, direct = self.prepare_answer(system, '那它有什么缺点？', history=history)
        self.assertIsNone(direct)
        self.assertEqual(documents, [])
        self.assertEqual(system.retrieval_calls, [])
        self.assertIn('未检索知识库', prompt['context'])

    def test_follow_up_after_company_question_stays_local(self):
        system = FakeSystem([])
        history = [{'question': '云帆科技的餐卡怎么充值？', 'answer': '资料不足。'}]
        _, _, direct = self.prepare_answer(system, '那在哪里办理？', history=history)
        self.assertIn('资料不足', direct)
        self.assertEqual(len(system.retrieval_calls), 1)

    def test_follow_up_after_tech_definition_with_action_retrieves(self):
        system = FakeSystem([])
        history = [{'question': 'VPN 是什么？', 'answer': '虚拟专用网络。'}]
        _, _, direct = self.prepare_answer(system, '那怎么申请？', history=history)
        self.assertIn('资料不足', direct)
        self.assertEqual(len(system.retrieval_calls), 1)
        self.assertIn('VPN 是什么？', system.retrieval_calls[0][0])
        self.assertIn('那怎么申请？', system.retrieval_calls[0][0])

    def test_follow_up_after_tech_definition_without_action_stays_general(self):
        system = FakeSystem([document(score=0.9)])
        history = [{'question': 'VPN 是什么？', 'answer': '虚拟专用网络。'}]
        prompt, documents, direct = self.prepare_answer(system, '那它有什么缺点？', history=history)
        self.assertIsNone(direct)
        self.assertEqual(documents, [])
        self.assertEqual(system.retrieval_calls, [])
        self.assertIn('未检索知识库', prompt['context'])

    def test_follow_up_after_tech_definition_asking_meaning_stays_general(self):
        system = FakeSystem([document(score=0.9)])
        history = [{'question': 'SSO 是什么？', 'answer': '单点登录。'}]
        prompt, documents, direct = self.prepare_answer(system, '那怎么理解它的原理？', history=history)
        self.assertIsNone(direct)
        self.assertEqual(documents, [])
        self.assertEqual(system.retrieval_calls, [])
        self.assertIn('未检索知识库', prompt['context'])

    def test_consecutive_follow_up_inherits_previous_retrieval(self):
        system = FakeSystem([])
        history = [
            {'question': 'VPN 是什么？', 'answer': '虚拟专用网络。'},
            {'question': '那怎么申请？', 'answer': '没有找到足够相关的资料，无法确认。', 'sources': []},
        ]
        _, _, direct = self.prepare_answer(system, '那需要多久？', history=history)
        self.assertIn('资料不足', direct)
        self.assertEqual(len(system.retrieval_calls), 1)

    def test_consecutive_follow_up_after_sourced_answer_retrieves(self):
        system = FakeSystem([])
        history = [
            {'question': '协作平台登录不上怎么办？', 'answer': '联系 IT 服务台。',
             'sources': [{'title': '登录故障与账号恢复'}]},
        ]
        _, _, direct = self.prepare_answer(system, '那需要准备什么？', history=history)
        self.assertIn('资料不足', direct)
        self.assertEqual(len(system.retrieval_calls), 1)


class CitationFilterTests(unittest.TestCase):
    def test_plain_text_is_emitted_as_one_block(self):
        chunks = ['这是一段普通文本，没有任何引用标记。']
        self.assertEqual(list(namespace['_filter_citation_markers'](chunks, 2)), chunks)

    def test_text_around_citation_stays_in_one_block(self):
        chunks = ['前文 [1] 后文']
        self.assertEqual(list(namespace['_filter_citation_markers'](chunks, 1)), chunks)

    def test_long_plain_text_is_not_split_per_character(self):
        chunks = ['第一段普通文本。', '第二段普通文本，包含方括号说明 [备注]。']
        result = list(namespace['_filter_citation_markers'](chunks, 2))
        self.assertEqual(''.join(result), ''.join(chunks))
        self.assertLessEqual(len(result), 4, result)

    def test_legal_citation_split_across_chunks_is_preserved(self):
        chunks = ['结论见资料 [', '2', ']。']
        self.assertEqual(''.join(namespace['_filter_citation_markers'](chunks, 2)), '结论见资料 [2]。')

    def test_out_of_range_citation_is_dropped_across_chunks(self):
        chunks = ['应联系 IT 服务台 [', '4]。', '资料 [1][2] 也支持。']
        result = ''.join(namespace['_filter_citation_markers'](chunks, 2))
        self.assertEqual(result, '应联系 IT 服务台 。资料 [1][2] 也支持。')

    def test_general_answer_has_all_citations_removed(self):
        answer = 'RAG 是一种检索增强生成技术 [1]。'
        self.assertEqual(''.join(namespace['_filter_citation_markers']([answer], 0)),
                         'RAG 是一种检索增强生成技术 。')

    def test_plain_brackets_are_preserved(self):
        text = '参见 [备注] 和 [1]。'
        self.assertEqual(''.join(namespace['_filter_citation_markers']([text], 1)), text)

    def test_unclosed_bracket_at_end_is_preserved(self):
        chunks = ['结论见 [', '1']
        self.assertEqual(''.join(namespace['_filter_citation_markers'](chunks, 2)), '结论见 [1')


class TermTests(unittest.TestCase):
    def test_internal_terms(self):
        requires = namespace['_requires_local_evidence']
        for query in ('无法登录协作平台应该联系谁？', '协作平台登不上去，该找哪个部门？',
                      '节日福利金额是多少？', '公司邮箱收不到激活邮件怎么办？',
                      '云帆科技的餐卡怎么充值？', '餐卡怎么充值？', '公司 VPN 怎么连接？',
                      '报销的审批流程顺序是什么？', '客户反馈工单从登记到关闭的步骤顺序是什么？',
                      '请假的申请材料和审批人分别是什么？'):
            with self.subTest(query=query):
                self.assertTrue(requires(query))

    def test_general_knowledge_is_not_internal(self):
        for query in ('什么是 RAG？', '解释一下决策树的原理', 'VPN 是什么？', '什么是 VPN？',
                      '什么是单点登录（SSO）？'):
            with self.subTest(query=query):
                self.assertFalse(namespace['_requires_local_evidence'](query))

    def test_source_filter_always_requires_local_evidence(self):
        self.assertTrue(namespace['_requires_local_evidence']('VPN 是什么？', 'products'))

    def test_follow_up_phrases(self):
        is_follow_up = namespace['_is_follow_up']
        for query in ('那应该联系谁？', '然后呢？', '找谁？', '那出差呢？'):
            with self.subTest(query=query):
                self.assertTrue(is_follow_up(query))
        self.assertFalse(is_follow_up('新员工如何申请办公设备？'))


if __name__ == '__main__':
    unittest.main()
