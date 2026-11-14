"""BERT 参与分流的回归测试：只抽取问答判断逻辑，不加载模型或向量库。"""

import ast
import json
import os
import re
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / 'rag_qa/core/rag_pipeline.py'
TREE = ast.parse(PIPELINE.read_text(encoding='utf-8'))

WANTED_ASSIGNMENTS = {'_COMPANY_MARKERS', '_ENTERPRISE_TOPICS', '_TECH_SYSTEM_TERMS',
                      '_TECH_OPERATIONS', '_DEFINITION_PATTERN', '_EXPLICIT_LOCAL'}
WANTED_FUNCTIONS = {'_is_follow_up', '_requires_local_evidence',
                    '_follow_up_requires_local_evidence', '_document_score',
                    '_filter_citation_markers'}

nodes = []
for node in TREE.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in WANTED_ASSIGNMENTS
                                            for t in node.targets):
        nodes.append(node)
    elif isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCTIONS:
        nodes.append(node)

namespace = {'re': re}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PIPELINE), 'exec'), namespace)

rag_class = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'RAGSystem')
prepare_method = next(node for node in rag_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == '_prepare_answer')


class FakePrompt:
    def format(self, **values):
        return values


class FakeSystem:
    rag_prompt = FakePrompt()

    def __init__(self, documents, classifier=None):
        self.documents = documents
        self.retrieval_calls = []
        if classifier is not None:
            self.query_classifier = classifier

    def retrieve_and_merge(self, query, source_filter=None):
        self.retrieval_calls.append((query, source_filter))
        return list(self.documents)


def document(score=0.9, text='新员工可向行政部门申领办公设备。'):
    return types.SimpleNamespace(page_content=text, metadata={
        'filename': '虚构演示-办公设备申领与异常处理.txt', 'source': 'onboarding', 'score': score})


class FakeClassifier:
    def __init__(self, label='企业资料', confidence=0.95, available=True, mapping=None):
        self.label = label
        self.confidence = confidence
        self.available = available
        self.mapping = mapping or {}
        self.calls = []

    def predict_with_confidence(self, query):
        self.calls.append(query)
        if query in self.mapping:
            value = self.mapping[query]
            return dict(available=True, label=value['label'], confidence=value['confidence'],
                        enterprise_probability=value['confidence'] if value['label'] == '企业资料' else 1 - value['confidence'])
        return dict(available=self.available, label=self.label if self.available else None,
                    confidence=self.confidence,
                    enterprise_probability=self.confidence if self.label == '企业资料' else 1 - self.confidence)


class ExplodingClassifier:
    def predict_with_confidence(self, query):
        raise AssertionError('规则已判定需要本地资料，不应再调用 BERT')


class RaisingClassifier:
    def predict_with_confidence(self, query):
        raise RuntimeError('模型推理失败')


class PrepareAnswerRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = dict(namespace)
        cls.module.update({
            'conf': types.SimpleNamespace(MIN_ANSWER_SCORE=0.35, MAX_HISTORY_CHARS=12000,
                                          MAX_CONTEXT_CHARS=12000),
            'logger': types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None,
                                            warning=lambda *a, **k: None, exception=lambda *a, **k: None),
            'RetrievalError': type('RetrievalError', (RuntimeError,), {}),
            'os': os,
        })
        method = ast.Module(body=[prepare_method], type_ignores=[])
        exec(compile(method, str(PIPELINE), 'exec'), cls.module)

    def prepare_answer(self, system, query, **kwargs):
        return self.module['_prepare_answer'](system, query, **kwargs)

    def test_high_confidence_enterprise_triggers_retrieval(self):
        classifier = FakeClassifier(label='企业资料', confidence=0.95)
        system = FakeSystem([document()], classifier)
        prompt, documents, direct = self.prepare_answer(system, '量子咖啡机怎么申请使用？')
        self.assertIsNone(direct)
        self.assertTrue(documents)
        self.assertEqual(len(system.retrieval_calls), 1)
        self.assertEqual(classifier.calls, ['量子咖啡机怎么申请使用？'])

    def test_definition_question_is_not_overridden_by_model(self):
        classifier = FakeClassifier(label='企业资料', confidence=0.99)
        system = FakeSystem([document()], classifier)
        prompt, documents, direct = self.prepare_answer(system, 'VPN 是什么？')
        self.assertIsNotNone(prompt)
        self.assertEqual(documents, [])
        self.assertEqual(system.retrieval_calls, [])

    def test_low_confidence_keeps_rule_result(self):
        classifier = FakeClassifier(label='企业资料', confidence=0.5)
        system = FakeSystem([document()], classifier)
        prompt, documents, direct = self.prepare_answer(system, '量子咖啡机怎么申请使用？')
        self.assertIsNotNone(prompt)
        self.assertEqual(documents, [])
        self.assertEqual(system.retrieval_calls, [])

    def test_unavailable_model_keeps_rule_result(self):
        classifier = FakeClassifier(available=False, label=None, confidence=0.0)
        system = FakeSystem([document()], classifier)
        prompt, documents, direct = self.prepare_answer(system, '量子咖啡机怎么申请使用？')
        self.assertIsNotNone(prompt)
        self.assertEqual(system.retrieval_calls, [])

    def test_rules_win_and_model_is_not_called(self):
        system = FakeSystem([document()], ExplodingClassifier())
        prompt, documents, direct = self.prepare_answer(system, '我们公司的 VPN 怎么申请？')
        self.assertIsNone(direct)
        self.assertTrue(documents)
        self.assertEqual(len(system.retrieval_calls), 1)

    def test_classifier_exception_falls_back_to_rules(self):
        system = FakeSystem([document()], RaisingClassifier())
        prompt, documents, direct = self.prepare_answer(system, '量子咖啡机怎么申请使用？')
        self.assertIsNotNone(prompt)
        self.assertEqual(system.retrieval_calls, [])

    def test_follow_up_uses_anchor_classification(self):
        classifier = FakeClassifier(mapping={
            '笔记本找谁领？': dict(label='企业资料', confidence=0.96),
            '那要提前几天？': dict(label='通用知识', confidence=0.9),
        })
        system = FakeSystem([document()], classifier)
        history = [dict(question='笔记本找谁领？', answer='向行政部门申领。')]
        prompt, documents, direct = self.prepare_answer(system, '那要提前几天？', history=history)
        self.assertIsNone(direct)
        self.assertTrue(documents)
        self.assertEqual(len(system.retrieval_calls), 1)
        self.assertIn('笔记本找谁领？', system.retrieval_calls[0][0])
        self.assertIn('那要提前几天？', classifier.calls)


class ClassifierDatasetTests(unittest.TestCase):
    def test_dataset_is_grouped_by_family_and_avoids_eval_questions(self):
        path = ROOT / 'rag_qa/classify_data/enterprise_scope_v2.jsonl'
        records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.assertEqual(len(records), 240)
        self.assertEqual(sum(1 for record in records if record['split'] == 'test'), 44)
        family_split = {}
        for record in records:
            self.assertIn(record['label'], {'企业资料', '通用知识'})
            family_split.setdefault(record['family'], set()).add(record['split'])
        self.assertTrue(all(len(splits) == 1 for splits in family_split.values()))

        def normalize(text):
            return re.sub(r'[\s，。！？、,.!?：:；;“”"\'（）()\[\]]+', '', text)

        eval_data = json.loads((ROOT / 'rag_qa/rag_assessment/enterprise_evaluate_data.json')
                               .read_text(encoding='utf-8'))
        eval_questions = {normalize(item.get('question') or item.get('query') or '') for item in eval_data}
        self.assertFalse(any(normalize(record['query']) in eval_questions for record in records))


if __name__ == '__main__':
    unittest.main()
