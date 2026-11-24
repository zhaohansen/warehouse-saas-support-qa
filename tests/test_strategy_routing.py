import ast
import types
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'rag_qa/core/rag_pipeline.py'
tree = ast.parse(path.read_text(encoding='utf-8'))
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'retrieve_and_merge')
namespace = {'conf': types.SimpleNamespace(RETRIEVAL_K=3, CANDIDATE_M=2),
             'logger': types.SimpleNamespace(info=lambda *a: None, debug=lambda *a: None, exception=lambda *a: None),
             '_document_identity': lambda doc: doc}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)

class Tests(unittest.TestCase):
    def service(self, direct, enhanced=None, strategy='直接检索', category='通用知识'):
        events = []
        def select(q): events.append(('select', q)); return strategy
        def search(q, **kwargs):
            events.append(('direct', kwargs.get('source_filter')))
            if isinstance(direct, Exception): raise direct
            return direct
        def rerank(q, docs): events.append(('rerank', docs)); return docs
        def augment(q, source_filter=None, filters=None): events.append(('enhanced', source_filter)); return enhanced or []
        obj = types.SimpleNamespace(
            strategy_selector=types.SimpleNamespace(select_strategy=select),
            query_classifier=types.SimpleNamespace(predict_category=lambda q: category),
            vector_store=types.SimpleNamespace(hybrid_search_with_rerank=search, rerank_documents=rerank),
            _retrieve_with_subqueries=augment, _retrieve_with_hyde=augment, _retrieve_with_backtracking=augment)
        return obj, events

    def test_select_before_direct_even_if_classifier_general(self):
        obj, events = self.service(['决策树资料'])
        self.assertEqual(namespace['retrieve_and_merge'](obj, '什么是决策树', 'python'), ['决策树资料'])
        self.assertEqual(events, [('select', '什么是决策树'), ('direct', 'python')])

    def test_enhanced_strategy_runs_after_direct_hit_and_deduplicates(self):
        obj, events = self.service(['资料1'], ['资料1', '资料2'], '子查询检索')
        self.assertEqual(namespace['retrieve_and_merge'](obj, '问题', 'ai'), ['资料1', '资料2'])
        self.assertEqual(events, [('select', '问题'), ('direct', 'ai'), ('enhanced', 'ai'), ('rerank', ['资料1', '资料2'])])

    def test_no_hits_returns_empty(self):
        obj, events = self.service([])
        self.assertEqual(namespace['retrieve_and_merge'](obj, '问题'), [])

    def test_retrieval_errors_propagate(self):
        obj, events = self.service(RuntimeError('向量库不可用'))
        with self.assertRaisesRegex(RuntimeError, '向量库不可用'):
            namespace['retrieve_and_merge'](obj, '问题')

    def test_explicit_strategy_skips_selection(self):
        obj, events = self.service(['资料1'], ['资料2'])
        namespace['retrieve_and_merge'](obj, '问题', strategy='回溯问题检索')
        self.assertEqual(events[0][0], 'direct')
        self.assertEqual(events[1][0], 'enhanced')

if __name__ == '__main__': unittest.main()
