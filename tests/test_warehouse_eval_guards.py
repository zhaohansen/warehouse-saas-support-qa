"""评估规则回归：文档命中、引用合法性和关键结论分开判断。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_qa.rag_assessment import warehouse_runner as runner

CASE = {
    'expected_docs': ['客户端 V2.4.4 及以下版本的接口限制（整理）'],
    'expected_keywords': ['V2.4.5.0', '不能', '升级'],
    'filters': {'version': '客户端 V2.4.4 及以下'},
    'expected_conclusion': {
        'required_terms': ['V2.4.5.0'],
        'required_any': ['不能', '不支持', '不满足'],
        'forbidden_patterns': [
            'V2\\.4\\.4[^。；\\n]{0,15}(?<!不)(?:可以|能|支持)[^。；\\n]{0,15}trade_push'],
    },
}
SOURCES = [{'title': '客户端 V2.4.4 及以下版本的接口限制（整理）',
            'version': '客户端 V2.4.4 及以下'}]


def record(answer, sources=SOURCES):
    return {'answer': answer, 'sources': sources,
            'citation_count': answer.count('[1]') + answer.count('[9]')}


class EvaluationGuardTests(unittest.TestCase):
    def test_empty_source_title_is_not_a_doc_hit(self):
        self.assertFalse(runner._doc_hit([{'title': ''}], ['错误码说明']))
        self.assertFalse(runner._doc_hit([{'title': '   '}], ['错误码说明']))

    def test_out_of_range_citation_fails_even_with_matching_title(self):
        answer = 'V2.4.4 可以调用 trade_push.php [9]'
        checks = runner._evaluate_record(record(answer), CASE)
        self.assertTrue(checks['doc_hit'])
        self.assertFalse(checks['citation_pass'])
        self.assertEqual(checks['citation_invalid'], [9])
        self.assertFalse(checks['passed'])
        self.assertTrue(any('引用编号越界' in reason for reason in checks['failure_reasons']))

    def test_opposite_conclusion_with_same_keywords_fails(self):
        answer = 'V2.4.4 可以调用 trade_push.php，不需要升级到 V2.4.5.0 [1]'
        checks = runner._evaluate_record(record(answer), CASE)
        self.assertGreaterEqual(checks['keyword_ratio'], 0.5)
        self.assertEqual(checks['conclusion_status'], 'failed')
        self.assertFalse(checks['passed'])

    def test_correct_conclusion_passes(self):
        answer = '不能。trade_push.php 要求客户端 V2.4.5.0 及以上，V2.4.4 及以下不适用 [1]'
        checks = runner._evaluate_record(record(answer), CASE)
        self.assertTrue(checks['citation_pass'])
        self.assertEqual(checks['conclusion_status'], 'passed')
        self.assertTrue(checks['passed'], checks['failure_reasons'])

    def test_partial_version_only_is_not_a_version_pass(self):
        checks = runner._version_check(
            [{'title': '库存回写', 'version': '客户端 V1.3.3.1 及以上'}],
            '客户端 V2.4.4 及以下')
        self.assertFalse(checks['pass'])
        self.assertEqual(len(checks['partial']), 1)

    def test_reference_document_does_not_count_as_applicable(self):
        checks = runner._version_check(
            [{'title': '版本对照', 'version': '多版本对照'}], '客户端 V2.4.4 及以下')
        self.assertFalse(checks['pass'])

    def test_manual_review_is_not_scored_as_failure(self):
        case = {'expected_docs': CASE['expected_docs'],
                'expected_keywords': CASE['expected_keywords'],
                'expected_conclusion': {'manual_review': True}}
        answer = 'V2.4.5.0 及以上可用，V2.4.4 及以下需要升级 [1]'
        checks = runner._evaluate_record(record(answer), case)
        self.assertEqual(checks['conclusion_status'], 'manual_review')
        self.assertTrue(checks['passed'], checks['failure_reasons'])


if __name__ == '__main__':
    unittest.main()
