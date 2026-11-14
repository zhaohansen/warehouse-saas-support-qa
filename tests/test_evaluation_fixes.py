import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'rag_qa/rag_assessment'

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

evaluator = load('rag_evaluator')
collector = load('collect_rag_results')
NAMES = ('faithfulness', 'answer_relevancy', 'context_precision', 'context_recall')
SCORES = dict.fromkeys(NAMES, 0.5)

class EvaluationFixTests(unittest.TestCase):
    def test_blank_context_is_not_ready(self):
        item = dict(question='问题', answer='答案', ground_truth='参考答案', context=['   '])
        with self.assertRaisesRegex(ValueError, '未完成采集或字段无效'):
            evaluator._prepare_evaluation_samples([item])

    def test_collector_rejects_blank_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            source.write_text(json.dumps([{'question': '问题', 'ground_truth': '  '}]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '非空'):
                collector.collect_rag_results(source, Path(directory) / 'result.json', None, 100)

    def test_corrupt_progress_is_not_used(self):
        for bad in ({'5': SCORES}, {'0': dict(SCORES, faithfulness=float('nan'))}, {'00': SCORES}):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'progress.json'
                path.write_text(json.dumps({'fingerprint':'id', 'sample_count':1, 'scores':bad}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, '无效样本索引或分数'):
                    evaluator._evaluate_in_batches([1], lambda batch: [SCORES], path, 'id')

    def test_valid_progress_resumes_remaining_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            path.write_text(json.dumps({'fingerprint':'id', 'sample_count':2, 'scores':{'0':SCORES}}), encoding='utf-8')
            batches = []
            def evaluate(batch):
                batches.append(batch)
                return [SCORES for _ in batch]
            result = evaluator._evaluate_in_batches(['已评估', '待评估'], evaluate, path, 'id')
            self.assertEqual(batches, [['待评估']])
            self.assertEqual(result, SCORES)

    def test_support_summary_separates_insufficient_and_misses(self):
        data = [
            dict(question='有依据', answer='答案 [1]', context=['资料'], ground_truth='参考'),
            dict(question='未命中但正确说明', answer='资料不足，无法确认具体金额。', context=[],
                 ground_truth='参考', expected_insufficient=True, _capture_status='completed'),
            dict(question='有无关资料但说明不足', answer='资料中未提及', context=['无关'],
                 ground_truth='参考', expected_insufficient=True),
            dict(question='有无关资料却猜测', answer='金额是 500 元', context=['无关'],
                 ground_truth='参考', expected_insufficient=True),
        ]
        summary = evaluator.summarize_support(data)
        self.assertEqual(summary['total'], 4)
        self.assertEqual(summary['answerable'], 1)
        self.assertEqual(summary['answerable_with_context'], 1)
        self.assertEqual(summary['answerable_cited'], 1)
        self.assertEqual(summary['insufficient'], 3)
        self.assertEqual(summary['insufficient_retrieval_miss'], 1)
        self.assertEqual(summary['insufficient_with_context'], 2)
        self.assertEqual(summary['insufficient_passed'], 2)
        self.assertEqual(summary['insufficient_failed'], 1)
        self.assertEqual(summary['insufficient_needs_review'], 0)
        self.assertEqual(summary['retrieval_miss'], 1)
        self.assertEqual(summary['failed']['资料不足题未说明资料不足或给出无依据结论'], ['有无关资料却猜测'])

    def test_empty_retrieval_with_correct_refusal_passes(self):
        status, _ = evaluator.assess_insufficient_answer('现有资料没有找到相关依据，无法确认节日福利金额。')
        self.assertEqual(status, 'passed')

    def test_empty_retrieval_with_invented_amount_fails(self):
        status, reason = evaluator.assess_insufficient_answer('节日福利每人 500 元。')
        self.assertEqual(status, 'failed')
        self.assertIn('未明确说明资料不足', reason)

    def test_refusal_then_guess_fails(self):
        status, reason = evaluator.assess_insufficient_answer(
            '资料不足，无法确认。不过按惯例应该是每人 500 元。')
        self.assertEqual(status, 'failed')
        self.assertIn('500 元', reason)

    def test_ambiguous_number_is_marked_for_review(self):
        status, _ = evaluator.assess_insufficient_answer('资料未提及具体金额，建议参考员工手册第 3 节。')
        self.assertEqual(status, 'needs_review')

    def test_chinese_numeral_guess_fails(self):
        status, reason = evaluator.assess_insufficient_answer(
            '资料不足，不过按惯例应该是每人五百元。')
        self.assertEqual(status, 'failed')
        self.assertIn('五百元', reason)

    def test_refusal_and_claim_in_same_sentence_fails(self):
        status, _ = evaluator.assess_insufficient_answer('资料不足但每人 500 元。')
        self.assertEqual(status, 'failed')

    def test_contrast_that_keeps_refusing_is_reviewed(self):
        status, _ = evaluator.assess_insufficient_answer('资料不足但未找到 500 元这一标准。')
        self.assertEqual(status, 'needs_review')

    def test_refusal_that_mentions_an_amount_is_reviewed(self):
        status, _ = evaluator.assess_insufficient_answer(
            '资料不足，未找到 500 元这一标准。')
        self.assertEqual(status, 'needs_review')

    def test_capture_failed_is_not_counted_as_hit_or_pass(self):
        data = [
            dict(question='未采集的资料不足题', answer='', context=[], ground_truth='参考',
                 expected_insufficient=True, _capture_status='failed'),
            dict(question='未采集的有依据题', answer='', context=[], ground_truth='参考',
                 _capture_status='failed'),
            dict(question='正常资料不足题', answer='资料不足，无法确认。', context=[],
                 ground_truth='参考', expected_insufficient=True, _capture_status='completed'),
        ]
        summary = evaluator.summarize_support(data)
        self.assertEqual(summary['capture_failed'], 2)
        self.assertEqual(summary['retrieval_miss'], 1)
        self.assertEqual(summary['insufficient_capture_failed'], 1)
        self.assertEqual(summary['insufficient_passed'], 1)
        self.assertEqual(summary['insufficient_failed'], 0)
        self.assertEqual(summary['insufficient_retrieval_miss'], 1)
        self.assertIn('采集失败或未完成', summary['failed'])
        self.assertNotIn('未采集的资料不足题', summary['failed'].get(
            '资料不足题未说明资料不足或给出无依据结论', []))

    def test_insufficient_samples_are_excluded_from_ragas(self):
        data = [
            dict(question='有依据', answer='答案 [1]', context=['资料'], ground_truth='参考'),
            dict(question='资料不足', answer='资料不足', context=['无关'], ground_truth='参考',
                 expected_insufficient=True),
        ]
        ready, without_context = evaluator._prepare_evaluation_samples(data)
        self.assertEqual([item['question'] for item in ready], ['有依据'])
        self.assertEqual(without_context, [])

    def test_collector_resume_skips_completed_and_restart_reruns(self):
        class FakeRag:
            def __init__(self):
                self.calls = []

            def generate_answer_stream(self, question, history=None, on_context=None, filters=None):
                self.calls.append(question)
                if on_context:
                    on_context([])
                yield '回答'

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            source.write_text(json.dumps([{'question': '问题一', 'ground_truth': '参考一'},
                                          {'question': '问题二', 'ground_truth': '参考二'}]), encoding='utf-8')
            output = Path(directory) / 'result.json'

            rag = FakeRag()
            collector.collect_rag_results(source, output, rag, 100)
            self.assertEqual(rag.calls, ['问题一', '问题二'])

            rag = FakeRag()
            collector.collect_rag_results(source, output, rag, 100)
            self.assertEqual(rag.calls, [])

            rag = FakeRag()
            with self.assertRaisesRegex(ValueError, '新的 --output'):
                collector.collect_rag_results(source, output, rag, 100, restart=True)

            rag = FakeRag()
            collector.collect_rag_results(source, output, rag, 100, restart=True, overwrite=True)
            self.assertEqual(rag.calls, ['问题一', '问题二'])

if __name__ == '__main__':
    unittest.main()
