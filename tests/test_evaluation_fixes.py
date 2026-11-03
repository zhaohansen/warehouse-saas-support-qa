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

if __name__ == '__main__':
    unittest.main()
