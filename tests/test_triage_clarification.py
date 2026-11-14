"""售后追问回归：区分泛化“报错”与真正可排查的错误详情。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_qa.core.triage import (is_clarification_answer, is_clarification_reply,
                                needs_clarification)


class ErrorDetailTests(unittest.TestCase):
    def test_generic_error_wording_is_not_enough(self):
        for query in ('抖店订单同步失败，有错误码但没看到具体值',
                      '抖店订单同步失败，有报错',
                      '抖店订单同步失败，提示错误码'):
            with self.subTest(query=query):
                result = needs_clarification(query)
                self.assertIsNotNone(result)
                self.assertEqual(result['missing'], ['报错原文或错误码'])
                self.assertIn('报错原文或错误码', result['question'])
                self.assertNotIn('哪个平台', result['question'])
                # 下一轮要被识别成“在回答上一轮追问”，才能继承订单问题和平台。
                self.assertTrue(is_clarification_answer(result['question']))

    def test_concrete_error_detail_skips_clarification(self):
        for query in ('抖店订单同步失败，错误码 100',
                      '抖店订单同步失败，提示“未找到符合条件的订单”',
                      '抖店订单同步失败，返回 E202'):
            with self.subTest(query=query):
                self.assertIsNone(needs_clarification(query))

    def test_unavailable_error_is_not_asked_again(self):
        for query in ('抖店订单同步失败，但是看不到错误码',
                      '抖店订单同步失败，系统没有显示报错',
                      '抖店订单同步失败，查不到错误信息'):
            with self.subTest(query=query):
                self.assertIsNone(needs_clarification(query))

    def test_knowledge_question_does_not_ask_for_site_error(self):
        for query in ('订单同步失败一般是什么原因？',
                      '订单同步失败有哪些常见原因',
                      '抖店订单同步失败怎么排查'):
            with self.subTest(query=query):
                self.assertIsNone(needs_clarification(query))

    def test_unknown_platform_and_error_asks_both(self):
        result = needs_clarification('订单一直同步不过来')
        self.assertIsNotNone(result)
        self.assertEqual(result['missing'], ['平台/店铺', '报错原文或错误码'])

    def test_follow_up_reply_continues_previous_order_question(self):
        self.assertTrue(is_clarification_reply('错误码 100'))
        self.assertTrue(is_clarification_reply('没有错误码'))
        retrieval_query = ('抖店订单同步失败，有错误码但没看到具体值\n'
                           '后续问题：错误码 100')
        self.assertIsNone(needs_clarification(retrieval_query))


if __name__ == '__main__':
    unittest.main()
