"""FAQ 通道的命令行入口，用于单独调试问题匹配。"""

import time

from mysql_qa.retrieval.bm25search import BM25Search
from base import logger


class FAQSystem:
    def __init__(self):
        self.bm25 = BM25Search()
        self.logger = logger

    def query(self, query):
        try:
            answer = self.bm25.search(query)
            if not answer:
                self.logger.info("FAQ 未找到匹配答案")
            return answer
        except Exception as e:
            self.logger.error(f"查询处理失败：{e}")
            return None


def main():
    system = FAQSystem()
    print("欢迎使用 FAQ 匹配调试入口")
    while True:
        query = input("请输入问题：")
        if query == "exit":
            print("系统关闭,感谢使用.")
            break
        else:
            started_at = time.perf_counter()
            answer = system.query(query)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            print("问题：", query)
            print("答案：", answer)
            print(f"查询耗时：{elapsed_ms:.2f} ms")

if __name__ == "__main__":
    main()
