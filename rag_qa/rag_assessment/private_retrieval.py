"""使用虚构资料做小样本检索对照。运行前先导入演示资料。"""

import json
from statistics import mean
from time import perf_counter


CASES = (
    ('WH-ORD-17 重试订单同步之前要核对哪些配置？', '仓配实施手册·订单同步（虚构演示）'),
    ('目标仓库编号没填，怎么让暂停的同步任务恢复？', '仓配实施手册·订单同步（虚构演示）'),
    ('库存盘点数字不对，可以让原盘点员自己复查吗？', '仓配作业规范·库存复核（虚构演示）'),
    ('复盘后谁批准修改库存余额？', '仓配作业规范·库存复核（虚构演示）'),
    ('WH-LBL-09 什么情况下允许补打一张标签？', '仓配售后手册·面单补打（虚构演示）'),
    ('包裹已经揽收了还能重新打印原来的电子面单吗？', '仓配售后手册·面单补打（虚构演示）'),
    ('WH-ORD-18 回执超时，要不要把订单重新推一次？', '仓配对接手册·订单回执（虚构演示）'),
    ('仓库已经收到了订单，平台还在等确认消息，该补发什么？', '仓配对接手册·订单回执（虚构演示）'),
    ('仓间调拨未签收，可以用盘点调整单冲销差异吗？', '仓配作业规范·在途库存（虚构演示）'),
    ('货从一个仓发出了，另一个仓还没收到，应该按什么单号核对？', '仓配作业规范·在途库存（虚构演示）'),
    ('WH-LBL-10 地址填错，未揽收时什么时候能重新取号？', '仓配售后手册·面单作废（虚构演示）'),
    ('收件地址写错了，能直接拿原号码打印新地址吗？', '仓配售后手册·面单作废（虚构演示）'),
)
MODES = ('dense', 'hybrid', 'hybrid_rerank')


def evaluate(store):
    # 预热三种路径，并轮转运行顺序，降低首次运行和固定顺序带来的干扰。
    for mode in MODES:
        store.hybrid_search_with_rerank(CASES[0][0], k=3, source_filter='internal_docs',
                                        retrieval_mode=mode)
    records = {mode: [] for mode in MODES}
    for index, (query, expected) in enumerate(CASES):
        offset = index % len(MODES)
        for mode in MODES[offset:] + MODES[:offset]:
            start = perf_counter()
            results = store.hybrid_search_with_rerank(query, k=3, source_filter='internal_docs',
                                                      retrieval_mode=mode)
            elapsed_ms = round((perf_counter() - start) * 1000, 1)
            titles = list(dict.fromkeys(doc.metadata.get('doc_title') for doc in results))
            rank = titles.index(expected) + 1 if expected in titles else None
            records[mode].append({'query': query, 'expected': expected, 'rank': rank,
                                  'retrieved_titles': titles, 'retrieval_ms': elapsed_ms})

    summaries = {}
    for mode, rows in records.items():
        summaries[mode] = {
            'hit_at_1': sum(row['rank'] == 1 for row in rows),
            'hit_at_3': sum(row['rank'] is not None and row['rank'] <= 3 for row in rows),
            'mean_retrieval_ms': round(mean(row['retrieval_ms'] for row in rows), 1),
        }
    return {'backend': store.describe_status(), 'cases': len(CASES),
            'metrics': summaries, 'details': records}


def main():
    from base.config import config
    from rag_qa.core.vector_store import VectorStore

    if not config.PRIVATE_MODE:
        raise SystemExit('请先设置 QA_PRIVATE_MODE=1 并导入虚构演示资料')
    store = VectorStore()
    try:
        print(json.dumps(evaluate(store), ensure_ascii=False, indent=2))
    finally:
        store.client.close()


if __name__ == '__main__':
    main()
