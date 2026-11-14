"""运行项目的 RAG 流程，采集真实回答和检索资料，生成评估结果文件。"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = Path(__file__).with_name('enterprise_evaluate_data.json')
DEFAULT_OUTPUT = Path(__file__).with_name('enterprise_evaluate_results.json')


def _save_json(path, data):
    """先写临时文件，再替换结果文件，避免中断时损坏已保存的结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _actual_context(documents, max_chars):
    """与问答流程一致，记录实际放入提示词的文档内容。"""
    snippets = []
    remaining = max_chars
    for doc in documents:
        if remaining <= 0:
            break
        snippet = doc.page_content[:remaining]
        snippets.append(snippet)
        remaining -= len(snippet)
    return snippets


def collect_rag_results(input_path, output_path, rag_system, max_context_chars, limit=None,
                        restart=False, overwrite=False):
    """逐题运行 RAG。

    - 默认是“继续上次采集”：结果文件中已完成（_capture_status=completed）的题目会跳过；
    - restart=True 是“重新跑一轮”：所有题目重新执行；若结果文件已存在，必须同时
      overwrite=True，或改用新的 --output，避免无意覆盖历史结果。
    """
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError('输出文件不能与原始题库相同，避免覆盖原文件。')
    if limit is not None and limit < 1:
        raise ValueError('--limit 必须大于 0。')

    source = json.loads(input_path.read_text(encoding='utf-8'))
    if not isinstance(source, list) or any(
        not isinstance(item, dict) or not isinstance(item.get('question'), str)
        or not item['question'].strip() or not isinstance(item.get('ground_truth'), str)
        or not item['ground_truth'].strip()
        for item in source
    ):
        raise ValueError('题库必须是列表，每条数据需要非空的 question 和 ground_truth 字符串。')

    def fresh_items():
        """从题库重建记录，避免源文件里遗留的状态影响新一轮采集。"""
        items = []
        for item in source:
            record = dict(item, answer='', context=[])
            record.pop('_capture_status', None)
            record.pop('_capture_error', None)
            items.append(record)
        return items

    if output_path.exists():
        data = json.loads(output_path.read_text(encoding='utf-8'))
        if (not isinstance(data, list) or len(data) != len(source)
                or any(not isinstance(a, dict) or a.get('question') != b.get('question')
                       or a.get('ground_truth') != b.get('ground_truth')
                       for a, b in zip(data, source))):
            raise ValueError('结果文件与当前题库不一致，请更换输出文件，避免混用旧结果。')
        if restart:
            if not overwrite:
                raise ValueError('重新跑一轮会覆盖已有结果；请改用新的 --output，或显式加 --overwrite。')
            data = fresh_items()
            print(f'重新采集全部 {len(data)} 题，原结果文件将被覆盖：{output_path}', flush=True)
        else:
            completed = sum(item.get('_capture_status') == 'completed' for item in data)
            print(f'继续上次采集：已有 {completed}/{len(data)} 题完成，将跳过这些题目。', flush=True)
    else:
        data = fresh_items()
        print(f'新建采集结果：{output_path}', flush=True)

    attempted = 0
    for index, item in enumerate(data):
        if item.get('_capture_status') == 'completed':
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        documents = []
        print(f'[{index + 1}/{len(data)}] 正在处理：{item["question"]}', flush=True)
        try:
            answer = ''.join(rag_system.generate_answer_stream(
                item['question'], history=[], on_context=documents.extend))
            if not answer.strip():
                raise RuntimeError('系统未生成有效回答')
            item['answer'] = answer
            item['context'] = _actual_context(documents, max_context_chars)
            item['_capture_status'] = 'completed'
            item.pop('_capture_error', None)
            print(f'  完成：检索到 {len(item["context"])} 段资料', flush=True)
        except Exception as exc:
            item['answer'] = ''
            item['context'] = []
            item['_capture_status'] = 'failed'
            item['_capture_error'] = str(exc)
            print(f'  失败：{exc}', file=sys.stderr, flush=True)
        _save_json(output_path, data)

    completed = sum(item.get('_capture_status') == 'completed' for item in data)
    empty_context = sum(item.get('_capture_status') == 'completed' and not item.get('context') for item in data)
    insufficient = [item for item in data if item.get('expected_insufficient')]
    insufficient_empty = sum(item.get('_capture_status') == 'completed' and not item.get('context')
                             for item in insufficient)
    print(f'已完成 {completed}/{len(data)} 题，其中 {empty_context} 题没有检索资料。')
    print(f'资料不足题 {len(insufficient)} 题，其中 {insufficient_empty} 题检索为空；'
          f'未标记资料不足但有依据题请检查采集结果。')
    print(f'结果文件：{output_path}')
    return data


def main():
    parser = argparse.ArgumentParser(description='采集 RAG 的真实回答和检索资料')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='原始题库 JSON 路径')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='结果 JSON 路径；重新跑一轮时请指定新文件')
    parser.add_argument('--limit', type=int, help='本次最多处理的题数，建议先填 2')
    parser.add_argument('--restart', action='store_true',
                        help='重新跑一轮：忽略结果文件中已完成的记录，全部重做')
    parser.add_argument('--overwrite', action='store_true',
                        help='与 --restart 配合，允许覆盖已存在的结果文件')
    args = parser.parse_args()
    if args.overwrite and not args.restart:
        parser.error('--overwrite 只能与 --restart 一起使用')
    if args.restart and args.output.exists() and not args.overwrite:
        parser.error('目标结果文件已存在；请改用新的 --output，或显式加 --overwrite 覆盖。')

    from cli import IntegratedQASystem
    from base.config import config

    qa = IntegratedQASystem()
    try:
        if qa.rag_system is None:
            raise RuntimeError('RAG 未就绪，请检查本地模型与 Milvus 服务。')
        if not config.LLM_API_KEY:
            raise RuntimeError('未配置模型 API 密钥。')
        collect_rag_results(args.input, args.output, qa.rag_system, config.MAX_CONTEXT_CHARS,
                            args.limit, restart=args.restart, overwrite=args.overwrite)
    finally:
        qa.close()


if __name__ == '__main__':
    main()
