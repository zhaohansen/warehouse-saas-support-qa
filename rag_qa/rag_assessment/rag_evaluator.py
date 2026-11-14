"""RAGAS 评估：计算忠实度、答案相关性、上下文精确率与召回率。"""

import os
import re

INSUFFICIENT_MARKERS = ('资料中未提及', '资料未提及', '资料不足', '没有找到', '未提及', '无法依据',
                        '没有相关', '缺少对应资料', '无法确认', '不能确认', '未说明', '缺少依据')
# 明确的无依据推断/猜测信号，用于区分“说明资料不足”和“资料不足后又给出结论”。
_REFUSAL_MARKERS = ('未提及', '没有提到', '未说明', '无法', '不能确认', '不确定', '不猜测', '不会猜测',
                    '缺少', '没有相关', '未找到', '不详', '需确认', '待确认', '资料不足', '无法依据')
# 具体金额、天数、时限等可核验结论。中文数字也纳入，避免“五百元”绕过规则。
_NUMBER_CLAIM = re.compile(
    r'(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万亿]+)\s*'
    r'(?:元|万元|块|天|个工作日|工作日|小时|分钟|%|折|人天)'
)
# “资料不足，但/不过/其实……”后面的具体数字仍属于无依据结论。
_CONTRAST_MARKERS = ('但', '不过', '然而', '可是', '却', '其实', '实际')
# 猜测性表述不能因为同一句里出现“资料不足”就被放过。
_SPECULATION_MARKERS = ('可能', '也许', '大概', '大约', '估计', '推测', '猜测', '按惯例',
                        '通常', '一般来说', '应该是', '应为', '据我', '我认为')
# 明确否定该数字时，不把数字当成回答给出的结论，但标为待复核。
_NEGATION_MARKERS = ('不是', '并非', '不代表', '不一定', '错误', '不正确', '不能说明')


def assess_insufficient_answer(answer):
    """判断资料不足题的回答是否通过。

    返回 (状态, 原因)：
    - passed：明确说明资料不足/缺少依据，且没有给出无依据的具体结论；
    - failed：没有说明资料不足，或说明后仍给出具体金额、天数等结论；
    - needs_review：含数字表述但无法自动判断是引用、否定还是结论，交由人工复核。
    """
    answer = answer if isinstance(answer, str) else ''
    says_insufficient = any(marker in answer for marker in INSUFFICIENT_MARKERS)
    claims, unclear = [], []
    for sentence in re.split(r'[。！？\n；;，,]', answer):
        sentence = sentence.strip()
        if not sentence:
            continue
        refusal = any(marker in sentence for marker in _REFUSAL_MARKERS)
        speculation = (any(marker in sentence for marker in _SPECULATION_MARKERS)
                       and not any(marker in sentence for marker in _NEGATION_MARKERS))
        negation = any(marker in sentence for marker in _NEGATION_MARKERS)
        contrast = any(marker in sentence for marker in _CONTRAST_MARKERS)
        # “资料不足但每人 500 元”仍是在给结论；“但未找到 500 元这一标准”则是拒绝。
        tail = sentence
        if contrast:
            last = max((sentence.rfind(marker), len(marker))
                       for marker in _CONTRAST_MARKERS if marker in sentence)
            tail = sentence[last[0] + last[1]:]
        tail_refusal = any(marker in tail for marker in _REFUSAL_MARKERS)
        numbers = list(_NUMBER_CLAIM.finditer(sentence))
        if numbers:
            # 同一句里先说“资料不足”再补具体金额，仍是在给结论；猜测性表述同理。
            if negation or (refusal and not ((speculation or contrast) and not tail_refusal)):
                unclear.extend(match.group(0) for match in numbers)
            else:
                claims.extend(match.group(0) for match in numbers)
        elif re.search(r'\d', sentence) and not refusal:
            unclear.append(sentence)
    if not says_insufficient:
        return 'failed', '未明确说明资料不足'
    if claims:
        return 'failed', '给出了无依据的具体结论：' + '、'.join(claims[:3])
    if unclear:
        return 'needs_review', '含无法自动判断的数字表述：' + '、'.join(unclear[:2])[:40]
    return 'passed', '明确说明资料不足且未给出无依据的具体结论'


def _capture_incomplete(item):
    """采集失败或没有有效回答时，不能把该题算成检索命中或回答通过。"""
    answer = item.get('answer')
    return item.get('_capture_status') == 'failed' or not isinstance(answer, str) or not answer.strip()


def summarize_support(data):
    """分别统计检索是否命中、是否说明资料不足，再给出通过/失败/待复核。"""
    total = len(data)
    answerable = [i for i, item in enumerate(data) if not item.get('expected_insufficient')]
    insufficient = [i for i, item in enumerate(data) if item.get('expected_insufficient')]
    capture_failed = [i for i, item in enumerate(data) if _capture_incomplete(item)]
    captured = [i for i in range(total) if i not in capture_failed]
    retrieval_miss = [i for i in captured if not data[i].get('context')]
    answerable_with_context = [i for i in answerable if i in captured and data[i].get('context')]
    answerable_cited = [i for i in answerable_with_context if re.search(r'\[\d+\]', data[i].get('answer', ''))]
    insufficient_miss = [i for i in insufficient if i in captured and not data[i].get('context')]
    insufficient_with_context = [i for i in insufficient if i in captured and data[i].get('context')]
    insufficient_capture_failed = [i for i in insufficient if i in capture_failed]
    completed_insufficient = [i for i in insufficient if i in captured]
    verdicts = {data[i]['question']: assess_insufficient_answer(data[i].get('answer', ''))
                for i in completed_insufficient}
    passed = [i for i in completed_insufficient if verdicts[data[i]['question']][0] == 'passed']
    failed = [i for i in completed_insufficient if verdicts[data[i]['question']][0] == 'failed']
    review = [i for i in completed_insufficient if verdicts[data[i]['question']][0] == 'needs_review']
    failures = {
        '有依据题未检索到资料': [data[i]['question'] for i in answerable
                                     if i in captured and not data[i].get('context')],
        '采集失败或未完成': [data[i]['question'] for i in capture_failed],
        '资料不足题未说明资料不足或给出无依据结论': [data[i]['question'] for i in failed],
        '资料不足题待人工复核': [data[i]['question'] for i in review],
        '资料不足题采集失败或未完成': [data[i]['question'] for i in insufficient_capture_failed],
    }
    return {
        'total': total,
        'answerable': len(answerable),
        'answerable_with_context': len(answerable_with_context),
        'answerable_cited': len(answerable_cited),
        'insufficient': len(insufficient),
        'insufficient_with_context': len(insufficient_with_context),
        'insufficient_retrieval_miss': len(insufficient_miss),
        'insufficient_capture_failed': len(insufficient_capture_failed),
        'insufficient_passed': len(passed),
        'insufficient_failed': len(failed),
        'insufficient_needs_review': len(review),
        'insufficient_verdicts': {question: {'status': status, 'reason': reason}
                                  for question, (status, reason) in verdicts.items()},
        'retrieval_miss': len(retrieval_miss),
        'capture_failed': len(capture_failed),
        'failed': {name: questions for name, questions in failures.items() if questions},
    }


def _prepare_evaluation_samples(data):
    """区分未采集样本与确实没有检索资料的样本。"""
    if not isinstance(data, list) or not data:
        raise ValueError('评估数据必须是非空列表。')
    pending, without_context, ready = [], [], []
    for index, item in enumerate(data):
        if (not isinstance(item, dict) or not isinstance(item.get('question'), str)
                or not item['question'].strip() or not isinstance(item.get('ground_truth'), str)
                or not item['ground_truth'].strip() or not isinstance(item.get('answer'), str)
                or not item['answer'].strip() or not isinstance(item.get('context'), list)
                or any(not isinstance(part, str) or not part.strip() for part in item['context'])
                or item.get('_capture_status') == 'failed'):
            pending.append(index)
        elif item.get('expected_insufficient'):
            # 资料不足题不参与 RAGAS；由 summarize_support 单独统计通过/失败/待复核。
            continue
        elif not item['context']:
            if item.get('_capture_status') == 'completed':
                without_context.append((index, item['question']))
            else:
                pending.append(index)
        else:
            ready.append(item)
    if pending:
        raise ValueError(f'有 {len(pending)} 条样本未完成采集或字段无效（从 0 开始的索引：{pending[:10]}）。请先检查并重新采集。')
    if not ready:
        raise ValueError('没有带检索资料的样本，无法计算基于上下文的 RAGAS 指标。')
    return ready, without_context


def _evaluate_in_batches(data, evaluate_batch, progress_path, fingerprint, batch_size=5):
    """逐批评估并保存分数；再次运行时跳过已完成的样本。"""
    import json
    import math
    import os
    import tempfile
    from pathlib import Path

    if not data or batch_size <= 0:
        raise ValueError('评估样本不能为空，批次大小必须大于 0。')
    names = ('faithfulness', 'answer_relevancy', 'context_precision', 'context_recall')
    progress_path = Path(progress_path)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        if not isinstance(progress, dict):
            raise ValueError('评估进度文件格式无效，请检查或指定新的进度文件。')
        if progress.get('fingerprint') != fingerprint or progress.get('sample_count') != len(data):
            raise ValueError('评估数据或模型配置已变化，请指定新的进度文件，避免混用结果。')
        scores = progress.get('scores')
        if not isinstance(scores, dict):
            raise ValueError('评估进度文件的分数格式无效，请检查或指定新的进度文件。')
        for key, row in scores.items():
            if (not isinstance(key, str) or not key.isascii() or not key.isdigit()
                    or str(int(key)) != key or int(key) >= len(data)
                    or not isinstance(row, dict)
                    or any(not isinstance(row.get(name), (int, float))
                           or isinstance(row[name], bool) or not math.isfinite(row[name]) for name in names)):
                raise ValueError('评估进度文件包含无效样本索引或分数，请检查或指定新的进度文件。')
    else:
        progress = {'fingerprint': fingerprint, 'sample_count': len(data), 'scores': {}}

    remaining = [i for i in range(len(data)) if str(i) not in progress['scores']]
    print(f"已保存 {len(data) - len(remaining)}/{len(data)} 条评估结果，本次还需处理 {len(remaining)} 条。")
    for offset in range(0, len(remaining), batch_size):
        indexes = remaining[offset:offset + batch_size]
        print(f'正在评估第 {offset + 1}～{offset + len(indexes)} 条待处理样本，共 {len(remaining)} 条。', flush=True)
        rows = evaluate_batch([data[i] for i in indexes])
        if len(rows) != len(indexes):
            raise RuntimeError('评估结果数量与输入不一致，当前批次未保存。')
        batch_scores = {}
        for index, row in zip(indexes, rows):
            scores = {name: float(row[name]) for name in names}
            if not all(math.isfinite(score) for score in scores.values()):
                raise RuntimeError(f'样本 {index} 存在无效分数，当前批次未保存。')
            batch_scores[str(index)] = scores
        progress['scores'].update(batch_scores)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=progress_path.parent,
                                             prefix='.ragas-', suffix='.tmp', delete=False) as f:
                temp_path = Path(f.name)
                json.dump(progress, f, ensure_ascii=False, indent=2)
                f.write('\n')
            os.replace(temp_path, progress_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        print(f"本批完成，已保存 {len(progress['scores'])}/{len(data)} 条。", flush=True)

    averages = {name: sum(row[name] for row in progress['scores'].values()) / len(data) for name in names}
    print('RAGAS 评估结果（仅包含有检索资料的样本）：', averages)
    print('逐题分数和断点进度：', progress_path)
    return averages


def evaluate_knowledge_base(data_path=None, progress_path=None):
    from ragas import evaluate
    from ragas.run_config import RunConfig
    import hashlib
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    from datasets import Dataset
    import json

    from langchain_ollama import ChatOllama, OllamaEmbeddings

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from pathlib import Path
    data_path = Path(data_path) if data_path else Path(__file__).with_name('enterprise_evaluate_results.json')
    progress_path = Path(progress_path) if progress_path else data_path.with_name('enterprise_ragas_scores_progress.json')
    if data_path.resolve() == progress_path.resolve():
        raise ValueError('评估进度文件不能覆盖原始评估数据。')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # answer 是系统回答，context 是检索资料，ground_truth 是参考答案。
    print(f'加载的评估样本数量: {len(data)}')

    all_data = data
    data, without_context = _prepare_evaluation_samples(data)
    support = summarize_support(all_data)
    print(f'样本分层：总数 {support["total"]}；有依据题 {support["answerable"]}'
          f'（有检索资料 {support["answerable_with_context"]}，回答含引用编号 {support["answerable_cited"]}）；'
          f'资料不足题 {support["insufficient"]}'
          f'（检索未命中 {support["insufficient_retrieval_miss"]}，有资料 {support["insufficient_with_context"]}，'
          f'通过 {support["insufficient_passed"]}，失败 {support["insufficient_failed"]}，'
          f'待人工复核 {support["insufficient_needs_review"]}，'
          f'采集失败 {support["insufficient_capture_failed"]}）；'
          f'全部检索未命中 {support["retrieval_miss"]}，采集失败/未完成 {support["capture_failed"]}')
    print('统计口径：RAGAS 只计算“有依据且检索到资料”的样本，资料不足题和检索未命中都不进入均值；'
          '资料不足题单独统计：回答明确说明缺少依据、且没有给出无依据的具体金额/天数/时限才算通过；'
          '说明资料不足后又给出具体结论判为失败；含数字但无法自动判断是引用还是结论的标为待人工复核。'
          '有依据题以检索资料、引用编号和人工核对参考答案为准。')
    if support['failed']:
        for reason, questions in support['failed'].items():
            print(f'  失败/待复核案例（{reason}）：')
            for question in questions:
                print(f'    - {question}')

    if without_context:
        print(f'其中 {len(without_context)} 题真实未检索到资料，本次四项评分暂不包含这些题；请单独分析，不能视为正确。')
        for index, question in without_context:
            print(f'  索引 {index}：{question}')
    print(f'本次实际评估 {len(data)} 条有检索资料的样本。')

    class RagasChatOllama(ChatOllama):
        """适配 RAGAS 传入的温度与多回答参数。"""

        def _chat_params(self, messages, stop=None, **kwargs):
            temperature = kwargs.pop('temperature', None)
            kwargs.pop('n', None)
            params = super()._chat_params(messages, stop, **kwargs)
            if temperature is not None:
                options = dict(params.get('options') or {})
                options['temperature'] = temperature
                params['options'] = options
            return params

    class OllamaRagasWrapper(LangchainLLMWrapper):
        """将多回答请求拆成独立请求。"""

        def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            if temperature is None:
                temperature = self.get_temperature(n=n)
            result = self.langchain_llm.generate_prompt(
                prompts=[prompt] * n, temperature=temperature, stop=stop, callbacks=callbacks)
            result.generations = [[group[0] for group in result.generations]]
            return result

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            if temperature is None:
                temperature = self.get_temperature(n=n)
            result = await self.langchain_llm.agenerate_prompt(
                prompts=[prompt] * n, temperature=temperature, stop=stop, callbacks=callbacks)
            result.generations = [[group[0] for group in result.generations]]
            return result

    ollama_model = 'qwen2.5:7b'
    ollama_url = 'http://localhost:11434'
    llm = OllamaRagasWrapper(
        RagasChatOllama(model=ollama_model, base_url=ollama_url, temperature=0)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model='bge-m3', base_url='http://localhost:11434')
    )
    fingerprint_input = {
        'data': [{key: item[key] for key in ('question', 'answer', 'context', 'ground_truth')} for item in data],
        'model': 'ollama:' + ollama_model,
        'base_url': ollama_url,
        'embeddings': 'ollama:bge-m3',
        'metrics': ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'],
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_input, ensure_ascii=False,
                                         sort_keys=True).encode('utf-8')).hexdigest()

    def evaluate_batch(samples):
        batch = Dataset.from_dict({
            'question': [item['question'] for item in samples],
            'answer': [item['answer'] for item in samples],
            'contexts': [item['context'] for item in samples],
            'ground_truth': [item['ground_truth'] for item in samples],
        })
        result = evaluate(
            dataset=batch,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(timeout=180, max_retries=2, max_wait=10,
                                 max_workers=int(os.getenv('RAGAS_MAX_WORKERS', '2') or 2)),
            raise_exceptions=True,
        )
        return result.to_pandas().to_dict('records')

    batch_size = int(os.getenv('RAGAS_BATCH_SIZE', '5') or 5)
    return _evaluate_in_batches(data, evaluate_batch, progress_path, fingerprint, batch_size=batch_size)



if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='对采集结果计算 RAGAS 指标，并单独统计资料不足题')
    parser.add_argument('--data', type=Path, help='采集结果 JSON；默认 enterprise_evaluate_results.json')
    parser.add_argument('--progress', type=Path,
                        help='评分断点文件；重跑一轮时请指定新文件，避免覆盖历史进度')
    args = parser.parse_args()
    evaluate_knowledge_base(args.data, args.progress)
