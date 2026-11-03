"""RAGAS 评估：计算忠实度、答案相关性、上下文精确率与召回率。"""


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
    data_path = Path(data_path) if data_path else Path(__file__).with_name('rag_evaluate_results.json')
    progress_path = Path(progress_path) if progress_path else data_path.with_name('ragas_scores_progress.json')
    if data_path.resolve() == progress_path.resolve():
        raise ValueError('评估进度文件不能覆盖原始评估数据。')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # answer 是系统回答，context 是检索资料，ground_truth 是参考答案。
    print(f'加载的评估样本数量: {len(data)}')

    data, without_context = _prepare_evaluation_samples(data)
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
            run_config=RunConfig(timeout=180, max_retries=2, max_wait=10, max_workers=2),
            raise_exceptions=True,
        )
        return result.to_pandas().to_dict('records')

    return _evaluate_in_batches(data, evaluate_batch, progress_path, fingerprint)



if __name__ == "__main__":
    evaluate_knowledge_base()
