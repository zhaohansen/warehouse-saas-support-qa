"""RAGAS 评估：计算忠实度、答案相关性、上下文精确率与召回率。"""

from ragas import evaluate
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

with open('rag_evaluate_data.json', 'r', encoding='utf-8') as f:
    data = json.load(
        f)  # data的数据格式: [{'question': '问题', 'answer': '答案', 'context': '上下文', 'ground_truth': '真实答案'}, ...]
print(f'加载的评估样本数量: {len(data)}')
eval_data = {
    'question': [item['question'] for item in data],
    'answer': [item['answer'] for item in data],
    'contexts': [item['context'] for item in data],
    'ground_truth': [item['ground_truth'] for item in data]
}
dataset = Dataset.from_dict(eval_data)
print(f'转换后的数据集: {dataset}')


class RagasChatOllama(ChatOllama):
    def _chat_params(self, messages, stop=None, **kwargs):
        temperature = kwargs.pop('temperature', None)
        kwargs.pop('n', None)
        params = super()._chat_params(messages, stop, **kwargs)
        if temperature is not None:
            options = dict(params.get('options') or {})
            options['temperature'] = temperature
            params['options'] = options
        return params

llm = LangchainLLMWrapper(
    RagasChatOllama(model='qwen2.5:7b', base_url='http://localhost:11434', temperature=0)
)
embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model='bge-m3', base_url='http://localhost:11434')
)
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm,
    embeddings=embeddings
)
print(f'RAGAS评估结果: {result}')
