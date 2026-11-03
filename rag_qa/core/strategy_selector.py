"""检索策略选择：由大模型在直接检索、HyDE、子查询、回溯中选一种。"""

from langchain_core.prompts import PromptTemplate
from base.config import config
from base.logger import logger
from openai import OpenAI


class StrategySelector:
    def __init__(self):
        self.client = OpenAI(api_key=config.LLM_API_KEY or "not-configured",
                             base_url=config.LLM_BASE_URL, timeout=20, max_retries=0)
        self.strategy_prompt_template = self._get_strategy_prompt()

    def call_llm(self, prompt):
        """
        调用配置的大模型 API，获取模型对输入提示的响应。
        :param prompt: 发给大模型的提示文本(字符和窜), 包含: 用户查询和任务要求.
        :return: 大模型返回的文本结果(字符串), 这里是选择的策略名称, 如果调用失败, 直接返回: 直接检索
        """
        try:
            completion = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手，能够根据用户输入的Prompt严格执行并返回可靠的结果"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                timeout=20
            )
            return (completion.choices[0].message.content or "直接检索") if completion.choices else "直接检索"
        except Exception as e:
            logger.error(f"模型 API 调用失败: {e}")
            return "直接检索"

    def _get_strategy_prompt(self):
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：AI 课件里讲的注意力机制是什么？
                        * 策略：直接检索
                    * 查询：Java 课件第三章的索引优化讲的是什么？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )

    def select_strategy(self, query):
        """
        函数作用: 根据用户查询, 选择最合适的检索增强策略.
        :param query: 用户输入的查询文本(字符串)
        :return: 字符串 -> 选中的检索策略名称 -> 例如: 直接检索, 子查询检索...
        """
        strategy = self.call_llm(self.strategy_prompt_template.format(query=query)).strip()
        valid = {"直接检索", "假设问题检索", "子查询检索", "回溯问题检索"}
        strategy = strategy.strip('"\' `。\n')
        if strategy not in valid:
            strategy = "直接检索"
        logger.debug(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy

if __name__ == '__main__':
    selector = StrategySelector()
    print(selector.select_strategy('课件第三章讲的索引优化是什么？'))
