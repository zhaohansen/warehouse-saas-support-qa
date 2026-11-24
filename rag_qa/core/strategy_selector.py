"""检索策略选择：先用规则短路精准查询，其余由大模型在直接检索、HyDE、子查询、回溯中选一种。"""

import re
from langchain_core.prompts import PromptTemplate
from base.config import config
from base.logger import logger
from openai import OpenAI

# 明确包含错误码、接口名或具体版本号的查询属于精准检索场景，
# 直接检索既更快，也能避免 HyDE 生成臆造的错误码/接口参数干扰召回。
_DIRECT_SHORTCUT = re.compile(
    r'(?:错误码|错误编号|err(?:or)?\s*code)\s*(?:为|是|[:：=])?\s*[A-Za-z0-9_-]{2,}'
    r'|\b[A-Z]{2,}(?:-[A-Z0-9]{2,}){1,2}\b'
    r'|\b[a-z][a-z0-9_]{2,}\.(?:php|json|do|action)\b'
    r'|[Vv]\d+\.\d+(?:\.\d+){1,2}',
    re.IGNORECASE,
)


def _should_direct_search(query: str) -> bool:
    text = (query or "").strip()
    if not text or len(text) > 120:
        return False
    # 含多方面并列提问（如“以及”“并且”“分别”“区别”）时仍交给模型判断是否拆子查询
    if any(word in text for word in ('分别', '区别', '对比', '以及', '同时')):
        return False
    return bool(_DIRECT_SHORTCUT.search(text))


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
            你是仓配 SaaS 实施与售后知识库的检索策略路由器。请分析用户查询 {query}，从以下四种策略中选择最合适的一种，只输出策略名称，不要输出任何解释。

            可选策略及适用场景：

            1. **直接检索**
               * 适用场景：意图明确，指向具体接口、错误码、字段、操作步骤或单一规则的问题。
               * 示例：
                 * 查询：旺店通库存同步查询接口要求什么客户端版本？ -> 直接检索
                 * 查询：拼多多电子面单取号失败常见原因是什么？ -> 直接检索

            2. **假设问题检索**
               * 适用场景：表述偏口语或较抽象，缺少文档中的专业术语，需要先生成假设性排查要点再检索。
               * 示例：
                 * 查询：为什么仓库这边发完货了，顾客在平台上还是看不到物流更新？ -> 假设问题检索

            3. **子查询检索**
               * 适用场景：一个问题同时涉及多个平台、多个业务环节或多个需要对比的对象，需要拆分检索后合并。
               * 示例：
                 * 查询：抖店和拼多多在电子面单取号前置条件与回传流程上分别有什么要求？ -> 子查询检索

            4. **回溯问题检索**
               * 适用场景：包含大量具体订单/门店背景细节，需要先抽象成更通用的基础规则问题再检索。
               * 示例：
                 * 查询：我们华东仓 3 号抖店昨天双十一大促期间有 200 多笔预售单在 ERP 里一直没抓下来，应该先查哪一块？ -> 回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称（直接检索 / 假设问题检索 / 子查询检索 / 回溯问题检索）。
            """,
            input_variables=["query"],
        )

    def select_strategy(self, query):
        """
        函数作用: 根据用户查询, 选择最合适的检索增强策略.
        :param query: 用户输入的查询文本(字符串)
        :return: 字符串 -> 选中的检索策略名称 -> 例如: 直接检索, 子查询检索...
        """
        if _should_direct_search(query):
            logger.debug(f"精准查询规则短路，直接使用直接检索：'{query}'")
            return "直接检索"
        strategy = self.call_llm(self.strategy_prompt_template.format(query=query)).strip()
        valid = {"直接检索", "假设问题检索", "子查询检索", "回溯问题检索"}
        strategy = strategy.strip('"\' `。\n')
        if strategy not in valid:
            strategy = "直接检索"
        logger.debug(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy

if __name__ == '__main__':
    selector = StrategySelector()
    print(selector.select_strategy('客户反馈处理流程是什么？'))
