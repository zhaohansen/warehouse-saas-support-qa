"""基础问答链路：分类 → 检索 → 生成，不含多轮历史。"""

import sys, os
import time
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.query_classifier import QueryClassifier
from rag_qa.core.strategy_selector import StrategySelector
from rag_qa.core.vector_store import VectorStore
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
sys.path.insert(0, rag_qa_path)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from base.config import Config
from base.logger import logger

# 学科特征词: 与 rag_pipeline 保持一致, 命中的问题不走"通用知识"短路.
from rag_qa.core.rag_pipeline import SUBJECT_HINTS


def looks_like_subject_query(query):
    text = (query or '').lower()
    return any(hint in text for hint in SUBJECT_HINTS)

conf = Config()


class RAGSystem:
    def __init__(self, vector_store, llm):
        """
        函数作用: 初始化RAG系统
        :param vector_store: 向量数据库对象 -> 用于存储和检索文档, 提供相似性搜索功能.
        :param llm: 大语言模型调用函数 -> 接收提示文本, 返回模型生成的回答.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = RAGPrompts.rag_prompt()
        classifier_path = os.path.join(rag_qa_path, 'models', 'bert_query_classifier')
        self.query_classifier = QueryClassifier(classifier_path)
        self.strategy_selector = StrategySelector()

    def _retrieve_with_hyde(self, query, source_filter=None):
        """
        函数作用: 针对于抽象/开放性查询, 先生成假设性答案, 再用假设答案检索文档 -> 解决 抽象查询 直接检索关键词 匹配度低 的问题.
        :param query: 用户的原始查询文本(str).
        :param source_filter: 检索来源过滤条件(str或者None), 例如: education表示只检索与教育相关的文档,  None表示不过滤.
        :return: list[Document], 每个Document对象的page_content属性为文档文本, 用于后续生成答案的上下文.
        """
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()
            logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
            #  注意：HyDE 通常只用于生成检索向量，不一定需要 rerank 这一步，但这里复用了
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    def _retrieve_with_subqueries(self, query, source_filter=None):
        """
        函数作用: 针对于多实体/多方面的复杂查询, 拆分为多个简单子查询, 分别检索后合并去重 -> 解决多维度查询检索不全面的问题.
        :param query: 用户的原始查询文本.
        :param source_filter: 检索来源过滤条件(str或者None), 例如: education表示只检索与教育相关的文档,  None表示不过滤.
        :return: list[Document]对象.
        """
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []

            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K, source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            return unique_docs

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return []

    def _retrieve_with_backtracking(self, query, source_filter=None):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):

        """
        函数作用: 统一入口: 根据指定策略(或自动选择策略)调用对应检索方法, 筛选最终用于生成答案的上下文文档.
        :param query: 用户的原始查询文本 -> str, 传递给策略选择器 和 检索方法.
        :param source_filter: 检索来源过滤条件
        :param strategy: 指定检索策略, 可选: 直接检索, 回溯问题检索...
        :return: list[Document]
        """
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)
        ranked_sub_chunks = []

        if strategy == "回溯问题检索":
            # 所有策略都传递来源过滤，避免回溯/子查询/HyDE 绕过用户筛选条件。
            ranked_sub_chunks = self._retrieve_with_backtracking(query, source_filter=source_filter)
        elif strategy == "子查询检索":
            ranked_sub_chunks = self._retrieve_with_subqueries(query, source_filter=source_filter)
            # 注意：子查询返回的是已 rerank 过的父文档或子块列表，后续合并逻辑可能需要调整
            # 当前实现中，子查询返回的是初步检索（可能已rerank）的块，再进行合并
        elif strategy == "假设问题检索":
            ranked_sub_chunks = self._retrieve_with_hyde(query, source_filter=source_filter)
        else:
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        # 3. 选择最终上下文文档, 截取前conf.CANDIDATE_M个文档(控制上下文长度, 避免LLM输入超限)
        logger.info(f"策略 {strategy} 检索到 {len(ranked_sub_chunks)} 个候选文档")
        final_context_docs = ranked_sub_chunks[: conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")

        return final_context_docs

    def generate_answer(self, query, source_filter=None):
        """
        函数作用: RAG系统对外核心接口 -> 接收用户查询, 自动完成'查询分类 -> 策略选择 -> 文档检索 -> 答案生成'全流程
        :param query: 用户录入的原始查询文本(字符串形式)
        :param source_filter: 检索来源过滤条件, 仅对'专业咨询'生效.
        :return: 生成的最终答案文本 -> 字符串形式.
        """
        start_time = time.time()
        logger.info(f"开始处理查询: {query}, 学科过滤: {source_filter}")
        query_category = self.query_classifier.predict_category(query)
        # 2.1 规则兜底: 指定了学科或命中学科关键词时, 一律按专业咨询检索.
        if source_filter or looks_like_subject_query(query):
            if query_category != '专业咨询':
                logger.info(f'分类器判为 {query_category}, 但命中学科特征, 按专业咨询走检索')
            query_category = '专业咨询'

        if query_category == "通用知识":
            logger.info("查询为通用知识, 直接调用LLM生成答案")
            prompt_input = self.rag_prompt.format(
                context="",
                history="",
                question=query,
                phone=conf.CUSTOMER_SERVICE_PHONE
            )

            try:
                answer = self.llm(prompt_input)
            except Exception as e:
                logger.error(f'直接调用LLM失败: {e}')
                answer = f"抱歉,处理您的通用知识问题时出错,请联系人工客服: {conf.CUSTOMER_SERVICE_PHONE}"

            processing_time = time.time() - start_time
            logger.info(f"通用知识查询处理完成, 耗时: {processing_time:.3f}s, 查询问题: {query}")

            return answer

        logger.info("查询为专业咨询, 执行完整 RAG 流程")
        strategy = self.strategy_selector.select_strategy(query)
        context_docs = self.retrieve_and_merge(query, source_filter=source_filter, strategy=strategy)

        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])
            logger.info(f"构建上下文完成, 包含 {len(context_docs)} 个文档块")
        else:
            context = ""
            logger.info("未检索到相关文档, 上下文为空")

        prompt_input = self.rag_prompt.format(
            context=context,
            history="",
            question=query,
            phone=conf.CUSTOMER_SERVICE_PHONE
        )

        try:
            answer = self.llm(prompt_input)
        except Exception as e:
            logger.error(f'直接调用LLM失败: {e}')
            answer = f"抱歉,处理您的 专业咨询 问题时出错,请联系人工客服: {conf.CUSTOMER_SERVICE_PHONE}"

        processing_time = time.time() - start_time
        logger.info(f"查询处理完成, 耗时: {processing_time:.3f}s, 查询问题: {query}")

        return answer

if __name__ == '__main__':
    vector_store = VectorStore()
    llm = StrategySelector().call_dashscope
    rag_system = RAGSystem(vector_store, llm)
    answer = rag_system.generate_answer('计算圆周率近似值的方法有哪些？', source_filter='ai')
    print(answer)
