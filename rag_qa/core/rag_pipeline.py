"""问答链路：本地检索优先，未命中时辅助分类与增强，支持多轮和流式。"""

import sys, os
from rag_qa.core.prompts import RAGPrompts
from rag_qa.core.query_classifier import QueryClassifier
from rag_qa.core.strategy_selector import StrategySelector
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)
sys.path.insert(0, rag_qa_path)
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from base.config import Config
from base.logger import logger

# 这些词仅在原问题检索未命中后辅助选择增强策略，不能用于跳过首次检索。
SUBJECT_HINTS = (
    "python", "java", "mysql", "redis", "sql", "jvm", "gil", "hashmap",
    "索引", "事务", "隔离级别", "mvcc", "锁", "主从", "分库", "分表",
    "慢查询", "存储引擎", "b+树", "执行计划", "回表", "聚簇",
    "缓存", "持久化", "哨兵", "集群", "穿透", "雪崩", "击穿", "过期",
    "线程", "进程", "协程", "装饰器", "生成器", "迭代器", "异常",
    "面向对象", "垃圾回收", "内存", "集合", "类加载",
    "大模型", "rag", "检索", "向量", "嵌入", "embedding", "重排", "召回",
    "微调", "transformer", "注意力", "提示词", "过拟合", "正则化",
    "知识库", "评估", "幻觉",
)


def _looks_like_subject_query(query):
    """判断问题是否带有学科特征 -> 用于兜底纠正分类器的误判."""
    text = (query or "").lower()
    return any(hint in text for hint in SUBJECT_HINTS)

# 注意: 这里不再在模块导入时加载 BERT 模型(原代码写死了另一台机器的绝对路径, 会直接报错),
# 需要用到时在 QueryClassifier 内部按配置路径加载.

conf = Config()


def _is_follow_up(query):
    import re
    text = re.sub(r'[\s？?！!。]', '', query).casefold()
    return (text in {'继续', '然后呢', '为什么', '为什么呢', '为啥', '举例', '举个例子',
                     '给个例子', '具体呢', '给代码', '怎么写', '怎么实现', 'python实现',
                     '用python实现', '用python实现一下', 'java实现', '用java实现'}
            or text.startswith(('那它', '那么它', '这个怎么', '它的', '它为什么', '刚才', '上面那个')))


def _requires_local_evidence(query):
    import re
    return bool(re.search(
        r'(?:根据|依据|按照|只用|仅用|只从).{0,12}(?:资料|文档|知识库|课件|pdf|ppt)'
        r'|(?:这份|这个|该|我上传的|上传的).{0,8}(?:资料|文档|文件|课件|pdf|ppt)'
        r'|你们|我们公司|本公司|本机构|内部政策|内部制度|学费|报名费用|开课时间|报销政策',
        query, re.IGNORECASE))


class RAGSystem:
    def __init__(self, vector_store, llm, stream_llm=None):
        """
        函数作用: 初始化RAG系统
        :param vector_store: 向量数据库对象 -> 用于存储和检索文档, 提供相似性搜索功能.
        :param llm: 大语言模型调用函数 -> 接收提示文本, 返回模型生成的回答.
        """
        self.vector_store = vector_store
        self.llm = llm
        self.stream_llm = stream_llm
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
                source_filter=source_filter, rerank_query=query
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return self.vector_store.hybrid_search_with_rerank(query, source_filter=source_filter)

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
            subqueries = [q.strip()[:500] for q in subqueries_text.split("\n") if q.strip()][:5]
            logger.info(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return self.vector_store.hybrid_search_with_rerank(query, source_filter=source_filter)

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
            return self.vector_store.rerank_documents(query, unique_docs)

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return self.vector_store.hybrid_search_with_rerank(query, source_filter=source_filter)

    def _retrieve_with_backtracking(self, query, source_filter=None):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
            logger.info(f"生成的回溯问题: '{simplified_query}'")
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K, source_filter=source_filter, rerank_query=query
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return self.vector_store.hybrid_search_with_rerank(query, source_filter=source_filter)
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):

        """
        函数作用: 统一入口: 根据指定策略(或自动选择策略)调用对应检索方法, 筛选最终用于生成答案的上下文文档.
        :param query: 用户的原始查询文本 -> str, 传递给策略选择器 和 检索方法.
        :param source_filter: 检索来源过滤条件
        :param strategy: 指定检索策略, 可选: 直接检索, 回溯问题检索...
        :return: list[Document]
        """
        if not strategy:
            # Always try the user's actual question before an LLM rewrite.
            baseline = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter)
            if baseline:
                logger.info('原问题命中本地资料: %d 个文档，来源: %s', len(baseline),
                            [(doc.metadata.get('filename') or doc.metadata.get('source'),
                              doc.metadata.get('page') or doc.metadata.get('slide')) for doc in baseline])
                return baseline[:conf.CANDIDATE_M]
            # Classification is advisory only after a genuine lookup miss.
            # A course concept can be both general knowledge and present in a user's files.
            classifier = getattr(self, 'query_classifier', None)
            if classifier is not None:
                try:
                    category = classifier.predict_category(query)
                except Exception:
                    category = '专业咨询'
                if category == '通用知识' and not _requires_local_evidence(query) and not _looks_like_subject_query(query):
                    return []
            strategy = self.strategy_selector.select_strategy(query)
            if strategy == '直接检索':
                return []
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
            # 与其他三种策略保持一致: 检索失败退化为"无上下文", 由模型凭通用能力作答,
            # 而不是把异常一路抛到接口层, 让用户看到 Milvus 的内部报错.
            try:
                ranked_sub_chunks = self.vector_store.hybrid_search_with_rerank(
                    query, k=conf.RETRIEVAL_K, source_filter=source_filter
                )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档
            except Exception as e:
                logger.exception(f"直接检索失败: {e}")
                raise

        # 选择最终上下文文档, 截取前conf.CANDIDATE_M个文档(控制上下文长度, 避免LLM输入超限)
        logger.info(f"策略 {strategy} 检索到 {len(ranked_sub_chunks)} 个候选文档")
        final_context_docs = ranked_sub_chunks[: conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")

        return final_context_docs

    def _prepare_answer(self, query, source_filter=None, history=None):
        history = [h for h in (history if isinstance(history, list) else [])[-5:] if isinstance(h, dict)
                   and isinstance(h.get('question'), str) and isinstance(h.get('answer'), str)]
        history_context = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history)
        history_context = history_context[-conf.MAX_HISTORY_CHARS:]
        retrieval_query = query
        anchor = ''
        if history and _is_follow_up(query):
            anchor = next((item['question'] for item in reversed(history)
                           if not _is_follow_up(item['question'])), history[-1]['question'])
            if ''.join(query.split()).casefold() in ''.join(anchor.split()).casefold():
                retrieval_query = anchor
            else:
                retrieval_query = anchor + "\n后续问题：" + query
        needs_local = _requires_local_evidence(query) or bool(anchor and _requires_local_evidence(anchor))
        # This is a knowledge-base assistant: every non-FAQ question must first
        # have a chance to match local material, regardless of BERT's old labels.
        try:
            documents = self.retrieve_and_merge(retrieval_query, source_filter=source_filter)
        except Exception:
            logger.exception('知识检索失败')
            return None, [], '知识库检索暂时失败，请确认向量库服务正常后重试。'
        if not documents and needs_local:
            return None, [], '当前知识库没有找到足够相关的资料，无法依据本地文件确认这一点。请补充对应资料，或选择正确的学科。'
        selected, snippets, remaining = [], [], conf.MAX_CONTEXT_CHARS
        for doc in documents:
            if remaining <= 0: break
            meta = doc.metadata
            title = meta.get('filename') or os.path.basename(meta.get('file_path') or '') or meta.get('source', '知识库')
            location = f"第 {meta['page']} 页" if meta.get('page') else f"第 {meta['slide']} 张幻灯片" if meta.get('slide') else ''
            snippet = doc.page_content[:remaining]
            snippets.append(f"[{len(selected)+1}] {title} {location}\n{snippet}")
            selected.append(doc)
            remaining -= len(snippet)
        prompt = self.rag_prompt.format(context="\n\n".join(snippets), history=history_context,
                                        question=query, phone=conf.CUSTOMER_SERVICE_PHONE)
        return prompt, selected, None

    def generate_answer_stream(self, query, source_filter=None, history=None, on_context=None):
        prompt, documents, direct = self._prepare_answer(query, source_filter, history)
        if on_context: on_context(documents)
        if direct:
            yield direct
            return
        if not documents:
            yield '以下为通用知识回答，未引用本地资料：\n\n'
        if self.stream_llm:
            yield from self.stream_llm(prompt)
        else:
            yield self.llm(prompt)

    def generate_answer(self, query, source_filter=None, history=None):
        return ''.join(self.generate_answer_stream(query, source_filter, history))
