"""问答链路：先分类，再按需选择检索策略，支持多轮和流式。"""

import sys, os, re
from concurrent.futures import CancelledError
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

# 高辨识度学科术语仅用于纠正明显误判，不将普通词如“锁”一概视作学科问题。
_SUBJECT_TERMS = re.compile(
    r'(?<![a-z0-9_])(?:python|java|mysql|redis|sql|jvm|bert|rag|transformer)(?![a-z0-9_])'
    r'|人工智能|机器学习|深度学习|神经网络|大语言模型|大模型|监督学习|无监督学习'
    r'|强化学习|注意力机制|向量数据库|向量检索|数据库索引|数据库事务|模型过拟合',
    re.IGNORECASE,
)


def _normalize_subject_terms(query):
    """修正常见且明确的技术名词拼写，不改变传给生成模型的原始问题。"""
    return re.sub(r'(?<![a-z0-9_])reides(?![a-z0-9_])', 'redis', query, flags=re.IGNORECASE)


def _has_clear_subject_term(query):
    return bool(_SUBJECT_TERMS.search(_normalize_subject_terms(query)))


# 注意: 这里不再在模块导入时加载 BERT 模型(原代码写死了另一台机器的绝对路径, 会直接报错),
# 需要用到时在 QueryClassifier 内部按配置路径加载.

conf = Config()


class RetrievalError(RuntimeError):
    """本地知识库检索失败；不能与正常的无命中结果混为一谈。"""


def _document_identity(doc):
    """保留同学科内不同文件、页码和父块的相同文本来源。"""
    meta = doc.metadata
    return (meta.get('source'), meta.get('file_path'), meta.get('page'), meta.get('slide'),
            meta.get('parent_id'), doc.page_content)


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
        r'|课件里写的|讲义里写的|老师要求|课程安排|评分标准|作业要求|实验要求|考核方式|内部政策|内部制度',
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
        logger.debug(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        hyde_prompt_template = RAGPrompts.hyde_prompt()
        try:
            hypo_answer = self.llm(hyde_prompt_template.format(query=query)).strip()
            logger.debug(f"HyDE 生成的假设答案: '{hypo_answer}'")
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
        logger.debug(f"使用子查询策略进行检索 (查询: '{query}')")
        subquery_prompt_template = RAGPrompts.subquery_prompt()
        try:
            subqueries_text = self.llm(subquery_prompt_template.format(query=query)).strip()
            subqueries = [q.strip()[:500] for q in subqueries_text.split("\n") if q.strip()][:5]
            logger.debug(f"生成的子查询: {subqueries}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return self.vector_store.hybrid_search_with_rerank(query, source_filter=source_filter)

            all_docs = []
            for sub_q in subqueries:
                docs = self.vector_store.hybrid_search_with_rerank(
                    sub_q, k=conf.RETRIEVAL_K, source_filter=source_filter
                )
                all_docs.extend(docs)
                logger.debug(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")
            unique_docs_dict = {_document_identity(doc): doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())
            logger.debug(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            return self.vector_store.rerank_documents(query, unique_docs)

        except Exception as e:
            logger.error(f"子查询策略执行失败: {e}")
            return self.vector_store.hybrid_search_with_rerank(query, source_filter=source_filter)

    def _retrieve_with_backtracking(self, query, source_filter=None):
        logger.debug(f"使用回溯问题策略进行检索 (查询: '{query}')")
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()
        try:
            simplified_query = self.llm(backtrack_prompt_template.format(query=query)).strip()
            logger.debug(f"生成的回溯问题: '{simplified_query}'")
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
        first_retrieval = None
        if not strategy:
            # 分类器可能误判；所有问题先检索一次本地知识库。
            first_retrieval = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter)
            if first_retrieval:
                result = first_retrieval[:conf.CANDIDATE_M]
                logger.info('原问题检索命中本地资料：%d 条', len(result))
                return result
            classifier = getattr(self, 'query_classifier', None)
            try:
                category = classifier.predict_category(query) if classifier is not None else '学科资料'
                logger.info('意图识别（问题分类）：%s', category)
            except Exception:
                logger.exception('问题分类失败，按学科资料继续检索')
                category = '学科资料'

            needs_local = _requires_local_evidence(query)
            has_subject_term = _has_clear_subject_term(query)
            if category == '通用知识' and not needs_local and not has_subject_term:
                logger.info('原问题检索未命中，按通用知识回答')
                return []
            if category == '通用知识':
                reason = '明确要求本地资料' if needs_local else '包含明确学科术语'
                logger.info('最终路由：学科资料（%s，修正分类器结果）', reason)
            else:
                logger.debug('最终路由：学科资料，选择检索策略')
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
            logger.debug(f"使用直接检索策略 (查询: '{query}')")
            # 与其他三种策略保持一致: 检索失败退化为"无上下文", 由模型凭通用能力作答,
            # 而不是把异常一路抛到接口层, 让用户看到 Milvus 的内部报错.
            try:
                ranked_sub_chunks = first_retrieval if first_retrieval is not None else (
                    self.vector_store.hybrid_search_with_rerank(
                        query, k=conf.RETRIEVAL_K, source_filter=source_filter))
            except Exception as e:
                logger.exception(f"直接检索失败: {e}")
                raise

        # 选择最终上下文文档, 截取前conf.CANDIDATE_M个文档(控制上下文长度, 避免LLM输入超限)
        final_context_docs = ranked_sub_chunks[: conf.CANDIDATE_M]
        logger.info('检索结果 | 策略：%s | 候选：%d | 选用：%d', strategy, len(ranked_sub_chunks), len(final_context_docs))

        return final_context_docs

    def _prepare_answer(self, query, source_filter=None, history=None):
        history = [h for h in (history if isinstance(history, list) else [])[-5:] if isinstance(h, dict)
                   and isinstance(h.get('question'), str) and isinstance(h.get('answer'), str)]
        retrieval_query = query
        anchor = ''
        is_follow_up = _is_follow_up(query)
        # 新的独立问题不携带无关历史；追问仍保留最近对话。
        relevant_history = history if is_follow_up else []
        history_context = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in relevant_history)
        history_context = history_context[-conf.MAX_HISTORY_CHARS:]
        if history and is_follow_up:
            anchor = next((item['question'] for item in reversed(history)
                           if not _is_follow_up(item['question'])), history[-1]['question'])
            if ''.join(query.split()).casefold() in ''.join(anchor.split()).casefold():
                retrieval_query = anchor
            else:
                retrieval_query = anchor + "\n后续问题：" + query
        normalized_query = _normalize_subject_terms(retrieval_query)
        if normalized_query != retrieval_query:
            logger.info('检索问题已修正常见学科术语拼写')
        retrieval_query = normalized_query
        needs_local = _requires_local_evidence(query) or bool(anchor and _requires_local_evidence(anchor))
        logger.info('问题判断 | 追问：%s | 需要本地资料：%s', '是' if is_follow_up else '否', '是' if needs_local else '否')
        try:
            documents = self.retrieve_and_merge(retrieval_query, source_filter=source_filter)
        except Exception as exc:
            logger.exception('知识检索失败')
            raise RetrievalError('知识库检索暂时失败，请确认向量库服务正常后重试。') from exc
        if not documents and needs_local:
            return None, [], '当前知识库没有找到足够相关的资料，无法依据本地文件确认这一点。请补充对应资料，或选择正确的学科。'
        selected, snippets, remaining = [], [], conf.MAX_CONTEXT_CHARS
        for doc in documents:
            if remaining <= 0: break
            meta = doc.metadata
            title = meta.get('filename') or os.path.basename(meta.get('file_path') or '') or meta.get('source',
                                                                                                      '知识库')
            location = f"第 {meta['page']} 页" if meta.get('page') else f"第 {meta['slide']} 张幻灯片" if meta.get(
                'slide') else ''
            snippet = doc.page_content[:remaining]
            snippets.append(f"[{len(selected) + 1}] {title} {location}\n{snippet}")
            selected.append(doc)
            remaining -= len(snippet)
        prompt = self.rag_prompt.format(context="\n\n".join(snippets), history=history_context,
                                        question=query)
        return prompt, selected, None

    def generate_answer_stream(self, query, source_filter=None, history=None, on_context=None,
                               cancel_event=None):
        if cancel_event and cancel_event.is_set(): raise CancelledError('生成已取消')
        prompt, documents, direct = self._prepare_answer(query, source_filter, history)
        if cancel_event and cancel_event.is_set(): raise CancelledError('生成已取消')
        if on_context: on_context(documents)
        if direct:
            yield direct
            return
        if not documents:
            yield '以下为通用知识回答，未引用本地资料：\n\n'
        if self.stream_llm:
            if cancel_event is None:
                yield from self.stream_llm(prompt)
            else:
                yield from self.stream_llm(prompt, cancel_event=cancel_event)
        else:
            yield self.llm(prompt)

    def generate_answer(self, query, source_filter=None, history=None):
        return ''.join(self.generate_answer_stream(query, source_filter, history))
