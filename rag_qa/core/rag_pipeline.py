"""问答链路：先分类，再按需选择检索策略，支持多轮和流式。"""

import re
import sys, os
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

# 是否必须检索企业资料，需要综合判断，不能只靠一份关键词表：
# 1) 明确指向公司/组织（如“云帆科技”“公司”“员工”）；
# 2) 属于企业业务主题（请假、报销、工单、协作平台等）；
# 3) 通用技术词只有在带操作诉求时才按内部资料处理，“VPN 是什么”仍按常识回答。
_COMPANY_MARKERS = re.compile(
    r'云帆|我们公司|我司|本公司|公司|企业|集团|内部|在职|员工|同事|部门|岗位|职级|工号|员工号'
    r'|组织架构|汇报对象|本部门|我所在',
    re.IGNORECASE,
)
_ENTERPRISE_TOPICS = re.compile(
    r'请假|休假|年假|病假|事假|调休|加班|排班|销假|考勤|打卡|入职|离职|转正|试用期|实习'
    r'|报销|发票|差旅|出差|补贴|工资|薪资|薪酬|奖金|绩效|晋升|调薪|社保|公积金|个税'
    r'|审批|权限|账号|账户|工牌|门禁|办公设备|办公用品|资产|采购|预算|合同|盖章|用印'
    r'|档案|保密|合规|审计|培训|导师|会议室|工位|班车|食堂|体检|福利|餐卡|餐补|停车'
    r'|工单|客户反馈|售后|退款|协作平台|办公平台|内部系统|业务系统|办公系统|服务台|客服'
    r'|激活邮件|公司邮箱|企业邮箱|知识库|手册',
    re.IGNORECASE,
)
# 通用技术词单独出现时按常识回答，只有“怎么开通/怎么连接/找谁”等操作诉求才检索企业资料。
_TECH_SYSTEM_TERMS = re.compile(r'\b(?:VPN|OA|ERP|CRM|SSO|API|LDAP)\b', re.IGNORECASE)
_TECH_OPERATIONS = re.compile(r'怎么|如何|在哪|哪里|申请|开通|连接|配置|找谁|联系谁|权限|账号|登录',
                              re.IGNORECASE)
_DEFINITION_PATTERN = re.compile(
    r'是什么|什么是|什么意思|含义|定义|概念|原理|解释一下|介绍一下|怎么理解',
    re.IGNORECASE,
)
# 明确要求“只依据本地资料回答”的说法。
_EXPLICIT_LOCAL = re.compile(
    r'(?:根据|依据|按照|只用|仅用|只从|基于).{0,12}(?:资料|文档|知识库|手册|制度|pdf|ppt)'
    r'|(?:这份|这个|该|我上传的|上传的|刚才的).{0,8}(?:资料|文档|文件|pdf|ppt)',
    re.IGNORECASE,
)


def _is_follow_up(query):
    text = re.sub(r'[\s？?！!。]', '', query).casefold()
    return (text in {'继续', '然后呢', '为什么', '为什么呢', '为啥', '举个例子', '给个例子', '具体呢',
                     '详细点', '详细说说', '还有呢', '需要什么材料', '要多久', '在哪办理', '找谁',
                     '联系谁', '谁负责', '怎么办', '怎么弄', '需要什么'}
            or text.startswith(('那它', '那么它', '那应该', '那要', '那需要', '那谁', '那怎么', '那如果',
                                '那这个', '那这', '那么', '然后', '还有', '接着', '它', '这个怎么', '它的',
                                '它为什么', '刚才', '上面那个', '这个流程', '该找', '该联系', '该问'))
            or (text.startswith('那') and len(text) <= 12))


def _requires_local_evidence(query, source_filter=None):
    """判断回答是否必须依赖企业资料。

    依次考虑：用户选中的资料分类、明确要求引用、纯概念定义、公司指向、企业业务主题、
    带操作诉求的通用技术词。纯概念问题（如“VPN 是什么？”）不检索企业资料。
    """
    text = (query or '').strip()
    if source_filter:
        return True
    if not text:
        return False
    if _EXPLICIT_LOCAL.search(text):
        return True
    # 先看公司指向和企业业务主题：例如“报销的审批流程顺序是什么”虽然以“是什么”结尾，
    # 仍属于企业问题；只有不含企业主题的纯概念问题才走通用回答。
    if _COMPANY_MARKERS.search(text) or _ENTERPRISE_TOPICS.search(text):
        return True
    if _DEFINITION_PATTERN.search(text):
        return False
    return bool(_TECH_SYSTEM_TERMS.search(text) and _TECH_OPERATIONS.search(text))


def _follow_up_requires_local_evidence(query, anchor):
    """判断追问是否应继承企业检索。

    上一轮本身是企业问题时直接继承；上一轮是“VPN 是什么”这类通用技术定义时，
    只有本轮出现申请、开通、连接等操作诉求才转入企业资料，避免把“那它有什么缺点”
    之类的概念追问误送去检索。
    """
    if not anchor:
        return False
    if _requires_local_evidence(anchor):
        return True
    return bool(_TECH_SYSTEM_TERMS.search(anchor) and _TECH_OPERATIONS.search(query)
                and not _DEFINITION_PATTERN.search(query))


conf = Config()


class RetrievalError(RuntimeError):
    """本地知识库检索失败；不能与正常的无命中结果混为一谈。"""


def _document_identity(doc):
    """保留同一分类内不同文件、页码和父块的相同文本来源。"""
    meta = doc.metadata
    return (meta.get('source'), meta.get('file_path'), meta.get('page'), meta.get('slide'),
            meta.get('parent_id'), doc.page_content)


def _document_score(doc):
    """重排分数缺失时按可用处理，避免测试替身或旧数据被误判为无关。"""
    try:
        return float(doc.metadata.get('score', 1.0))
    except (TypeError, ValueError):
        return 1.0


def _filter_citation_markers(chunks, max_index):
    """按块透传普通文本，只保留末尾可能被拆开的引用标记。

    同一模型块内的普通文本会合并输出；合法引用 [n]（1 <= n <= max_index）原样输出；
    越界引用丢弃；普通方括号文本（如 [备注]）原样保留；块末尾的“[”“[1”等未完成
    标记留到下一块再判断，流结束时仍未闭合的方括号也会原样输出，避免吞掉正文。
    """
    pending = ''
    for chunk in chunks:
        if not chunk:
            continue
        text = pending + chunk
        pending = ''
        start = 0
        parts = []
        while True:
            opening = text.find('[', start)
            if opening < 0:
                if start < len(text):
                    parts.append(text[start:])
                break
            if opening > start:
                parts.append(text[start:opening])
            rest = text[opening + 1:]
            closing = rest.find(']')
            if closing < 0:
                candidate = rest
                # 只有空内容或纯数字才可能发展成引用编号，其余按普通文本立即输出。
                if len(candidate) <= 10 and (not candidate or candidate.isdigit()):
                    pending = text[opening:]
                else:
                    parts.append(text[opening:])
                break
            end = opening + 1 + closing + 1
            inner = rest[:closing]
            if inner.isdigit() and 1 <= int(inner) <= max_index:
                parts.append(text[opening:end])
            elif not inner.isdigit():
                # 普通方括号文本，不当作引用，完整保留。
                parts.append(text[opening:end])
            # 纯数字但越界：丢弃整个 [n]。
            start = end
        # 同一模型块内的普通文本合并后输出，避免因过滤引用而被拆成多段。
        if parts:
            yield ''.join(parts)
    if pending:
        yield pending


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
        classifier_path = os.path.join(rag_qa_path, 'models', 'bert_enterprise_classifier')
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
        """先选择策略，再检索原问题；增强策略结果与原问题结果合并重排。"""
        if not strategy:
            classifier = getattr(self, 'query_classifier', None)
            try:
                category = classifier.predict_category(query) if classifier is not None else '企业资料'
                logger.info('意图识别（问题分类）：%s', category)
            except Exception:
                logger.exception('问题分类失败，仍继续选择策略并检索')
            try:
                strategy = self.strategy_selector.select_strategy(query)
            except Exception:
                logger.exception('选择检索策略失败，回退直接检索')
                strategy = '直接检索'

        # 不论分类结果或选中策略，都先检索原问题。
        direct_docs = self.vector_store.hybrid_search_with_rerank(
            query, k=conf.RETRIEVAL_K, source_filter=source_filter)

        if strategy == '回溯问题检索':
            enhanced_docs = self._retrieve_with_backtracking(query, source_filter=source_filter)
        elif strategy == '子查询检索':
            enhanced_docs = self._retrieve_with_subqueries(query, source_filter=source_filter)
        elif strategy == '假设问题检索':
            enhanced_docs = self._retrieve_with_hyde(query, source_filter=source_filter)
        else:
            enhanced_docs = []

        if enhanced_docs:
            unique_docs = {}
            for doc in direct_docs + enhanced_docs:
                unique_docs.setdefault(_document_identity(doc), doc)
            ranked_docs = self.vector_store.rerank_documents(query, list(unique_docs.values()))
        else:
            ranked_docs = direct_docs

        final_docs = ranked_docs[:conf.CANDIDATE_M]
        logger.info('检索结果 | 策略：%s | 原问题：%d | 增强结果：%d | 选用：%d',
                    strategy, len(direct_docs), len(enhanced_docs), len(final_docs))
        return final_docs

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
        previous = history[-1] if history else {}
        previous_answer = (previous.get('answer') or '') if isinstance(previous, dict) else ''
        previous_sources = previous.get('sources') if isinstance(previous, dict) else None
        # 连续追问时，上一轮已经检索到来源或明确提示资料不足，本轮继续按企业问题处理。
        inherits_local = bool(previous_sources) or any(
            marker in previous_answer for marker in ('没有找到足够相关的资料', '资料不足'))
        needs_local = (bool(source_filter) or _requires_local_evidence(query, source_filter)
                       or (bool(is_follow_up) and inherits_local)
                       or _follow_up_requires_local_evidence(query, anchor))
        rule_needs_local = needs_local
        classifier = getattr(self, 'query_classifier', None)
        predict = getattr(classifier, 'predict_with_confidence', None)
        if rule_needs_local:
            logger.info('规则判定需要本地资料，BERT 不参与本次分流')
        elif callable(predict):
            # 规则判为通用知识时，仅允许 BERT 高置信“企业资料”做反向兜底；
            # 模型缺失、异常或置信度不足时保持规则结果。
            for text in [query] + ([anchor] if anchor else []):
                if needs_local: break
                try:
                    result = predict(text)
                    definition_only = bool(_DEFINITION_PATTERN.search(text)) and not (
                        _COMPANY_MARKERS.search(text) or _ENTERPRISE_TOPICS.search(text))
                    triggered = bool(
                        result.get('available') and result.get('label') == '企业资料'
                        and result.get('confidence', 0.0) >= 0.8 and not definition_only)
                    logger.info('BERT 意图分类 | 文本：%s | 标签：%s | 置信度：%.4f | 触发检索：%s',
                                text, result.get('label'), result.get('confidence', 0.0),
                                '是' if triggered else '否')
                    if triggered: needs_local = True
                except Exception:
                    logger.warning('BERT 意图分类失败，沿用规则判断', exc_info=True)
                    break
        logger.info('问题判断 | 追问：%s | 规则需要本地资料：%s | 最终需要本地资料：%s',
                    '是' if is_follow_up else '否', '是' if rule_needs_local else '否',
                    '是' if needs_local else '否')
        if not needs_local:
            # 普通知识问题不检索企业资料，避免无关文档被当作用户问题的依据。
            prompt = self.rag_prompt.format(
                context='（本次未检索知识库：该问题不涉及企业内部资料，没有可供引用的资料编号；'
                        '回答中不要输出 [1] 等引用编号）',
                history=history_context, question=query)
            return prompt, [], None
        try:
            documents = self.retrieve_and_merge(retrieval_query, source_filter=source_filter)
        except Exception as exc:
            logger.exception('知识检索失败')
            raise RetrievalError('知识库检索暂时失败，请确认向量库服务正常后重试。') from exc
        # 召回成功不等于与问题相关：分数过低时按未命中处理，避免用无关资料编造答案。
        if documents:
            documents = [doc for doc in documents
                         if _document_score(doc) >= conf.MIN_ANSWER_SCORE]
            if not documents:
                logger.info('检索到的资料相关度低于阈值 %.2f，按资料不足处理', conf.MIN_ANSWER_SCORE)
        if not documents:
            return None, [], ('当前知识库没有找到足够相关的资料，无法依据已导入的文件确认这一点。'
                              '我不会猜测内部人员、审批规则或联系方式。'
                              '请补充对应资料，或切换到包含该资料的分类后重试。')
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
        context_text = "\n\n".join(snippets)
        if selected:
            context_text = (f"可用引用编号：1 到 {len(selected)}；回答中不得使用该范围以外的编号。\n\n"
                            + context_text)
        prompt = self.rag_prompt.format(context=context_text, history=history_context,
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
            yield '以下回答未引用知识库资料，仅供参考：\n\n'
        if self.stream_llm:
            if cancel_event is None:
                chunks = self.stream_llm(prompt)
            else:
                chunks = self.stream_llm(prompt, cancel_event=cancel_event)
        else:
            chunks = (self.llm(prompt),)
        yield from _filter_citation_markers(chunks, len(documents))

    def generate_answer(self, query, source_filter=None, history=None):
        return ''.join(self.generate_answer_stream(query, source_filter, history))
