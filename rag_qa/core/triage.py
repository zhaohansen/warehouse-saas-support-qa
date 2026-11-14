"""售后初判：问题缺少区分条件时，先问一个最具体的问题。"""

import re

PLATFORM_WORDS = ('抖音', '抖店', '拼多多', '淘宝', '天猫', '京东', '快手', '小红书', '有赞',
                  '唯品会', '视频号', '自有商城', '自有平台')

ORDER_WORDS = ('订单', '同步', '漏单', '抓单', '下载单', '推送单', '进不来', '没进来')
INVENTORY_WORDS = ('库存', '可售', '可用库存', '锁库存', '锁定库存', '在途', '超卖', '少卖')
WAYBILL_WORDS = ('面单', '电子面单', '取号', '打印', '运单', '物流回传', '发货')

# 只有给出了可核对的错误值、报错原文或明确状态，才算“有报错详情”。
# 单独出现“报错”“错误码”只是问题描述，不能当作用户已经提供排查线索。
ERROR_PATTERN = re.compile(
    r'(错误码|错误编号|err(?:or)?\s*code|code)\s*(?:为|是|[:：=])?\s*[A-Za-z]?\d{2,}'
    r'|(?:报错|提示|状态码)\s*(?:为|是|[:：=])\s*[^\s，。；,;]{1,30}'
    r'|[A-Z]{2,}[-_]?\d{2,}'
    r'|未找到|不存在|无权限|权限不足|超时|频率限制|请求过于频繁|为空|非法|不符合|失败原因'
    r'|(?<!\d)\d{3,}(?!\d)',
    re.I)
# 用户明确看不到报错时，不再重复追问同一条信息。
ERROR_UNAVAILABLE_PATTERN = re.compile(
    r'(看不到|看不见|没看到|无法查看|无法看到|没有|没显示|未显示|不显示|查不到|找不到)'
    r'(?:显示|提示|具体|的){0,2}(?:报错|错误码|错误信息|失败原因)'
    r'|(?:报错|错误码|错误信息|失败原因)(?:也|都)?'
    r'(?:看不到|看不见|无法查看|无法看到|没显示|未显示|不显示|查不到|找不到)',
    re.I)
VERSION_PATTERN = re.compile(r'[Vv]\d+(?:\.\d+){0,3}')
DIFFERENCE_WORDS = ('差异', '不一致', '对不上', '不同步', '少了', '多了', '不准', '超卖')


def _has_any(text, words):
    return any(word in text for word in words)


def _platform(query, filters):
    return (_has_any(query, PLATFORM_WORDS)
            or bool((filters or {}).get('platform')))


def _error_detail(query):
    """有具体错误码/报错原文/订单状态，才算能区分情况。"""
    if ERROR_PATTERN.search(query):
        return True
    return _has_any(query, ('待付款', '已付款', '待审核', '已审核', '已发货', '已关闭',
                            '订单号', '部分同步', '全部同步'))


def _error_unavailable(query):
    """用户明确表示看不到报错/错误码。"""
    return bool(ERROR_UNAVAILABLE_PATTERN.search(query))


def _domain(query):
    # 库存、面单是更具体的业务对象，优先于泛化的“同步”等订单词。
    if _has_any(query, INVENTORY_WORDS) and not _has_any(query, ('订单', '抓单', '漏单')):
        return 'inventory'
    if _has_any(query, WAYBILL_WORDS) or '物流' in query:
        return 'waybill'
    scores = {
        'order': sum(1 for word in ORDER_WORDS if word in query),
        'inventory': sum(1 for word in INVENTORY_WORDS if word in query),
        'waybill': sum(1 for word in WAYBILL_WORDS if word in query),
    }
    domain, score = max(scores.items(), key=lambda item: item[1])
    return domain if score else None


def _is_knowledge_question(query):
    """问后果、区别、定义、常见原因或准备流程时，问题本身已足够明确，不应按现场故障追问。"""
    return _has_any(query, ('会有什么后果', '有什么影响', '有什么区别', '是什么', '为什么',
                            '会怎样', '准备', '前置', '开通条件', '接入前', '流程',
                            '步骤', '顺序', '怎么接入', '如何接入', '常见原因', '有哪些原因',
                            '一般原因', '排查思路', '排查方法', '如何排查', '怎么排查'))


def _inventory_kind(query):
    return _has_any(query, ('可用库存', '可售库存', '锁定库存', '锁定量', '在途', '库存量',
                            '盘点', '入库', '出库', '调拨', '退货'))


def _waybill_stage(query):
    return _has_any(query, ('取号', '打印', '回传', '发货'))


def _interface_version_question(query):
    """接口适用版本/兼容性问题已有足够区分条件，不应按订单缺平台追问。"""
    if not _has_any(query, ('接口', 'API', 'api')):
        return False
    if ERROR_PATTERN.search(query):
        return False
    return bool(VERSION_PATTERN.search(query)
                or _has_any(query, ('适用版本', '版本', '兼容', '支持', '能用', '可用', '限制', '区别')))


def is_clarification_reply(query):
    """判断这句话是否在回答上一轮的追问，而不是另起一个新问题。"""
    text = (query or '').strip()
    if not text or len(text) > 40:
        return False
    if ERROR_PATTERN.search(text) or VERSION_PATTERN.search(text) or _error_unavailable(text):
        return True
    if _has_any(text, PLATFORM_WORDS):
        return True
    if _has_any(text, ('可用库存', '锁定库存', '在途库存', '盘点')):
        return True
    return text.startswith(('是', '用的', '平台是', '来自')) and _has_any(
        text, ('取号', '打印', '回传'))


def needs_clarification(query, filters=None):
    """返回追问信息；不需要追问时返回 None。

    只在问题落在三个业务域、且缺少能区分处理路径的关键信息时触发。
    """
    query = (query or '').strip()
    filters = filters or {}
    if len(query) < 3:
        return None
    domain = _domain(query)
    if not domain:
        return None
    if _interface_version_question(query) or _is_knowledge_question(query):
        return None
    platform = _platform(query, filters)

    if domain == 'order':
        detail = _error_detail(query)
        unavailable = _error_unavailable(query)
        if platform and (detail or unavailable):
            return None
        missing = []
        if not platform:
            missing.append('平台/店铺')
        if not detail and not unavailable:
            missing.append('报错原文或错误码')
        question = '请先确认一个信息：这条订单来自哪个平台/店铺，以及 ERP 里看到的报错原文或错误码是什么？'
        if platform and not detail and not unavailable:
            question = ('请补充 ERP 里看到的报错原文或错误码：是“未找到符合条件的订单”、'
                        '错误码 100，还是其他提示？')
        elif (detail or unavailable) and not platform:
            question = '请确认这条订单来自哪个平台/店铺，例如抖音、拼多多、淘宝天猫或自有商城？'
        return {'domain': '订单同步', 'missing': missing, 'question': question}

    if domain == 'inventory':
        if not _has_any(query, DIFFERENCE_WORDS) and not _inventory_kind(query):
            return None
        if _inventory_kind(query):
            return None
        return {'domain': '库存差异', 'missing': ['差异类型或发生环节'],
                'question': '请先确认差异属于哪一类：可用库存、锁定库存，还是在途库存？'}

    if domain == 'waybill':
        if _waybill_stage(query) and platform:
            return None
        missing = []
        if not platform:
            missing.append('平台')
        if not _waybill_stage(query):
            missing.append('异常阶段')
        question = '请先确认异常发生在哪个阶段：取号失败、打印失败，还是发货后物流不回传？'
        if not platform and _waybill_stage(query):
            question = '请确认使用的是哪个平台的电子面单：抖音、拼多多，还是淘宝天猫？'
        elif not platform and not _waybill_stage(query):
            question = '请先确认平台和异常阶段：是抖音、拼多多还是淘宝天猫，卡在取号、打印还是物流回传？'
        return {'domain': '发货/面单异常', 'missing': missing, 'question': question}

    return None


def is_clarification_answer(text):
    text = text or ''
    return text.rstrip().endswith('？') and any(
        marker in text for marker in ('请先确认', '请补充', '请确认'))
