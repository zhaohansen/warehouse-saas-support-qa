"""公开资料元数据：解析文档头，并按产品/平台/版本判断适用范围。"""

import re


HEADER_START = '【文档元数据】'
BODY_START = '【正文】'

FIELD_MAP = {
    '文档标题': 'doc_title',
    '章节': 'section',
    '产品': 'product',
    '平台': 'platform',
    '适用版本': 'version',
    '文档类型': 'doc_type',
    '原文链接': 'source_url',
    '更新时间': 'updated_at',
    '授权状态': 'authorization',
}

# 会写入向量库并在引用中展示的字段。
INDEX_FIELDS = ('doc_title', 'section', 'product', 'platform', 'version', 'doc_type',
                'source_url', 'updated_at', 'authorization')


def parse_metadata_header(text):
    """拆分“【文档元数据】”头与正文；没有头时返回空元数据和原文。"""
    text = (text or '').lstrip('\ufeff')
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == HEADER_START)
    except StopIteration:
        return {}, text

    metadata = {}
    body_start = None
    for index in range(start + 1, len(lines)):
        line = lines[index].strip()
        if not line:
            continue
        if line == BODY_START:
            body_start = index + 1
            break
        match = re.match(r'^([^:：]{1,20})[:：]\s*(.*)$', line)
        if not match:
            body_start = index
            break
        key, value = match.group(1).strip(), match.group(2).strip()
        field = FIELD_MAP.get(key)
        if field and value:
            metadata[field] = value
    if body_start is None:
        body_start = len(lines)
    body = '\n'.join(lines[body_start:]).strip()
    return metadata, body


VERSION_RE = re.compile(r'[Vv](\d+(?:\.\d+){0,3})')
_VERSION_NUMBER_RE = re.compile(r'\d+(?:\.\d+){0,3}')


def parse_version(text):
    """把 V2.4.5 / V2.4.5.0 统一解析成补零到 4 段的元组；无法解析时返回 None。"""
    match = VERSION_RE.search(text or '')
    if match:
        raw = match.group(1)
    else:
        # 兼容外部已经去掉 V 前缀的版本号，例如 parse_version('2.4.5')。
        fallback = _VERSION_NUMBER_RE.search(text or '')
        if not fallback:
            return None
        raw = fallback.group(0)
    parts = [int(part) for part in raw.split('.')]
    return tuple((parts + [0, 0, 0, 0])[:4])


def parse_version_selection(selected):
    """区分用户选择的是网页版、平台版本、具体版本还是版本范围。"""
    selected = (selected or '').strip()
    if '网页版' in selected:
        return {'kind': 'web', 'version': None}
    if '平台' in selected:
        return {'kind': 'platform', 'version': None}
    match = VERSION_RE.search(selected)
    if not match:
        return {'kind': 'unknown', 'version': None}
    version = parse_version(match.group(1))
    if '及以上' in selected:
        return {'kind': 'at_least', 'version': version}
    if '及以下' in selected:
        return {'kind': 'at_most', 'version': version}
    return {'kind': 'exact', 'version': version}


def _label_ranges(label):
    """把资料标签解析成版本区间列表 (lower, upper)，单点版本 lower == upper。"""
    ranges = []
    for version, bound in re.findall(r'[Vv](\d+(?:\.\d+){0,3})\s*(及以上|及以下)?', label):
        value = parse_version(version)
        if bound == '及以上':
            ranges.append((value, None))
        elif bound == '及以下':
            ranges.append((None, value))
        else:
            ranges.append((value, value))
    return ranges


def _selection_range(spec):
    kind, version = spec['kind'], spec['version']
    if kind == 'exact':
        return version, version
    if kind == 'at_least':
        return version, None
    if kind == 'at_most':
        return None, version
    return None


def _overlap(doc_range, selected_range):
    doc_low, doc_high = doc_range
    sel_low, sel_high = selected_range
    if doc_low is not None and sel_high is not None and doc_low > sel_high:
        return False
    if sel_low is not None and doc_high is not None and sel_low > doc_high:
        return False
    return True


def _covers(doc_range, selected_range):
    """判断资料区间是否覆盖用户选择的全部版本。"""
    doc_low, doc_high = doc_range
    sel_low, sel_high = selected_range
    # 用户选择的下界无界（如“V2.4.4 及以下”）时，资料也必须无下界才算完全覆盖。
    if sel_low is None:
        lower_ok = doc_low is None
    else:
        lower_ok = doc_low is None or doc_low <= sel_low
    if sel_high is None:
        upper_ok = doc_high is None
    else:
        upper_ok = doc_high is None or sel_high <= doc_high
    return lower_ok and upper_ok


def version_match_detail(label, selected):
    """返回资料版本与用户选择的匹配细节。

    status 取值：
    - match：资料覆盖用户选择的全部版本；
    - partial：只覆盖一部分，回答必须说明附加条件；
    - reference：多版本对照，只能用于比较，不能证明每条规则逐条适用；
    - no_match：明确不覆盖；
    - unknown：资料没有版本信息或无法解析。
    """
    label = (label or '').strip()
    selected = (selected or '').strip()
    if not selected:
        return {'status': 'match', 'reason': '未设置版本筛选'}
    if not label:
        return {'status': 'unknown', 'reason': '资料没有适用版本信息'}
    if '多版本' in label or '对照' in label:
        return {'status': 'reference',
                'reason': '多版本对照资料只能用于比较，不能证明每条规则适用于所选版本'}
    if any(marker in label for marker in ('全版本', '通用')):
        return {'status': 'match', 'reason': '资料标注为全版本/通用'}
    spec = parse_version_selection(selected)
    if spec['kind'] == 'web':
        if '网页版' in label:
            return {'status': 'match', 'reason': '网页版资料'}
        return {'status': 'no_match', 'reason': '非网页版资料'}
    if spec['kind'] == 'platform':
        if '平台' in label or '最新' in label:
            return {'status': 'match', 'reason': '平台版本文档'}
        return {'status': 'no_match', 'reason': '非平台版本文档'}
    if spec['version'] is None:
        return {'status': 'unknown', 'reason': '无法解析用户版本'}
    # 网页版和平台版本文档不能用来回答某个客户端版本的问题。
    if '网页版' in label:
        return {'status': 'no_match', 'reason': '网页版资料不适用于客户端版本'}
    if '平台' in label or '最新' in label:
        return {'status': 'no_match', 'reason': '平台版本文档不适用于客户端版本'}
    ranges = _label_ranges(label)
    if not ranges:
        return {'status': 'unknown', 'reason': '资料没有可比较的版本号'}
    selected_range = _selection_range(spec)
    if any(_covers(item, selected_range) for item in ranges):
        return {'status': 'match', 'reason': '资料覆盖所选版本范围'}
    if any(_overlap(item, selected_range) for item in ranges):
        return {'status': 'partial',
                'reason': f'资料只覆盖“{label}”，与所选“{selected}”部分重叠'}
    return {'status': 'no_match', 'reason': f'资料适用“{label}”，不覆盖所选版本'}


def version_matches(label, selected):
    """兼容旧调用：完全覆盖或部分覆盖都视为可进入候选集。"""
    return version_match_detail(label, selected)['status'] in ('match', 'partial')


def filter_documents_by_version(documents, selected):
    """版本后置过滤：保留完全覆盖和部分覆盖的资料，并写入匹配状态。

    多版本对照资料不进入版本筛选后的上下文，避免把“可用于比较”当成“逐条适用”。
    """
    kept = []
    for doc in documents:
        detail = version_match_detail(doc.metadata.get('version'), selected)
        if detail['status'] not in ('match', 'partial'):
            continue
        doc.metadata['version_match'] = detail['status']
        doc.metadata['version_match_reason'] = detail['reason']
        kept.append(doc)
    return kept


def build_filter_expression(source_filter=None, filters=None, include_platform=True):
    """生成 Milvus 过滤表达式；产品/平台下推，版本由 version_matches 后置判断。"""
    filters = filters or {}
    parts = []
    if source_filter:
        parts.append('source == ' + _quote(source_filter))
    product = (filters.get('product') or '').strip()
    if product:
        parts.append('product in ' + _quote_list([product, '通用']))
    platform = (filters.get('platform') or '').strip()
    if platform and include_platform:
        # 公共错误码等资料平台为“通用”，筛选具体平台时仍应保留。
        parts.append('platform in ' + _quote_list([platform, '通用']))
    return ' and '.join(parts)


def _quote(value):
    import json
    return json.dumps(value, ensure_ascii=False)


def _quote_list(values):
    import json
    return json.dumps(values, ensure_ascii=False)
