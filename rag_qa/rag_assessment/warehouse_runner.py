"""仓配 SaaS 场景的现场演示与回归评估：真实运行检索、回答和引用。

用法：
    python -m rag_qa.rag_assessment.warehouse_runner --mode all
    python -m rag_qa.rag_assessment.warehouse_runner --mode demo
    python -m rag_qa.rag_assessment.warehouse_runner --mode eval
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.config import config
from rag_qa.core.doc_metadata import version_match_detail
from rag_qa.core.triage import is_clarification_answer
from rag_qa.rag_assessment.rag_evaluator import assess_insufficient_answer

HERE = Path(__file__).with_name('warehouse_runner.py').parent
DEFAULT_EVAL_INPUT = HERE / 'warehouse_evaluate_data.json'
DEFAULT_DEMO_OUTPUT = HERE / 'warehouse_demo_results.json'
DEFAULT_EVAL_OUTPUT = HERE / 'warehouse_evaluate_results_v2.json'
PROJECT_NAME = '仓配 SaaS 实施与售后知识问答系统'

DEMO_CASES = [
    {
        'id': 'D1',
        'type': '能找到出处的常见故障',
        'question': '抖店电子面单取号失败，应该按什么顺序排查？',
        'filters': {'platform': '抖音电商'},
        'expected_docs': ['电子面单常见对接场景'],
    },
    {
        'id': 'D2-low',
        'type': '同一问题在不同版本中的处理差异（低版本）',
        'question': '客户是旺店通客户端 V2.4.4 及以下，能用 trade_push.php 创建原始订单接口吗？',
        'filters': {'product': '旺店通ERP', 'version': '客户端 V2.4.4 及以下'},
        'expected_docs': ['客户端 V2.4.4 及以下版本的接口限制（整理）'],
    },
    {
        'id': 'D2-supported',
        'type': '同一问题在不同版本中的处理差异（满足版本）',
        'question': '客户是旺店通客户端 V2.4.5.0，能用 trade_push.php 创建原始订单接口吗？',
        'filters': {'product': '旺店通ERP', 'version': '客户端 V2.4.5.0'},
        'expected_docs': ['trade_push.php（创建原始订单）'],
    },
    {
        'id': 'D3-ask',
        'type': '缺少客户现场信息时的追问',
        'question': '订单一直同步不过来',
        'filters': {},
        'expected_clarification': True,
        'session_group': 'D3',
    },
    {
        'id': 'D3-answer',
        'type': '补充平台和错误码后的回答',
        'question': '抖音店铺，ERP 提示错误码 100',
        'filters': {},
        'session_group': 'D3',
        'expected_docs': ['错误码说明'],
    },
    {
        'id': 'D4',
        'type': '资料不足时的说明',
        'question': '拼多多电子面单取号返回错误码 E202 是什么意思？',
        'filters': {'platform': '拼多多'},
        'expected_insufficient': True,
    },
]


def _save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        with open(temp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _clean_sources(sources):
    keys = ('index', 'title', 'subject', 'section', 'product', 'platform', 'version',
            'doc_type', 'source_url')
    return [{key: item.get(key) for key in keys if item.get(key) is not None} for item in sources]


def _backend_info(qa):
    """记录本次实际使用的检索后端，避免把原生混合检索和客户端回退混为一谈。"""
    describe = getattr(getattr(qa, 'vector_store', None), 'describe_status', None)
    if callable(describe):
        try:
            return describe()
        except Exception:
            return {}
    return {}


def _run_turn(qa, question, filters, session_id=None):
    sources = []
    parts = []
    started = time.time()
    for token, _complete in qa.query(
            question, session_id=session_id,
            product=filters.get('product'), platform=filters.get('platform'),
            version=filters.get('version'),
            source_callback=lambda items: sources.extend(items)):
        if token:
            parts.append(token)
    answer = ''.join(parts)
    return {
        'question': question,
        'filters': filters,
        'answer': answer,
        'sources': _clean_sources(sources),
        'citation_count': len(re.findall(r'\[\d+\]', answer)),
        'elapsed_seconds': round(time.time() - started, 2),
    }


def _doc_hit(sources, expected_docs):
    if not expected_docs:
        return None
    expected_docs = [str(expected).strip() for expected in expected_docs if str(expected).strip()]
    if not expected_docs:
        return None
    # 空标题不能因为 `'' in expected` 被算作命中。
    titles = [str(source.get('title') or '').strip() for source in sources]
    titles = [title for title in titles if title]
    if not titles:
        return False
    return any(expected in title or title in expected
               for expected in expected_docs for title in titles)


def _citation_check(answer, sources):
    """引用编号必须存在，并且都落在 1..来源数量 之内。"""
    numbers = [int(number) for number in re.findall(r'\[(\d+)\]', answer or '')]
    if not numbers:
        return {'pass': False, 'invalid': [], 'count': 0, 'reason': '回答没有引用编号'}
    invalid = sorted({number for number in numbers if not 1 <= number <= len(sources)})
    result = {'pass': not invalid, 'invalid': invalid, 'count': len(numbers)}
    if invalid:
        result['reason'] = f'引用编号越界：{invalid}，来源只有 {len(sources)} 个'
    return result


def _version_check(sources, version_filter):
    """版本检查：至少一个来源完全覆盖所选版本，且不能混入明确不适用或仅供对照的资料。"""
    if not version_filter:
        return {'applicable': False, 'pass': None, 'partial': [], 'statuses': []}
    if not sources:
        return {'applicable': True, 'pass': False, 'partial': [], 'statuses': [],
                'reason': '没有来源可用于核对版本'}
    statuses, partial = [], []
    for item in sources:
        detail = version_match_detail(item.get('version'), version_filter)
        statuses.append(detail['status'])
        if detail['status'] == 'partial':
            partial.append({'title': item.get('title'), 'version': item.get('version'),
                            'reason': detail['reason']})
    passed = (all(status not in ('no_match', 'reference', 'unknown') for status in statuses)
              and any(status == 'match' for status in statuses))
    result = {'applicable': True, 'pass': passed, 'partial': partial, 'statuses': statuses}
    if not passed:
        result['reason'] = '来源只有部分覆盖或仅有多版本对照资料'
    return result


def _version_pass(sources, version_filter):
    return _version_check(sources, version_filter)['pass']


def _conclusion_check(answer, case):
    """按结构化预期检查关键结论；关键字相同但结论相反的答案判为失败。"""
    expected = case.get('expected_conclusion')
    if not expected:
        return None
    if expected.get('manual_review'):
        return {'status': 'manual_review', 'reason': '结论无法稳定自动判断，需人工复核'}
    text = re.sub(r'\s+', '', answer or '')

    def _normalize(term):
        return re.sub(r'\s+', '', term or '')

    required = [term for term in expected.get('required_terms') or [] if _normalize(term)]
    required_any = [term for term in expected.get('required_any') or [] if _normalize(term)]
    forbidden = [term for term in expected.get('forbidden_terms') or [] if _normalize(term)]
    forbidden_patterns = [pattern for pattern in expected.get('forbidden_patterns') or [] if pattern]

    missing = [term for term in required if _normalize(term) not in text]
    if required_any and not any(_normalize(term) in text for term in required_any):
        missing.append(' / '.join(required_any))
    forbidden_hits = [term for term in forbidden if _normalize(term) in text]
    forbidden_hits += [pattern for pattern in forbidden_patterns
                       if re.search(pattern, answer or '', re.I)]
    status = 'passed' if not missing and not forbidden_hits else 'failed'
    return {'status': status, 'missing': missing, 'forbidden_hits': forbidden_hits}


def run_demo(qa):
    sessions = {}
    records = []
    for case in DEMO_CASES:
        group = case.get('session_group')
        session_id = sessions.setdefault(group, str(uuid.uuid4())) if group else str(uuid.uuid4())
        turn = _run_turn(qa, case['question'], case.get('filters') or {}, session_id=session_id)
        answer = turn['answer']
        record = dict(case, **turn)
        record['is_clarification'] = is_clarification_answer(answer)
        record['doc_hit'] = _doc_hit(turn['sources'], case.get('expected_docs'))
        record['version_filter_pass'] = _version_pass(turn['sources'], (case.get('filters') or {}).get('version'))
        if case.get('expected_insufficient'):
            status, reason = assess_insufficient_answer(answer)
            record['insufficient_status'] = status
            record['insufficient_reason'] = reason
        records.append(record)
        print(f"[demo] {case['id']} 完成：引用 {turn['citation_count']} 处，"
              f"来源 {len(turn['sources'])} 个，{turn['elapsed_seconds']}s", flush=True)
    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'project': PROJECT_NAME,
        'scenario': '国内仓配 SaaS 企业的实施与售后支持',
        'model': config.LLM_MODEL,
        'collection': config.MILVUS_COLLECTION_NAME,
        'backend': _backend_info(qa),
        'demo_cases': records,
    }


def _evaluate_record(record, case):
    answer = record['answer']
    filters = case.get('filters') or {}
    checks = {'doc_hit': _doc_hit(record['sources'], case.get('expected_docs'))}
    citation = _citation_check(answer, record['sources'])
    checks['citation_pass'] = citation['pass']
    checks['citation_invalid'] = citation['invalid']
    checks['citation_count'] = citation['count']
    # 没有设置版本筛选时，版本检查是“不适用”，既不算通过也不算失败。
    version = _version_check(record['sources'], filters.get('version'))
    checks['version_filter_applicable'] = version['applicable']
    checks['version_filter_pass'] = version['pass']
    checks['version_partial_sources'] = version.get('partial', [])
    conclusion = _conclusion_check(answer, case)
    checks['conclusion_status'] = conclusion['status'] if conclusion else None
    checks['conclusion_detail'] = conclusion
    if case.get('expected_clarification'):
        checks['clarification_pass'] = is_clarification_answer(answer)
        checks['doc_hit'] = None
        checks['citation_pass'] = None
        checks['citation_invalid'] = []
    if case.get('expected_insufficient'):
        status, reason = assess_insufficient_answer(answer)
        checks['insufficient_status'] = status
        checks['insufficient_reason'] = reason
        checks['citation_pass'] = None
        checks['citation_invalid'] = []
    keywords = case.get('expected_keywords') or []
    matched = [word for word in keywords if word in answer]
    checks['keyword_matches'] = matched
    checks['keyword_ratio'] = round(len(matched) / len(keywords), 3) if keywords else None
    if case.get('expected_clarification'):
        checks['passed'] = bool(checks.get('clarification_pass'))
    elif case.get('expected_insufficient'):
        checks['passed'] = checks.get('insufficient_status') == 'passed'
    else:
        version_ok = (not checks.get('version_filter_applicable')
                      or checks.get('version_filter_pass') is True)
        conclusion_ok = checks.get('conclusion_status') in (None, 'passed', 'manual_review')
        checks['passed'] = bool(checks.get('doc_hit')) and bool(checks.get('citation_pass')) \
            and (checks.get('keyword_ratio') or 0) >= 0.5 \
            and version_ok and conclusion_ok
    failures = []
    if not checks['passed']:
        if checks.get('doc_hit') is False:
            failures.append('召回未命中预期文档（或来源标题为空）')
        if checks.get('version_filter_pass') is False:
            failures.append('版本筛选后混入了不适用版本的资料，或只有部分覆盖')
        if checks.get('citation_pass') is False:
            failures.append('回答没有引用编号' if not checks.get('citation_invalid')
                            else f"引用编号越界：{checks['citation_invalid']}")
        if checks.get('conclusion_status') == 'failed':
            detail = checks.get('conclusion_detail') or {}
            failures.append('关键结论与预期相反或缺少必要结论：'
                            f"缺少 {detail.get('missing') or []}，"
                            f"出现相反表述 {detail.get('forbidden_hits') or []}")
        if case.get('expected_clarification') and not checks.get('clarification_pass'):
            failures.append('没有按预期先追问')
        if case.get('expected_insufficient') and checks.get('insufficient_status') != 'passed':
            failures.append('资料不足处理不合格')
        if not failures:
            failures.append('答案要点覆盖不足')
    checks['failure_reasons'] = failures
    return checks


def run_evaluation(qa, input_path=DEFAULT_EVAL_INPUT):
    cases = json.loads(Path(input_path).read_text(encoding='utf-8'))
    records = []
    for case in cases:
        turn = _run_turn(qa, case['question'], case.get('filters') or {})
        checks = _evaluate_record(turn, case)
        record = dict(case, **turn, checks=checks)
        records.append(record)
        print(f"[eval] {case['id']} {'通过' if checks['passed'] else '未通过'}"
              f"（引用 {turn['citation_count']}，来源 {len(turn['sources'])}）", flush=True)

    normal = [record for record in records if not record.get('expected_clarification')
              and not record.get('expected_insufficient')]

    def _applicable(records, key):
        return [record for record in records if record['checks'].get(key) is not None]

    doc_hit = _applicable(normal, 'doc_hit')
    version = _applicable(normal, 'version_filter_pass')
    citation = _applicable(normal, 'citation_pass')
    conclusion = [record for record in normal if record['checks'].get('conclusion_status')]
    summary = {
        'total': len(records),
        'passed': sum(1 for record in records if record['checks']['passed']),
        'normal_total': len(normal),
        # 每个指标单独给出适用题目数和通过数；不适用的题目不进入分母。
        'normal_doc_hit_applicable': len(doc_hit),
        'normal_doc_hit_passed': sum(1 for record in doc_hit if record['checks'].get('doc_hit')),
        'normal_version_applicable': len(version),
        'normal_version_passed': sum(1 for record in version
                                     if record['checks'].get('version_filter_pass')),
        'normal_version_not_applicable': len(normal) - len(version),
        'normal_version_partial': sum(1 for record in normal
                                      if record['checks'].get('version_partial_sources')),
        'normal_citation_applicable': len(citation),
        'normal_citation_passed': sum(1 for record in citation
                                      if record['checks'].get('citation_pass')),
        'normal_citation_invalid': sum(1 for record in citation
                                       if record['checks'].get('citation_invalid')),
        'normal_conclusion_applicable': len(conclusion),
        'normal_conclusion_passed': sum(1 for record in conclusion
                                        if record['checks'].get('conclusion_status') == 'passed'),
        'normal_conclusion_failed': sum(1 for record in conclusion
                                        if record['checks'].get('conclusion_status') == 'failed'),
        'normal_conclusion_manual_review': sum(
            1 for record in conclusion
            if record['checks'].get('conclusion_status') == 'manual_review'),
        'clarification_total': sum(1 for record in records if record.get('expected_clarification')),
        'clarification_passed': sum(1 for record in records if record.get('expected_clarification')
                                    and record['checks']['passed']),
        'insufficient_total': sum(1 for record in records if record.get('expected_insufficient')),
        'insufficient_passed': sum(1 for record in records if record.get('expected_insufficient')
                                   and record['checks']['passed']),
    }
    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'project': PROJECT_NAME,
        'scenario': '国内仓配 SaaS 企业的实施与售后支持',
        'model': config.LLM_MODEL,
        'collection': config.MILVUS_COLLECTION_NAME,
        'backend': _backend_info(qa),
        'summary': summary,
        'records': records,
    }


def main():
    parser = argparse.ArgumentParser(description='仓配 SaaS 场景演示与评估')
    parser.add_argument('--mode', choices=('demo', 'eval', 'all'), default='all')
    parser.add_argument('--input', type=Path, default=DEFAULT_EVAL_INPUT)
    parser.add_argument('--demo-output', type=Path, default=DEFAULT_DEMO_OUTPUT)
    parser.add_argument('--eval-output', type=Path, default=DEFAULT_EVAL_OUTPUT)
    args = parser.parse_args()

    from cli import IntegratedQASystem
    qa = IntegratedQASystem()
    try:
        if qa.rag_system is None:
            raise RuntimeError('RAG 未就绪，请检查本地模型与 Milvus 服务。')
        if not config.LLM_API_KEY:
            raise RuntimeError('未配置模型 API 密钥。')
        if args.mode in ('demo', 'all'):
            data = run_demo(qa)
            _save_json(args.demo_output, data)
            print(f'演示结果：{args.demo_output}')
        if args.mode in ('eval', 'all'):
            data = run_evaluation(qa, args.input)
            _save_json(args.eval_output, data)
            print('评估汇总：', json.dumps(data['summary'], ensure_ascii=False))
            print(f'评估结果：{args.eval_output}')
    finally:
        qa.close()


if __name__ == '__main__':
    main()
