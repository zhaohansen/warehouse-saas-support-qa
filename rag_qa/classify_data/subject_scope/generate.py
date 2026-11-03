#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成「知序·学科知识助手」BERT 二分类数据集。

标签按知识领域划分（不看是否提到课件/讲义）：
    学科资料  Python / Java / AI / MySQL / Redis 五个学科的问题
    通用知识  其余领域的问题

规模：5000 条，两类各 2500；五个学科各 500。

用法：
    python generate.py
    python generate.py --out ../subject_scope_5000.json --seed 20261111
"""

import argparse
import json
import os
import random
import re
from collections import Counter

from seeds_python import PYTHON
from seeds_java import JAVA
from seeds_ai import AI
from seeds_mysql import MYSQL
from seeds_redis import REDIS
from seeds_general_a import GENERAL_A
from seeds_general_b import GENERAL_B

HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECTS = [("Python", PYTHON), ("Java", JAVA), ("AI", AI), ("MySQL", MYSQL), ("Redis", REDIS)]
GENERAL = {}
GENERAL.update(GENERAL_A)
GENERAL.update(GENERAL_B)

LABEL_SUBJECT = "学科资料"
LABEL_GENERAL = "通用知识"
PER_SUBJECT = 500
PER_GENERAL = 2500

# ---------------------------------------------------------------- 学科问法
# 每个类型 5 组角度，每个知识点取其中 4 组，避免同类知识点问法完全一致。
SUBJECT_TEMPLATES = {
    "c": [
        ["{subj} 的{s}是什么？", "解释一下 {subj} 的{s}", "{subj} 里的{s}具体指什么"],
        ["{subj} 的{s}解决了什么问题？", "{subj} 里为什么要用{s}", "{subj} 的{s}存在的意义是什么"],
        ["{subj} 的{s}是怎么工作的？", "{subj} 的{s}原理是什么", "{subj} 的{s}工作机制是怎样的"],
        ["{subj} 的{s}什么时候需要用到？", "什么情况下会用到 {subj} 的{s}", "{subj} 的{s}在实际项目里常见吗"],
        ["{subj} 的{s}在实际使用中要注意什么", "用 {subj} 的{s}容易踩哪些坑", "{subj} 的{s}有什么局限"],
    ],
    "issue": [
        ["{subj} 的{s}是什么？", "解释一下 {subj} 的{s}", "{subj} 里的{s}具体指什么"],
        ["{subj} 里为什么会出现{s}？", "{s}是怎么产生的", "{subj} 的{s}是什么原因导致的"],
        ["{s}会带来什么问题", "{subj} 的{s}影响大吗", "{s}有什么后果"],
        ["怎么避免 {subj} 的{s}", "{subj} 里怎么解决{s}", "{s}能彻底解决吗"],
    ],
    "w": [
        ["在 {subj} 里，为什么{p}？", "{p}，原因是什么", "为什么{p}"],
        ["{p}，背后的机制是什么", "{p}是怎么发生的", "从原理上解释一下，{p}"],
        ["{p}会带来什么影响", "{p}有什么后果", "{p}对性能有什么影响"],
        ["{p}，有没有例外", "{p}，所有情况都成立吗", "{p}，这个结论绝对吗"],
        ["怎么验证{p}", "怎么观察{p}", "{p}，怎么证明"],
    ],
    "h": [
        ["{subj} 里{v}怎么做？", "{subj} 里{v}怎么实现", "如何{v}"],
        ["除了常规写法，{subj} 里{v}还有别的思路吗", "{subj} 里{v}有几种写法", "{v}还有更简洁的实现吗"],
        ["{subj} 里{v}要注意什么", "{subj} 里{v}容易踩哪些坑", "{v}有什么隐患"],
        ["{v}还有别的做法吗", "不用这种方式，{v}还能怎么做", "{v}有没有更简单的办法"],
        ["{v}的最佳实践是什么", "生产环境里{v}推荐怎么做", "{subj} 里{v}的规范写法是什么"],
    ],
    "d": [
        ["{subj} 的{a}和{b}有什么区别？", "{subj} 里{a}和{b}有什么不同", "对比一下 {subj} 的{a}和{b}"],
        ["什么时候用{a}，什么时候用{b}", "{subj} 里{a}和{b}该怎么选", "实际项目里选{a}还是{b}"],
        ["{a}能替代{b}吗", "有了{a}还需要{b}吗", "{a}和{b}可以互换吗"],
        ["{a}和{b}在实际使用中有什么差异", "{a}和{b}哪个更常用", "{a}和{b}的取舍是什么"],
        ["{a}和{b}分别适合什么场景", "{subj} 里{a}和{b}的使用场景有什么不同", "{a}和{b}各自用在哪里"],
    ],
    "p": [
        ["在 {subj} 里，为什么{p}？", "{p}，这是为什么", "为什么在 {subj} 里{p}"],
        ["{p}，还有哪些类似的优化点", "{p}，还能怎么优化", "{p} 有哪些优化空间"],
        ["{p}，优化的代价是什么", "{p}，优化会不会带来其他问题", "{p} 的取舍在哪里"],
        ["{p}，什么情况下值得优化", "什么情况下需要关注{p}", "{p} 什么时候会成为瓶颈"],
        ["{p}，优化效果怎么衡量", "{p} 用什么指标评估", "{p} 优化前后怎么对比"],
    ],
    "e": [
        ["{err} 怎么解决？", "遇到 {err} 怎么办", "{err} 的解决办法"],
        ["{err} 是什么原因", "为什么会出现 {err}", "{err} 是怎么引起的"],
        ["{err} 的排查思路是什么", "{err} 怎么定位问题", "遇到 {err} 从哪里开始查"],
        ["{err} 怎么避免", "怎么预防 {err}", "{err} 以后还会再出现吗"],
    ],
}

# 现象/故障类知识点：问"为什么出现、有什么影响、怎么解决"，而不是"用在什么场景"。
ISSUE_WORDS = (
    "脏读", "不可重复读", "幻读", "死锁", "长事务", "内存泄漏", "缓存穿透", "缓存击穿",
    "缓存雪崩", "脑裂", "数据倾斜", "内存碎片", "锁等待", "主从延迟", "缓冲区溢出",
    "依赖冲突", "线程安全", "毛刺", "泄漏", "阻塞命令", "持久化阻塞", "超时", "溢出",
    "冲突", "失效", "损坏", "限制", "过拟合", "欠拟合", "梯度消失", "梯度爆炸",
    "类别不平衡", "幻觉", "倾斜", "抖动", "场景", "丢失",
)

# ---------------------------------------------------------------- 通用问法
GENERAL_TEMPLATES = {
    "meta": [
        ["{t}是什么？", "{t}具体指什么", "解释一下{t}"],
        ["{t}为什么重要？", "{t}解决了什么问题", "{t}的意义在哪里"],
        ["怎么理解{t}？", "{t}能通俗解释一下吗", "怎么看待{t}"],
        ["关于{t}常见的误解有哪些？", "学{t}最容易卡在哪里", "{t}有什么容易混淆的地方"],
    ],
    "num": [
        ["{t}是多少？", "{t}一般是多少", "{t}大概是什么数值"],
        ["{t}大概在什么范围？", "{t}的正常范围是多少", "{t}一般控制在多少"],
        ["{t}受什么因素影响？", "哪些因素会影响{t}", "{t}为什么会变化"],
        ["{t}怎么测量？", "{t}怎么计算", "{t}用什么方法测"],
    ],
    "formula": [
        ["{t}怎么表示？", "{t}的表达式是什么", "{t}怎么写出来"],
        ["{t}怎么推导？", "{t}是怎么得出来的", "{t}的推导过程是什么"],
        ["{t}的适用条件是什么？", "{t}在什么条件下成立", "{t}有什么前提"],
        ["{t}在生活中有哪些应用？", "{t}一般用在哪里", "{t}有什么实际用途"],
    ],
    "event": [
        ["{t}发生在什么时期？", "{t}是哪一年的事", "{t}的时间线是怎样的"],
        ["{t}的背景是什么？", "{t}为什么会发生", "{t}是在什么背景下出现的"],
        ["{t}产生了什么影响？", "{t}带来了哪些变化", "{t}的后果是什么"],
        ["怎么评价{t}？", "{t}的历史意义是什么", "{t}有什么争议"],
    ],
    "rule": [
        ["{t}是什么？", "{t}具体怎么规定的", "{t}有哪些要求"],
        ["{t}具体怎么判？", "{t}的判定标准是什么", "{t}怎么算违规"],
        ["{t}有什么例外情况？", "{t}有哪些特殊情况", "{t}有没有例外"],
        ["{t}要注意什么？", "{t}容易出错的地方在哪", "执行{t}的时候要注意哪些细节"],
    ],
    "symptom": [
        ["{t}有哪些表现？", "{t}有什么症状", "{t}一般是什么样子"],
        ["{t}的早期信号是什么？", "{t}早期有什么征兆", "{t}刚开始会有什么感觉"],
        ["{t}是怎么引起的？", "{t}的常见原因有哪些", "什么原因会引起{t}"],
        ["{t}需要做什么检查？", "{t}一般怎么诊断", "{t}怎么确诊"],
    ],
    "method": [
        ["{t}是什么？", "{t}的正确做法是什么", "{t}一般怎么操作"],
        ["{t}有哪些步骤？", "{t}的流程是什么", "{t}分几步完成"],
        ["{t}有什么技巧？", "{t}有什么窍门", "{t}怎么做得更好"],
        ["{t}要注意什么？", "{t}容易忽略什么", "{t}有哪些坑"],
    ],
    "mechanism": [
        ["{t}的原理是什么？", "{t}是怎么工作的", "{t}的工作原理是什么"],
        ["{t}的关键环节是什么？", "{t}包括哪些步骤", "{t}的流程是什么"],
        ["{t}需要什么条件？", "{t}受什么因素影响", "{t}为什么能实现"],
        ["{t}有什么局限？", "{t}在什么情况下不适用", "{t}有什么代价"],
    ],
    "formation": [
        ["{t}是怎么形成的？", "{t}的形成原因是什么", "为什么会出现{t}"],
        ["{t}的形成需要什么条件？", "形成{t}需要哪些条件", "{t}受什么因素影响"],
        ["{t}的形成过程是怎样的？", "{t}经历哪些阶段", "{t}是怎么一步步形成的"],
        ["{t}常见吗？", "{t}一般发生在什么情况下", "{t}有什么规律"],
    ],
    "category": [
        ["{t}有哪些种类？", "{t}包括哪几种", "{t}主要分几类"],
        ["{t}按什么标准分类？", "{t}的分类依据是什么", "{t}怎么划分"],
        ["{t}中哪一类最常见？", "{t}里哪种用得最多", "{t}各自有什么特点"],
        ["不同种类的{t}怎么区分？", "{t}之间有什么区别", "怎么分辨{t}"],
    ],
    "property": [
        ["{t}有什么特点？", "{t}的主要特征是什么", "{t}有什么性质"],
        ["{t}有什么规律？", "{t}一般呈现什么趋势", "{t}的变化特点是什么"],
        ["{t}有什么实际用途？", "{t}一般用在哪些地方", "{t}有什么价值"],
        ["{t}容易和什么混淆？", "{t}有什么容易搞错的地方", "学{t}要注意什么"],
    ],
}

SUFFIX_RULES = [
    ("的工作原理", "mechanism"),
    ("的原理", "mechanism"),
    ("的症状", "symptom"),
    ("的表现", "symptom"),
    ("的形成", "formation"),
    ("的成因", "formation"),
    ("的原因", "formation"),
    ("的分类", "category"),
    ("的种类", "category"),
    ("的类型", "category"),
    ("的特点", "property"),
    ("的特征", "property"),
    ("的性质", "property"),
]

# 主题分类：命中即用对应问法组，未命中走 meta。
TOPIC_RULES = [
    ("formula", ("公式", "方程", "定律", "定理")),
    ("symptom", ("症状", "表现", "征兆")),
    ("rule", ("规则", "违例", "判定", "规定", "赛制", "计分", "级别", "段位")),
    ("event", ("战争", "革命", "变法", "改革", "统一", "开辟", "开通", "成立",
               "起义", "条约", "会议", "王朝", "帝国", "文明", "维新")),
    ("method", ("方法", "清洗", "去除", "处理", "清理", "技巧",
                "操作", "配制", "步骤", "保存", "挑选", "辨别")),
    ("num", ("温度", "含量", "寿命", "用水量", "烟点", "距离", "比例",
             "系数", "浓度", "速度", "重量", "高度", "面积", "体积", "时长",
             "年龄", "年份", "人数", "次数", "频率", "强度", "电压", "功率",
             "环数", "杆数", "时间", "日期")),
]

# ---------------------------------------------------------------- 去重
PUNCT = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9+#]+")


def normalize(text):
    return PUNCT.sub("", text).lower()


def trigrams(text):
    s = normalize(text)
    if len(s) <= 3:
        return {s}
    return {s[i:i + 3] for i in range(len(s) - 2)}


class Deduper:
    """归一化精确去重 + 3-gram Jaccard 近似去重。"""

    def __init__(self, threshold=0.72):
        self.threshold = threshold
        self.index = {}
        self.sets = []
        self.seen = set()
        self.blocked = 0
        self.prefixed = 0
        self.max_sim = 0.0

    def similarity(self, grams):
        candidates = set()
        for gram in grams:
            candidates.update(self.index.get(gram, ()))
        best = 0.0
        for idx in candidates:
            other = self.sets[idx]
            inter = len(grams & other)
            if not inter:
                continue
            score = inter / len(grams | other)
            if score > best:
                best = score
                if best >= self.threshold:
                    return best
        return best

    def accept(self, text):
        grams = trigrams(text)
        if normalize(text) in self.seen:
            self.blocked += 1
            return False
        score = self.similarity(grams)
        if score >= self.threshold:
            self.blocked += 1
            return False
        self.seen.add(normalize(text))
        idx = len(self.sets)
        self.sets.append(grams)
        for gram in grams:
            self.index.setdefault(gram, []).append(idx)
        self.max_sim = max(self.max_sim, score)
        return True


COLLOQUIAL_SUBJECT = [
    "{subj} 的{s}到底是啥？",
    "{subj} 里的{s}是啥？",
    "问下，{subj} 的{s}是咋回事？",
    "{subj} 的{s}能简单说说吗？",
]

COLLOQUIAL_GENERAL = [
    "{t}到底是啥？",
    "{t}能简单说下吗？",
    "问一下，{t}是啥？",
    "{t}是咋回事？",
]


def polish(text):
    """中英文之间补空格，补齐句末问号。"""
    text = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9_])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z0-9_])([\u4e00-\u9fff])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith(("？", "?", "。", "！")):
        text += "？"
    return text


def classify_topic(topic):
    if "要件" in topic:
        return "meta"
    for kind, keywords in TOPIC_RULES:
        if any(word in topic for word in keywords):
            return kind
    return "meta"


def render_subject(subject, seed, position, rng):
    stype = seed[0]
    if stype == "c" and any(word in seed[1] for word in ISSUE_WORDS):
        stype = "issue"
    groups = SUBJECT_TEMPLATES[stype]
    if len(groups) <= 4:
        chosen = groups
    else:
        skip = position % len(groups)
        chosen = [group for i, group in enumerate(groups) if i != skip][:4]
    if stype in ("w", "p"):
        params = {"p": seed[1], "subj": subject}
    elif stype == "d":
        params = {"a": seed[1][0], "b": seed[1][1], "subj": subject}
    elif stype == "h":
        params = {"v": seed[1], "subj": subject}
    elif stype == "e":
        params = {"err": seed[1], "subj": subject}
    else:
        params = {"s": seed[1], "subj": subject}
    rendered = []
    for group in chosen:
        rendered.append([polish(tpl.format(**params)) for tpl in group])
    # 约两成概念类问题改用口语化问法，覆盖真实提问里的随意表达。
    if stype in ("c", "issue") and rng.random() < 0.2:
        rendered[-1] = [polish(tpl.format(**params)) for tpl in COLLOQUIAL_SUBJECT]
    return rendered


def render_general(topic, rng):
    kind = None
    core = topic
    for suffix, suffix_kind in SUFFIX_RULES:
        if topic.endswith(suffix):
            kind = suffix_kind
            core = topic[: -len(suffix)]
            break
    if kind is None:
        kind = classify_topic(topic)
    groups = [[polish(tpl.format(t=core)) for tpl in group] for group in GENERAL_TEMPLATES[kind]]
    if rng.random() < 0.2:
        groups[-1] = [polish(tpl.format(t=core)) for tpl in COLLOQUIAL_GENERAL]
    return groups


def emit(choices, label, deduper, bucket):
    """按顺序尝试变体，取第一条通过去重的；都冲突时加礼貌前缀重试，绝不写入重复句。"""
    for text in choices:
        if deduper.accept(text):
            bucket.append({"query": text, "label": label})
            return
    for prefix in ("请问，", "想请教一下，", "麻烦问下，", "打扰一下，"):
        text = prefix + choices[0]
        if deduper.accept(text):
            deduper.prefixed += 1
            bucket.append({"query": text, "label": label})
            return
    raise RuntimeError(f"无法为「{choices[0]}」生成不重复的问法，请调整模板或种子")


def build(rng, deduper):
    subject_items = []
    for subject, seeds in SUBJECTS:
        if len(seeds) != 125:
            raise ValueError(f"{subject} 种子应为 125 条，实际 {len(seeds)}")
        for position, seed in enumerate(seeds):
            for choices in render_subject(subject, seed, position, rng):
                emit(choices, LABEL_SUBJECT, deduper, subject_items)

    general_items = []
    for domain, topics in GENERAL.items():
        for topic in topics:
            for choices in render_general(topic, rng):
                emit(choices, LABEL_GENERAL, deduper, general_items)
    return subject_items, general_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "..", "subject_scope_5000.json"))
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    deduper = Deduper()
    subjects, general = build(rng, deduper)

    if len(subjects) != PER_SUBJECT * len(SUBJECTS):
        raise ValueError(f"学科样本应为 {PER_SUBJECT * len(SUBJECTS)}，实际 {len(subjects)}")
    if len(general) != PER_GENERAL:
        raise ValueError(f"通用样本应为 {PER_GENERAL}，实际 {len(general)}")

    data = subjects + general
    rng.shuffle(data)

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    labels = Counter(item["label"] for item in data)
    queries = [item["query"] for item in data]
    lengths = [len(q) for q in queries]
    repeats = len(queries) - len(set(queries))

    print(f"输出文件   : {out_path}")
    print(f"总条数     : {len(data)}")
    print(f"标签分布   : {dict(labels)}")
    print(f"学科样本   : {len(subjects)}（每学科 {len(subjects) // len(SUBJECTS)}）")
    print(f"通用样本   : {len(general)}")
    print(f"精确重复   : {repeats}")
    print(f"相似度拦截 : {deduper.blocked}（阈值 {deduper.threshold}）")
    print(f"前缀兜底   : {deduper.prefixed}")
    print(f"字符长度   : 最短 {min(lengths)} / 平均 {sum(lengths) / len(lengths):.1f} / 最长 {max(lengths)}")
    print(f"主题覆盖   : 学科 5 个 × 125 知识点；通用 {sum(len(v) for v in GENERAL.values())} 个主题")


if __name__ == "__main__":
    main()
