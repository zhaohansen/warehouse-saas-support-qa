# -*- coding: utf-8 -*-
"""学科知识问答意图数据集生成器.

用途: 为 rag_qa/core/query_classifier.py 的 BERT 二分类器生成训练语料.

两个标签:
  通用知识  —— 不依赖本地资料, 模型自身知识即可回答的学科/编程/计算问题.
  学科资料  —— 必须查阅用户导入的课件/讲义/PDF/资料库才能回答的问题.

关键设计: 两类共用同一主题池, 只有"是否指向本地资料"不同.
这样模型学到的是"要不要查资料", 而不是"看到 MySQL 就归某一类".

用法:
    python build_dataset.py                    # 默认 5000 条, 两类各 2500；旧版标注规则
    python build_dataset.py --per-class 1000   # 自定义规模, 共 2000 条
"""

import argparse
import json
import os
import random
from collections import Counter

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 主题池
# 每项: 主题名, 用于"概念/原理/区别/代码/复杂度"等通用问法.
TOPICS = [
    # Python
    "Python 的列表和元组", "Python 的生成器", "Python 的迭代器", "Python 的装饰器",
    "Python 的深浅拷贝", "Python 的 GIL", "Python 的闭包", "Python 的上下文管理器",
    "Python 的垃圾回收机制", "Python 的字典底层实现", "Python 的魔法方法",
    "Python 的异步 async/await", "Python 的类型注解", "Python 的虚拟环境",
    "Python 的多线程和多进程", "Python 的异常处理", "Python 的列表推导式",
    # Java
    "Java 的 JVM 内存模型", "Java 的垃圾回收器", "Java 的线程池", "Java 的 synchronized 和 Lock",
    "Java 的 HashMap 扩容机制", "Java 的接口和抽象类", "Java 的反射", "Java 的泛型擦除",
    "Java 的 Spring IOC", "Java 的 Spring AOP", "Java 的 volatile 关键字",
    "Java 的类加载机制", "Java 的集合框架", "Java 的序列化", "Java 的 JUC 并发工具",
    "Java 的 MyBatis 映射", "Java 的 Maven 依赖冲突",
    # MySQL
    "MySQL 的 B+ 树索引", "MySQL 的事务 ACID", "MySQL 的隔离级别", "MySQL 的 MVCC",
    "MySQL 的慢查询优化", "MySQL 的聚簇索引", "MySQL 的回表查询", "MySQL 的最左前缀原则",
    "MySQL 的锁机制", "MySQL 的主从复制", "MySQL 的分库分表", "MySQL 的执行计划 explain",
    "MySQL 的死锁排查", "InnoDB 和 MyISAM 的区别", "数据库的三大范式", "MySQL 的索引失效",
    # Redis
    "Redis 的五种数据类型", "Redis 的 RDB 持久化", "Redis 的 AOF 持久化", "Redis 的过期删除策略",
    "Redis 的内存淘汰策略", "缓存穿透", "缓存击穿", "缓存雪崩", "Redis 实现分布式锁",
    "Redis 的哨兵模式", "Redis 的集群模式", "Redis 的事务", "Redis 的 pipeline",
    "Redis 的大 key 问题", "Redis 和数据库的一致性",
    # 机器学习
    "监督学习和无监督学习", "过拟合", "正则化 L1 和 L2", "交叉验证", "梯度下降",
    "损失函数", "决策树", "随机森林", "支持向量机", "K-means 聚类", "特征工程",
    "精确率召回率和 F1", "AUC 指标", "偏差和方差", "集成学习", "朴素贝叶斯",
    "线性回归", "逻辑回归",
    # 深度学习
    "神经网络的反向传播", "激活函数", "卷积神经网络 CNN", "池化层", "RNN 和 LSTM",
    "Attention 注意力机制", "Transformer 结构", "BatchNorm", "Dropout", "学习率衰减",
    "Adam 优化器", "迁移学习", "梯度消失和梯度爆炸", "显存不足的优化",
    # NLP 与大模型
    "中文分词", "Word2Vec 词向量", "BERT 的预训练任务", "RAG 检索增强生成",
    "向量检索和 embedding", "Prompt 设计", "大模型幻觉", "LoRA 微调",
    "向量数据库", "重排序 rerank", "文本切块 chunk 策略",
    # 数据结构与算法
    "数组和链表", "栈和队列", "二叉树", "平衡二叉树", "哈希表", "图的遍历",
    "快速排序", "归并排序", "二分查找", "动态规划", "贪心算法", "回溯算法",
    "递归", "时间复杂度", "红黑树", "堆和优先队列", "布隆过滤器",
    # 操作系统与网络
    "进程和线程", "死锁的四个必要条件", "内存分页", "虚拟内存", "进程调度算法",
    "用户态和内核态", "TCP 三次握手", "TCP 四次挥手", "HTTP 和 HTTPS", "DNS 解析过程",
    "TCP 拥塞控制", "滑动窗口", "OSI 七层模型", "WebSocket 长连接", "CDN 加速原理",
    # 工程与工具
    "Docker 镜像和容器", "Dockerfile 编写", "Docker 数据卷", "Kubernetes 的 Pod",
    "Git 的分支合并", "git rebase 和 merge", "Git 回滚操作", "Linux 的权限 chmod",
    "Linux 查看端口占用", "Linux 的软链接和硬链接", "Shell 脚本循环", "正则表达式",
    "RESTful API 设计", "HTTP 状态码", "JSON 和 XML", "UTF-8 编码",
]

# ---------------------------------------------------------------- 对比对
PAIRS = [
    ("列表", "元组"), ("进程", "线程"), ("TCP", "UDP"), ("HTTP", "HTTPS"),
    ("GET", "POST"), ("堆", "栈"), ("深度学习", "机器学习"), ("BERT", "GPT"),
    ("RDB", "AOF"), ("Redis", "Memcached"), ("InnoDB", "MyISAM"), ("聚簇索引", "二级索引"),
    ("synchronized", "ReentrantLock"), ("ArrayList", "LinkedList"), ("HashMap", "TreeMap"),
    ("过拟合", "欠拟合"), ("精确率", "召回率"), ("batch", "epoch"), ("镜像", "容器"),
    ("git merge", "git rebase"), ("软链接", "硬链接"), ("值传递", "引用传递"),
    ("深拷贝", "浅拷贝"), ("生成器", "迭代器"), ("悲观锁", "乐观锁"),
    ("垂直分表", "水平分表"), ("主键索引", "唯一索引"), ("drop", "delete"),
    ("truncate", "delete"), ("内连接", "左连接"), ("where", "having"),
    ("RNN", "Transformer"), ("CNN", "RNN"), ("L1 正则", "L2 正则"),
    ("Bagging", "Boosting"), ("Docker", "虚拟机"), ("K8s Deployment", "StatefulSet"),
]

# ---------------------------------------------------------------- 代码类
CODE_TASKS = [
    "用 Python 读取一个 CSV 文件", "用 Python 统计字符串里每个字符出现的次数",
    "用 Python 判断一个数是不是质数", "用 Python 实现斐波那契数列",
    "用 Python 合并两个有序列表", "用 Python 找出列表中的重复元素",
    "用 Python 实现一个单例模式", "用 Python 写一个带重试的请求函数",
    "用 Python 实现二分查找", "用 Python 批量重命名文件",
    "用 Java 实现冒泡排序", "用 Java 写一个线程安全的计数器",
    "用 Java 实现单例模式", "用 Java 读取文件内容",
    "写一条 SQL 查询每个部门的平均工资", "写一条 SQL 找出重复的记录",
    "写一条 SQL 查询连续登录三天以上的用户", "写一条 SQL 实现行转列",
    "用 Redis 实现一个简单的限流器", "用 Shell 写一个定时备份脚本",
    "用 Python 实现一个 LRU 缓存", "用 Python 解析 JSON 并取值",
]

# 适合"手写"的算法/模式名, 与上面"用 XX 实现"的任务分开, 避免出现"手写用 Python 实现..."
HANDWRITE = [
    "快速排序", "冒泡排序", "二分查找", "单例模式", "LRU 缓存", "堆排序",
    "链表反转", "二叉树层序遍历", "KMP 算法", "并查集", "Dijkstra 算法",
    "TopK 问题", "归并排序", "滑动窗口最大值", "生产者消费者模型", "限流器",
]

# ---------------------------------------------------------------- 报错类
ERRORS = [
    "Python 报错 TypeError: 'NoneType' object is not subscriptable",
    "Python 报错 KeyError", "Python 报错 IndentationError",
    "Python 报错 ModuleNotFoundError", "Java 报错 NullPointerException",
    "Java 报错 ClassNotFoundException", "Java 报错 OutOfMemoryError",
    "MySQL 报错 Too many connections", "MySQL 报错 Lock wait timeout exceeded",
    "MySQL 报错 Duplicate entry for key", "Redis 报错 OOM command not allowed",
    "Docker 报错 port is already allocated", "pip 安装时报 SSL 证书错误",
    "Git 报错 fatal: refusing to merge unrelated histories",
]

# ---------------------------------------------------------------- 计算类
CALCS = [
    "1024 乘以 768", "2 的 10 次方", "100 的阶乘除以 98 的阶乘",
    "从 10 个数里选 3 个的组合数", "根号 2 保留四位小数",
    "1 加到 100 的和", "二进制 101101 转成十进制",
    "log2 1024", "sin30 度加 cos60 度", "一个 8 位二进制数最多能表示多少个值",
    "3 的 5 次方模 7", "矩阵 [[1,2],[3,4]] 的行列式", "圆的半径 3 时面积是多少",
    "2 的 32 次方减 1", "16 的平方根加上 9 的平方根",
]

# 只有这些主题适合问复杂度, 避免出现"缓存穿透的空间复杂度"这类怪问题
COMPLEXITY_TOPICS = [
    "快速排序", "归并排序", "二分查找", "哈希表", "二叉树", "红黑树", "堆和优先队列",
    "动态规划", "广度优先遍历", "深度优先遍历", "布隆过滤器", "MySQL 的 B+ 树索引",
    "MySQL 的最左前缀原则", "Redis 的 pipeline", "数组和链表", "栈和队列",
]

CHAPTERS = ["第一章", "第二章", "第三章", "第四章", "第五章", "第六章",
            "第七章", "第八章", "第九章", "第十章", "第十一章", "第十二章"]

DOC_KINDS = ["课件", "讲义", "PPT", "笔记", "实验指导书", "复习提纲", "教材", "资料"]

# ---------------------------------------------------------------- 通用知识模板
GENERAL_TEMPLATES = {
    "概念": [
        "{t}是什么？", "解释一下{t}", "{t}的原理是什么？", "简单说说{t}",
        "搞不懂{t}，能讲讲吗", "{t}到底是什么意思", "帮我理解一下{t}",
        "{t}是怎么工作的", "{}的核心思想是什么".format("{t}"),
        "能通俗地讲讲{t}吗", "{t}有什么用", "{t}一般用在什么场景",
    ],
    "对比": [
        "{a}和{b}有什么区别？", "{a}跟{b}比，各自优缺点是什么？",
        "什么时候用{a}，什么时候用{b}？", "{a}和{b}该怎么选",
        "{a}能替代{b}吗", "面试官问{a}和{b}的区别，怎么答",
    ],
    "代码": [
        "{c}，代码怎么写？", "{c}，给个示例", "帮我{c}",
        "{c}，用最简洁的写法", "{c}，注意什么坑", "手写{a}",
    ],
    "报错": [
        "{e}，怎么解决？", "遇到{e}，是什么原因", "{e}，排查思路是什么",
        "{e}，一般怎么修",
    ],
    "计算": [
        "{m}等于多少？", "帮我算一下{m}", "{m}，结果是多少",
        "口算一下{m}",
    ],
    "复杂度": [
        "{t}的时间复杂度是多少？", "{t}的空间复杂度是多少",
        "{t}的性能怎么样", "{t}在大数据量下会不会很慢",
    ],
}

# ---------------------------------------------------------------- 学科资料模板
LOCAL_TEMPLATES = {
    "章节": [
        "{ch}讲的{t}是什么？", "{ch}关于{t}那节讲了啥",
        "课件{ch}的{t}部分能再讲一遍吗", "{ch}里{t}的推导过程是什么",
        "{ch}提到{t}的时候举了什么例子",
    ],
    "定位": [
        "{doc}里{t}在第几页？", "我上传的{doc}里，{t}这块怎么写的",
        "{doc}中{t}的定义是什么", "{doc}里有没有讲{t}",
        "帮我找一下{doc}里关于{t}的内容",
        "{doc}里{t}的原文是怎么描述的",
        "{doc}里{t}的代码示例能贴一下吗",
    ],
    "复习": [
        "复习提纲里{t}是重点吗", "老师说{t}必考，在{doc}哪一页",
        "{doc}里{t}的考点有哪些", "考前想看{doc}里{t}那一节",
    ],
    "作业实验": [
        "实验指导书第{n}步要求做什么", "作业里{t}那道题的要求是什么",
        "{doc}第{n}页的练习题怎么做", "上次布置的{t}作业，格式要求是什么",
    ],
    "结构": [
        "{s}的{doc}目录结构是怎样的", "我导入的{doc}一共几章",
        "{doc}的整体框架是什么", "这门课{doc}的章节顺序是怎样的",
        "{doc}里{t}这一节在哪个位置",
    ],
}

PREFIXES = ["请问，", "想问一下，", "请教一下，", "麻烦问下，", "老师，", "打扰一下，"]


def with_prefix(rng, q, p=0.18):
    """偶尔加句首礼貌前缀, 增加句式多样性; 已有前缀或口语开头的不重复加."""
    if rng.random() >= p:
        return q
    if q.startswith(("请问", "想问", "请教", "麻烦", "老师", "打扰")):
        return q
    return rng.choice(PREFIXES) + q


def build_general(rng, n):
    out = []
    seen = set()

    def add(q):
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append({"query": q, "label": "通用知识"})

    # 轮转取样, 保证各模板均衡
    kinds = list(GENERAL_TEMPLATES)
    i = 0
    while len(out) < n and i < n * 40:
        kind = kinds[i % len(kinds)]
        tpl = rng.choice(GENERAL_TEMPLATES[kind])
        if kind == "对比":
            a, b = rng.choice(PAIRS)
            q = tpl.format(a=a, b=b)
        elif kind == "代码":
            if "{a}" in tpl:
                q = tpl.format(a=rng.choice(HANDWRITE))
            else:
                q = tpl.format(c=rng.choice(CODE_TASKS))
        elif kind == "报错":
            q = tpl.format(e=rng.choice(ERRORS))
        elif kind == "计算":
            q = tpl.format(m=rng.choice(CALCS))
        elif kind == "复杂度":
            q = tpl.format(t=rng.choice(COMPLEXITY_TOPICS))
        else:
            q = tpl.format(t=rng.choice(TOPICS))
        add(with_prefix(rng, q))
        i += 1
    return out


def build_local(rng, n):
    out = []
    seen = set()

    def add(q):
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append({"query": q, "label": "学科资料"})

    kinds = list(LOCAL_TEMPLATES)
    i = 0
    while len(out) < n and i < n * 40:
        kind = kinds[i % len(kinds)]
        tpl = rng.choice(LOCAL_TEMPLATES[kind])
        t = rng.choice(TOPICS)
        doc = rng.choice(DOC_KINDS)
        if kind == "章节":
            q = tpl.format(ch=rng.choice(CHAPTERS), t=t)
        elif kind == "定位":
            q = tpl.format(doc=doc, t=t)
        elif kind == "复习":
            q = tpl.format(doc=doc, t=t)
        elif kind == "作业实验":
            q = tpl.format(doc=doc, t=t, n=rng.randint(1, 20))
        else:
            a, _b = rng.choice(PAIRS)
            s = rng.choice(["Python", "Java", "MySQL", "Redis", "AI", "算法", "网络"])
            q = tpl.format(doc=doc, t=t, a=a, s=s)
        add(with_prefix(rng, q))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=20261109)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = build_general(rng, args.per_class) + build_local(rng, args.per_class)
    rng.shuffle(data)
    out_file = os.path.join(DATA_DIR, f"model_subject_{len(data)}.json")

    with open(out_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---------------- 自检
    cnt = Counter(x["label"] for x in data)
    qs = [x["query"] for x in data]
    banned = ["学费", "报名", "就业", "师资", "校区", "优惠", "试听",
              "包就业", "招生", "多少钱", "贵校", "培训班"]

    def is_dirty(q):
        if any(w in q for w in banned):
            return True
        # "证书" 只在非 SSL/HTTPS 语境下视为培训销售词
        return "证书" in q and "SSL" not in q and "ssl" not in q

    dirty = [q for q in qs if is_dirty(q)]
    lens = [len(q) for q in qs]

    print(f"输出文件 : {out_file}")
    print(f"总条数   : {len(data)}")
    print(f"标签分布 : {dict(cnt)}")
    print(f"去重后   : {len(set(qs))}  重复: {len(qs) - len(set(qs))}")
    print(f"长度     : 最短 {min(lens)} / 平均 {sum(lens)//len(lens)} / 最长 {max(lens)}")
    print(f"残留销售词: {len(dirty)}")
    if dirty:
        for q in dirty[:5]:
            print("   ", q)


if __name__ == "__main__":
    main()
