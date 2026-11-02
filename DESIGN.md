# 学科知识问答系统 · 设计说明

面向 Python / Java / AI / MySQL / Redis 五个学科的知识问答服务。核心思路是**双通道**：常见问题走 FAQ 精确匹配，长尾问题走 RAG 检索增强生成。

---

## 一、目录结构

```
subject-knowledge-qa/
├── app.py                          # FastAPI 服务入口（HTTP + WebSocket）
├── cli.py                          # 命令行入口（多轮对话 + 流式输出）
├── base/
│   ├── config.py                   # 配置加载（config.ini + 环境变量）
│   ├── config.ini                  # 学科、模型、检索参数
│   └── logger.py                   # 日志
├── mysql_qa/                       # FAQ 通道
│   ├── db/mysql_client.py          # MySQL 读写，连不上时降级读本地 CSV
│   ├── db/faq_knowledge_base.csv   # FAQ 语料（学科 / 问题 / 答案）
│   ├── cache/redis_client.py       # 缓存，连不上时降级为内存字典
│   ├── retrieval/bm25search.py     # BM25 召回 + 阈值判断
│   └── utils/preprocess.py         # 中文分词与清洗
├── rag_qa/                         # RAG 通道
│   ├── core/
│   │   ├── document_processor.py   # 文档加载与父子块切分
│   │   ├── vector_store.py         # Milvus 建表、入库、混合检索、重排
│   │   ├── rag_pipeline.py         # 完整问答链路（分类 → 策略 → 检索 → 生成）
│   │   ├── rag_system.py           # 基础版链路（无多轮）
│   │   ├── query_classifier.py     # BERT 意图分类
│   │   ├── strategy_selector.py    # 四种检索策略选择
│   │   └── prompts.py              # 提示词模板
│   ├── edu_document_loaders/       # PDF / Word / PPT / 图片 / OCR 加载器
│   ├── edu_text_splitter/          # 中文递归切分、Markdown 切分
│   ├── rag_assessment/rag_evaluator.py  # RAGAS 评估脚本
│   ├── classify_data/              # 分类器训练数据与标注提示词
│   └── data/{学科}_data/           # 各学科知识库原文
└── static/index.html               # 前端（单文件，无外部依赖）
```

---

## 二、一次提问的完整链路

```
用户提问
   │
   ├─ 命中问候语正则 ──────────────→ 直接返回固定文案（不调模型）
   │
   ├─ FAQ 通道：BM25 匹配历史问题
   │     相似度 ≥ 0.85 ────────────→ 返回标准答案（不检索、不调模型）
   │
   └─ RAG 通道
         ① 意图分类（BERT）：通用知识 → 模型直接作答
         ② 专业咨询 → 选择检索策略（直接 / HyDE / 子查询 / 回溯）
         ③ BGE-M3 编码 → Milvus 稠密 + 稀疏混合检索
         ④ 父块去重 → bge-reranker 重排 → Top-2
         ⑤ 拼装上下文与对话历史 → 大模型生成
```

---

## 三、关键设计决策

### 1. 双通道：能查表就不调模型

FAQ 库里的问题是人工确认过的标准答案，质量与稳定性都优于模型生成；同时一次查询不需要向量编码与模型调用，成本和延迟都低。只有 FAQ 覆盖不到的长尾问题才进入 RAG，这是成本与效果之间的权衡。

### 2. 父子块切分

父块 1200 字、子块 300 字、重叠 50 字。**子块用于检索，父块用于生成**：子块小，向量表达更聚焦，召回更准；命中后回溯父块，上下文完整，避免答案被切断。

### 3. 稠密 + 稀疏双路召回

BGE-M3 同时输出 1024 维稠密向量与稀疏向量。稠密向量负责语义相似（"苹果关机" 能匹配 "iPhone 重启"），稀疏向量负责术语精确匹配（代码里的类名、参数名不会因为语义泛化而丢失）。两路结果用 `WeightedRanker` 按 0.7 / 1.0 加权融合。

索引侧：稠密用 `IVF_FLAT`（nlist=128），稀疏用 `SPARSE_INVERTED_INDEX`（构建时丢弃 20% 低权重项，压缩体积、降低噪声）。

### 4. 重排而非只靠向量相似度

向量内积衡量的是整体语义接近程度，对"是否真正回答了问题"不敏感。因此召回 Top-K 后，按父文档去重，再用 `bge-reranker-large` 对「查询—父块」逐对打分重排，只保留 Top-2 进入提示词。重排更准但更慢，所以只作用于少量候选。

### 5. 意图分类 + 关键词兜底

BERT 分类器区分"通用知识"与"专业咨询"，通用知识直接作答以省掉一次检索。但分类器是在特定语料上训练的，学科范围变化后容易把专业问题误判成闲聊，一旦误判就完全跳过检索。因此在分类之后加一层规则兜底：**用户指定了学科，或问题命中学科关键词，一律按专业咨询处理**。模型负责泛化，规则负责兜底。

### 6. 依赖不可用时的降级

| 依赖 | 不可用时的行为 |
|---|---|
| MySQL | 自动改读本地 CSV，FAQ 仍可用 |
| Redis | 降级为进程内字典缓存，重启失效但不报错 |
| MySQL 且需多轮 | 自动关闭历史功能，主流程继续 |

降级的目标是"功能减弱但不崩"，而不是静默失败——每种降级都会写一条 warning 日志。

### 7. 参数外置

检索权重、`nprobe`、Top-K、切分长度、模型名、学科列表全部放在 `config.ini`。调检索策略不需要改代码，换学科只需要加一个目录并改一行配置。

---

## 四、运行

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 基础设施（MySQL 3306 / Redis 6379 / Milvus 19530）
#    Milvus 可用 docker compose 启动

# 3. 建索引（首次或知识库变更后执行）
python -m rag_qa.main --data-processing

# 4. 启动服务
python app.py            # 默认 8082
```

访问 `http://127.0.0.1:8082`，接口文档在 `/docs`。

---

## 五、已知限制

- `config.ini` 里的 `min_score` 尚未接入检索链路。当前靠重排控制相关性，低分文档仍可能进入上下文。数据量增大后应加阈值过滤。
- BERT 分类器未用新学科数据重训，目前依赖关键词兜底。长期方案是用 `rag_qa/classify_data` 的格式重新标注并训练。
- 向量编码与重排在 CPU 上运行，单次专业咨询约 12–15 秒，其中检索与重排占 1–2 秒，其余为模型生成。上 GPU 或改流式可进一步改善体感。
