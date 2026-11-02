# 学科知识问答系统

面向 Python / Java / AI / MySQL / Redis 五个学科的知识问答服务，采用 **FAQ + RAG 双通道**：常见问题走 BM25 精确匹配，长尾问题走向量检索 + 大模型生成。

## 功能

- **双通道召回**：FAQ 命中（相似度 ≥ 0.85）直接返回标准答案，不触发向量检索与模型调用；未命中转入 RAG。
- **混合检索**：BGE-M3 同时产出稠密与稀疏向量，Milvus 双路召回后加权融合。
- **重排**：父块去重后用 bge-reranker-large 逐对打分，只保留 Top-2 进入提示词。
- **意图识别**：BERT 二分类区分「通用知识 / 专业咨询」，并加学科关键词兜底，避免误判跳过检索。
- **策略路由**：专业咨询由大模型在直接检索、HyDE、子查询、回溯四种策略中择优。
- **流式输出**：FastAPI + WebSocket 逐字推送，前端单文件实现，支持深色模式与移动端。
- **降级可用**：MySQL 不可用改读本地 CSV，Redis 不可用降级为内存缓存。

## 技术栈

Python 3.10 · FastAPI · WebSocket · LangChain · Milvus 2.5 · BGE-M3 · bge-reranker-large · BERT（transformers） · MySQL 8 · Redis · 通义千问（DashScope）

## 快速开始

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 配置（config.ini 不入库，从模板复制后按需修改）
cp base/config.example.ini base/config.ini
export DASHSCOPE_API_KEY=你的百炼密钥

# 3. 启动基础设施：MySQL 3306 / Redis 6379 / Milvus 19530
#    Milvus 可用 docker compose 启动

# 4. 建索引（首次或知识库变更后执行）
python -m rag_qa.main --data-processing

# 5. 启动服务
python app.py          # 默认 8082
```

打开 <http://127.0.0.1:8082>，接口文档在 `/docs`。

模型文件放在 `rag_qa/models/` 下（bge-m3、bge-reranker-large、bert-base-chinese、bert_query_classifier），体积较大，未纳入仓库，需自行下载。

## 目录结构

```
├── app.py                  FastAPI 服务入口
├── cli.py                  命令行入口（多轮 + 流式）
├── base/                   配置与日志
├── mysql_qa/               FAQ 通道：MySQL / Redis / BM25
├── rag_qa/
│   ├── core/               切分、向量库、问答链路、分类、策略、提示词
│   ├── edu_document_loaders/  PDF / Word / PPT / 图片 / OCR
│   ├── edu_text_splitter/  中文递归切分
│   ├── rag_assessment/     RAGAS 评估
│   └── data/               各学科知识库原文
├── static/index.html       前端页面
└── DESIGN.md               设计说明与关键取舍
```

## 效果

- 专业咨询端到端 10–15 秒（其中检索与重排约 1–2 秒，其余为模型生成），通用知识 0.7–0.9 秒。
- 五个学科共 79 个文档块，按学科过滤可命中对应领域内容。

## 已知限制

- `min_score` 阈值尚未接入检索链路，低分文档仍可能进入上下文。
- BERT 分类器未用新学科数据重训，目前依赖关键词兜底。
- 向量编码与重排在 CPU 上运行，延迟主要来自模型生成。

更多设计细节见 [DESIGN.md](DESIGN.md)。
