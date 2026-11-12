# 知序 · 学科知识助手

面向 Python、Java、AI、MySQL、Redis 的个人知识问答工作台。支持 FAQ、知识库检索、真正的流式回答，以及 PDF / PPTX / TXT 网页导入。

## 启动

启动前先确认 Docker Desktop 与 Milvus 容器已运行；如果后端在 Milvus 就绪前启动过，需要重启后端。打开 `/health`，`retrieval_ready` 为 `true` 表示知识库已就绪。Windows 上若系统 `python` 指向商店占位程序，请直接调用已安装的 `edu_rag` 环境解释器，例如 `C:\Software\Anaconda3\envs\edu_rag\python.exe app.py`（按本机实际路径调整）。

使用已有的 `edu_rag` Python 环境，从项目目录运行：

```powershell
python app.py
```

默认打开 http://127.0.0.1:8081 ，可通过 `PORT` 环境变量调整。修改代码后需要重启后端并刷新网页。

首次安装请先按硬件安装对应的 PyTorch（CPU、CUDA 或 ROCm），再使用 `pip install -r requirements.txt`；已有 AMD ROCm 环境请保留原来的 PyTorch，不要强制重装，复制 `base/config.example.ini` 为 `base/config.ini` 后配置 MySQL、Redis、Milvus 与模型路径。模型通常位于 `rag_qa/models/`；模型相对路径以项目根目录为基准，也可以填写绝对路径。

模型服务由 `base/config.ini` 中的 `base_url` 和 `model` 决定。使用 DeepSeek 官方接口时，从环境变量 `DEEPSEEK_API_KEY` 读取密钥；使用其他兼容接口时，优先读取 `DASHSCOPE_API_KEY`，否则使用配置文件的 `api_key`。不要把真实密钥提交到仓库。

## 使用

- 直接提问，或先限定学科。FAQ 只在相应学科中匹配；未命中再检索资料。
- 点击「导入资料」，选择学科与 PDF、PPTX、UTF-8 TXT。单文件上限 20 MB，同一学科的相同文件不重复导入。
- 「我的资料库」展示导入文件、学科、知识块数量及识别警告。扫描件解析可能需要更长时间。
- 最终回答逐段显示，可停止生成、复制回答和代码、导出当前对话。中断或失败的半截答案不会被作为完整回答保存。
- 左侧可搜索、切换与删除会话；支持手机布局及深浅主题。

## 数据与模型

聊天记录存储在 `storage/history.sqlite3`，不会只保留 5 轮。给模型的上下文仍只取最近 5 轮并限制字数。网页单次最多读取最近 500 轮，数据库保留完整记录。

首次启动会尝试将旧 MySQL `conversations` 表复制到本地历史库，成功后不重复迁移；不删除或修改原记录。过去已被旧程序删除的聊天无法恢复。备份历史时应先停止服务，再复制 `storage/`。

文件保存到对应学科 `uploads/` 下，文件名、知识块数量及警告另存为隐藏的 JSON 元数据。新导入的 PDF/PPTX 按页或幻灯片保留来源；旧索引缺少的页码不会凭空补齐，可重新建库以更新。

```powershell
python -m rag_qa.main --data-processing
```

建库会按文件更新索引并清理该文件的旧块，不清空其他文件或学科。已从磁盘删除的文件不会自动从向量库删除。不要单独清空向量库后依赖文件的重复导入标记，应重新执行建库。

文本转向量与重排按配置使用 CUDA，不可用时回退 CPU；模型访问串行化以降低并发显存压力。OCR 按需初始化并复用。PPT 个别 WMF/EMF 或损坏图片无法读取时会跳过该图，网页明确显示警告；该图中的文字没有入库。需要这部分内容时，把图片转为 PNG/JPG，或导出 PDF 后导入。

MySQL 不可用时 FAQ 回退本地 CSV；Redis 不可用时回退有容量与过期限制的内存缓存。向量库或本地模型不可用时，标准 FAQ 与聊天记录仍可使用。本地检索未命中时，一般学习与编程问题仍会用通用知识回答，并明确说明未引用本地资料；用户明确要求依据文档或涉及课件中的具体参数、评分标准、内部规定等特定事实时，资料不足仍会提示补充。检索服务异常会单独报告，不冒充正常知识库回答。

BERT 分类器缺失时默认走知识检索，不使用未经训练的随机分类头。默认训练语料为 `rag_qa/classify_data/subject_scope_5000.json`（通用知识 / 学科资料各 2500 条），由 `rag_qa/classify_data/subject_scope/generate.py` 生成。该数据集按知识领域分类，学科资料包含 Python、Java、AI、MySQL 和 Redis 相关问题。旧的 `build_dataset.py` 采用不同的标注规则，不是默认数据集的生成器。

标准 FAQ 库 `mysql_qa/db/faq_knowledge_base.csv` 只覆盖 Python、Java、AI、MySQL、Redis 五门学科，每门 20 个知识点、两种问法；读取时会按 `valid_sources` 过滤，避免旧题库中其它学科的条目混入回答。`rag_qa/rag_assessment/rag_evaluate_data.json` 是覆盖五门学科各 20 题的 RAGAS 评估题库（共 100 条，分基础概念、原理分析、概念对比、应用场景和资料依据不足五类），用于衡量忠实度、答案相关性与上下文精确率、召回率；`splits/` 下是按学科拆分的版本，额外保留 `subject`、`category`、`has_sufficient_evidence` 和 `note`，便于分组统计或剔除资料不足题。主文件每条只保留 `question`、`answer`、`context`、`ground_truth`，其中 `answer` 和 `context` 需先用系统实跑回填，否则评估脚本会直接报错提示。

## 访问范围

默认只监听本机 `127.0.0.1`。这是单个共享学习空间，不提供独立的多用户账号。

远程使用需设置 `HOST` 和 `QA_ACCESS_TOKEN`，建议使用 HTTPS 与足够长的随机口令。网页通过访问口令验证，API 也支持 `Authorization: Bearer <访问口令>`；该口令与模型 API 密钥是两回事。未设置口令时，远程 API 与 WebSocket 访问会被拒绝。

请求内容和来源经过校验，浏览器跨站写入被拒绝；模型返回的文本不会作为 HTML 执行。

## 测试

```powershell
python -m unittest discover -s tests -v
```

当前自动化测试使用临时文件和模拟依赖，不消耗模型额度；覆盖检索策略路由、评估进度、资料删除、接口鉴权及模拟流式问答。测试接口依赖 `httpx`，已包含在 `requirements.txt`。额外运行 `node tests/test_upload_selection.js` 可检查上传控件状态。自动化测试不代替真实模型和 Milvus 链路测试。

检索先选择策略，再用原问题从每路召回候选（`recall_candidates=20`）。若选择了子查询、HyDE 或回溯策略，还会执行相应增强检索，合并去重后按原问题重排，最终选取 `candidate_m` 个上下文。增强策略可能增加模型调用和等待时间。
BGE-M3 同时传入包装器 `device` 和底层 FlagEmbedding 的 `devices`，确保设备设置在当前依赖版本实际生效。

问答优先级：纯问候 → 标准 FAQ → 本地资料检索 → 确实未命中后的通用知识回答。BERT 的“通用知识”标签不会跳过本地检索；当前分类结果用于记录，不决定是否调用增强检索策略。这样新上传的任意主题课件都能参与回答，而不依赖写死的学科关键词。
