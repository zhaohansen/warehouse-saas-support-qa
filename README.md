# 企业知识库智能问答系统

面向企业知识查询场景的 RAG 演示项目。用户可以导入企业制度、产品手册、业务流程和入职资料，通过自然语言查找资料并获取带来源的回答。项目目前是共享空间，没有多用户账号、角色权限或多租户隔离，也未声称已经在真实企业部署。

## 功能

- 支持 PDF、PPTX、DOCX 和 UTF-8 TXT 文件导入、资料分类、查看与删除。
- 使用 BGE-M3 生成稠密和稀疏向量，借助 Milvus 混合检索、重排和大模型生成回答，并展示检索来源。
- 支持流式输出、停止生成、对话历史和网页访问口令；检索失败时单独提示。
- 企业问题会先检索知识库：检索不到或相关度不足时明确说明缺少依据；通用概念问题走常识回答并标注未引用知识库。
- 附带四类共 16 份虚构企业资料与 18 条配套评估问题，全部仅供演示，不能当作真实企业制度。

## 最短演示路径（Windows）

1. 复制 `base/config.example.ini` 为 `base/config.ini`，确认模型路径、大模型服务和 Milvus 配置。没有 Milvus 服务端时，在 `[milvus]` 填写 `uri = ./storage/milvus_lite.db`，或设置环境变量 `QA_MILVUS_URI`（不要用 `MILVUS_URI` 指向本地文件，pymilvus 会把它当成服务地址），使用 Milvus Lite。
2. 安装依赖并在项目根目录启动：`python app.py`。默认地址 http://127.0.0.1:8081，访问 `/health` 确认 `ready` 和 `retrieval_ready` 都为 `true`。
3. 打开网页，在「导入资料」中按分类导入以下 **4 份** 文件即可完成最短演示（服务不会自动导入）：

| 分类 | 文件 |
| --- | --- |
| 制度规范 `policies` | `rag_qa/data/policies_data/虚构演示-请假与报销规范.txt` |
| 产品手册 `products` | `rag_qa/data/products_data/虚构演示-协作平台登录故障与账号恢复.txt` |
| 业务流程 `operations` | `rag_qa/data/operations_data/虚构演示-客户反馈处理流程.txt` |
| 员工入职 `onboarding` | `rag_qa/data/onboarding_data/虚构演示-新员工入职指南.txt` |

4. 依次提问：
   - 「无法登录协作平台应该联系谁？」应回答 IT 服务台并提供来源；
   - 「新员工领取办公设备的流程顺序是什么？」应给出申请、核对、配置登记、当面确认四步并引用入职指南；
   - 「节日福利金额是多少？」应明确说明资料不足，不猜测金额。
5. 在同一会话追问「那应该找谁？」，或在分类筛选中只选「产品手册」后再问报销问题，观察追问上下文与分类过滤效果。

修改代码或配置后需重启服务。首次安装依赖及模型：按硬件安装兼容的 PyTorch 后安装 `requirements.txt`；模型权重放在 `rag_qa/models/`，密钥使用环境变量（`DASHSCOPE_API_KEY` 或 `DEEPSEEK_API_KEY`），不要写入配置文件。

## 完整评估资料与批量建库

企业评估题依赖 `rag_qa/data` 下 **全部 16 份** 文件（四个分类各 4 份），导入不全会导致跨文档题和部分事实题召回不到依据：

| 分类目录 | 文件 |
| --- | --- |
| `policies_data` | 虚构演示-请假与报销规范.txt；虚构演示-请假申请与销假流程.txt；虚构演示-费用报销办理指引.txt；虚构演示-差旅申请与行程变更指引.txt |
| `products_data` | 虚构演示-协作平台产品手册.txt；虚构演示-协作平台使用指南.txt；虚构演示-协作平台角色与权限说明.txt；虚构演示-协作平台登录故障与账号恢复.txt |
| `operations_data` | 虚构演示-客户反馈处理流程.txt；虚构演示-客户反馈登记规范.txt；虚构演示-工单分级与超时升级规则.txt；虚构演示-工单交接与关闭规范.txt |
| `onboarding_data` | 虚构演示-新员工入职指南.txt；虚构演示-新员工入职第一天安排.txt；虚构演示-办公设备申领与异常处理.txt；虚构演示-系统账号与权限开通指引.txt |

在项目根目录执行批量建库（使用 `base/config.ini` 中的同一数据库 `enterprise_knowledge`、集合 `enterprise_knowledge_qa` 和同一套分块参数）：

```powershell
python -m rag_qa.main --data-processing
```

使用 Milvus Lite 时，建库与启动网页必须指向同一个本地库文件：

```powershell
$env:QA_MILVUS_URI = "./storage/milvus_lite.db"
python -m rag_qa.main --data-processing
python app.py
```

批量建库只扫描四个 `*_data` 目录中的正式资料，会跳过网页导入产生的 `uploads/` 哈希副本，避免同一文件重复入库。`rag_qa/data` 之外的验收题文件 `rag_qa/demo_acceptance_questions.md` 不会被建库命令读取，也不应上传到知识库。

建库前可核对当前实际使用的数据库、集合和 Milvus 地址（不会输出任何密钥）：

```powershell
python -c "from base.config import config; print('collection=', config.MILVUS_COLLECTION_NAME); print('database=', config.MILVUS_DATABASE_NAME); print('uri=', config.MILVUS_URI or f'{config.MILVUS_HOST}:{config.MILVUS_PORT}')"
```

建库和启动网页必须使用同一个 `uri`、数据库和集合：如果建库时设置了 `QA_MILVUS_URI`，启动 `app.py` 的同一个终端也要保留该环境变量。

## 企业评估

`rag_qa/rag_assessment/enterprise_evaluate_data.json` 包含 18 条根据虚构演示资料编写的问题，覆盖直接事实、流程顺序、跨文档、换一种问法和资料不足五类。先导入全部 16 份资料，再执行采集与评分。

### 继续上次采集

默认行为是断点续采：结果文件中已完成的题目会跳过，未完成或失败的题目继续执行。

```powershell
python -m rag_qa.rag_assessment.collect_rag_results `
  --output rag_qa/rag_assessment/enterprise_evaluate_results.json

python -m rag_qa.rag_assessment.rag_evaluator `
  --data rag_qa/rag_assessment/enterprise_evaluate_results.json `
  --progress rag_qa/rag_assessment/enterprise_ragas_scores_progress.json
```

### 重新跑一轮（不覆盖历史结果）

重新采集必须指定一个**新的输出文件**；`--restart` 会忽略已有完成记录并重做全部题目。若目标文件已存在，脚本会在加载模型前拒绝执行，除非显式加 `--overwrite`。评分时同时指定新的进度文件。

```powershell
# 先确认新文件名没有被历史结果占用；下面的 round3 是示例，可换成当天日期
Test-Path rag_qa/rag_assessment/enterprise_evaluate_results_round3.json

python -m rag_qa.rag_assessment.collect_rag_results `
  --output rag_qa/rag_assessment/enterprise_evaluate_results_round3.json `
  --restart

python -m rag_qa.rag_assessment.rag_evaluator `
  --data rag_qa/rag_assessment/enterprise_evaluate_results_round3.json `
  --progress rag_qa/rag_assessment/enterprise_ragas_scores_round3.json
```

`round2` 文件已存在，仅用于历史对照；不要把它当作新一轮输出。若确实要覆盖某个已有文件，必须同时写 `--restart --overwrite`。

只想先试跑少量题目时加 `--limit 2`。采集会真实调用已配置的大模型，并记录每题的回答和实际放入提示词的检索资料；评估默认使用本地 Ollama 的 `qwen2.5:7b` 和 `bge-m3` 计算 RAGAS。

### 统计口径

- **RAGAS 指标**：只计算“有依据且检索到资料”的样本；资料不足题和检索未命中都不进入四项均值。
- **有依据题**：以是否检索到资料、回答是否含引用编号，以及人工核对参考答案为准。
- **资料不足题**：单独统计“检索是否命中”和“回答是否正确说明资料不足”。
  - 通过：明确说明资料不足/缺少依据，且没有给出无依据的具体金额、天数、时限等结论；
  - 失败：没有说明资料不足，或先说资料不足又给出具体猜测（例如“资料不足，但应该是每人 500 元”）；
  - 待人工复核：含数字表述、但自动规则无法判断是引用还是结论的回答。
- 检索未命中会分别统计“有依据题未命中”和“资料不足题未命中”，不能把未命中直接算作通过。
- 采集失败或回答为空的题目单独计入“采集失败/未完成”，既不算检索命中，也不计入资料不足题的通过/失败；需重新采集后再评分。

只有实际完成采集和评分后，才应把本轮指标写入文档。历史文件 `enterprise_evaluate_results.json`、`enterprise_ragas_scores_progress.json` 保留用于对照，不要用模拟分数或旧学科结果替代企业场景成绩。

上一轮实测（2026-11-14，Milvus Lite + 真实 BGE-M3/BGE-Reranker + qwen-plus 采集，Ollama 评估）：

- 18 题中 14 题有检索资料、4 题资料不足；4 道资料不足题全部检索为空。
- RAGAS（14 条）：faithfulness 0.976，answer_relevancy 0.909，context_precision 1.000，context_recall 0.982。
- 需人工复核：新员工账号题 faithfulness 0.667；办公设备流程题 context_recall 0.75；请假材料题回答没有保留引用编号，但来源列表正确。

## 高频问答与分类模型

### 标准问答库

`mysql_qa/db/faq_knowledge_base.csv` 现有 21 条企业高频问答，覆盖制度规范 7 条、产品手册 5 条、业务流程 5 条、员工入职 4 条；每条的 `来源文档` 都指向 `rag_qa/data` 中的虚构演示文件。跨文档、依赖个人条件或资料不足的问题不进入 FAQ，继续走检索。

首次部署或更新 FAQ 后执行（可重复运行）：

```powershell
python -m mysql_qa.import_faq --dry-run
python -m mysql_qa.import_faq
```

导入按 `资料分类 + 问题` 判重，只插入不存在的记录，不更新、不清空已有 `jpkb` 数据；MySQL 不可用时应用自动回退本地 CSV。FAQ 命中时来源显示“标准问答库 · 原始文档名”。

### BERT 企业问题分类

训练数据 `rag_qa/classify_data/enterprise_scope_v2.jsonl` 共 240 条，按 `family` 改写族划分训练/测试（196/44），测试集不包含训练集中的近似改写，也不复用 RAG 评估题。企业资料类的定义是“必须查本地资料才能确认的业务事实”，即使当前资料没有覆盖也归入该类。

```powershell
python -m rag_qa.core.query_classifier --train
python -m rag_qa.core.query_classifier --predict "VPN 是什么？" "我们公司的 VPN 怎么申请？"
python -m rag_qa.core.query_classifier --compare-routing
```

模型保存在 `rag_qa/models/bert_enterprise_classifier`，指标写入 `training_metrics.json`，规则对比写入 `routing_comparison.json`。本轮独立测试集：准确率 0.977，企业资料精确率 0.960、召回率 1.000；仅规则召回率 0.792，规则+BERT 召回率 1.000，代价是 1 条通用问题被误判为企业资料。

分流原则：规则判定需要本地资料时直接检索，BERT 不参与；规则判为通用知识时，只有 BERT 以 ≥0.80 置信度判为企业资料才反向触发检索；定义类问题（如“VPN 是什么？”）保留保护；模型缺失、异常或置信度不足时沿用规则结果。

## 数据隔离与限制

企业项目使用 Milvus 数据库 `enterprise_knowledge` 下的集合 `enterprise_knowledge_qa`，以及对话文件 `storage/enterprise_history.sqlite3`。验证时可设置 `QA_HISTORY_PATH` 指向独立的会话库，避免写入正式历史。旧数据已单独备份，企业项目不会使用旧集合和历史。FAQ 已补充 21 条虚构演示标准问答；企业分类模型已训练并参与分流，模型或置信度不可靠时回退规则判断。

真实企业制度、审批信息和产品规则须由使用者提供并核验。示例资料均为虚构。内部问题在检索为空或重排分数低于 `min_answer_score` 时直接提示资料不足，不猜测内部人员、审批规则或联系方式；通用概念问题不检索企业资料，并在回答前标明未引用知识库。MySQL 不可用时回退本地 FAQ（来源会显示 `CSV`），Redis 不可用时回退内存缓存；模型或向量库不可用时检索不可用。

远程访问需配置 `HOST` 和 `QA_ACCESS_TOKEN`，并建议配合 HTTPS；访问口令不是用户账号或权限系统。

## 测试

```powershell
python -m pytest tests -q
node tests/test_upload_selection.js
```

自动化测试中的模拟依赖不能替代真实模型与 Milvus 的端到端验证。
