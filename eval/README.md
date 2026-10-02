# Ragas 问答评估

`eval.py` 读取 `qa.csv`，执行真实 RAG 图并使用 Ragas 0.4.3 计算五项指标，结果为 `eval_result.csv`。两个 CSV 都使用 UTF-8 BOM，结果按输入顺序输出。

当前 `qa.csv` 是用户选择生成的 **5 条教育演示样例**，来源为 `data/samples/education/course-policy.md`。它没有覆盖真实五领域语料，不能当作生产评测集或面试中的真实业务指标。可直接替换为你自己的 BOM CSV，列头必须为 `question,ground_truth`。

## 数据及模型

- `answer` 来自项目原有 LangGraph 的真实回答，不使用参考答案代填。
- `contexts` 从图最终状态 `state["reranked_docs"]` 获取：其中保留真实重排片段 ID 顺序，程序按当前权限读取权威正文，再按顺序导出 JSON 字符串数组。不会再次召回，也不会把参考答案充当 context。
- 稠密/稀疏召回和重排仍使用项目 BGE-M3 / BGE-Reranker-Large；Ragas 的语义指标通过 BGE-M3 的 1024 维稠密向量计算。
- 判分默认使用已有自建 Qwen3-8B，生成与判分使用同一模型可能存在偏差。若要使用独立评估模型，显式设置 `EVAL_LLM_BASE_URL`、`EVAL_LLM_API_KEY`、`EVAL_LLM_MODEL`，并确认资料可发给该模型。
- Ragas 0.4.3 使用支持指定五项传统指标的 `aevaluate` 及模型包装器。该接口在新版已标为弃用，但本项目固定该版本；没有用 Factual Correctness 偷换 Answer Correctness。
- 单独的 `pyproject.toml` / `uv.lock` 锁定评估依赖，`langchain-community==0.3.31` 保留 Ragas 需要的兼容模块，避免影响正在运行的业务服务。

## 已有远程 CPU 环境运行

当前 MongoDB、向量库和 CPU→GPU 隧道仅在 CPU 上可达，建议在 CPU 主机运行评估容器：

```bash
cd /home/ubuntu/knowledge-service
sudo -n docker compose --env-file .env.remote -f infra/compose.eval.yml build evaluator
sudo -n docker compose --env-file .env.remote -f infra/compose.eval.yml run --rm --no-deps evaluator
```

输入及输出挂载到 CPU 项目目录的 `eval/`。程序默认使用 `reader` 的授权范围及 `kb_education`，读取其实际已发布版本；需要先启动项目并发布资料。评估不自动上传测试资料、不发布文件、不修改知识库外发策略。

指定知识库或单条验证：

```bash
sudo -n docker compose --env-file .env.remote -f infra/compose.eval.yml run --rm --no-deps evaluator \
  --input /workspace/qa.csv --output /workspace/eval_result.csv \
  --user-id reader --kb-ids kb_education --limit 1
```

## 直接运行 Python

在数据库和 GPU 接口实际可达的主机安装独立依赖，运行：

```bash
uv sync --project eval --frozen
uv run --project eval --frozen python eval/eval.py --env-file .env.remote
```

Windows 下若直接使用当前 `.env.remote` 的 `127.0.0.1` 存储/模型地址，还需要相应 SSH 端口转发；网页网关的单个 18080 转发不覆盖这些连接。

## 输出及失败处理

CSV 列严格为：

```text
question,contexts,answer,ground_truth,Faithfulness,Answer Relevancy,Context Prcision,Context Recall,Answer Correctness
```

按用户要求保留 `Context Prcision` 列名；实际计算的是 **Context Precision**。CSV `contexts` 为 JSON 数组，支持中文、换行和引号，Python 中可用 `json.loads(row["contexts"])` 还原。

核心步骤函数使用 `step_1_` 至 `step_7_` 前缀，所有函数均有说明注释。输入为空、BOM/列头不正确或 RAG 未完成时返回失败；Ragas 某项判分失败则保存 `NaN`，不会填零或满分。退出码 `0` 表示所有指标有效，`2` 表示已输出但存在缺失指标，`1` 表示执行失败。

`eval_result.summary.json` 保存测试集哈希、模型/快照、实际重排引用、降级信息、指标均值与缺失数量，供复核。均值仅统计有效值，必须与缺失数量一起看。真实回答会按原系统规则保存为该评估账号的会话记录。

## 本次实际运行结果

2026-10-02 在远程 CPU 评估容器完成 5 条教育演示问题的真实评估，生成和判分均使用 GPU 上的 Qwen3-8B，语义指标使用 BGE-M3。每条问题获得 5 个实际重排片段，检索没有降级，25 个指标分数全部有效，退出码为 `0`。

| 指标 | 均值 | 缺失数量 |
|---|---:|---:|
| Faithfulness | 1.0000 | 0 |
| Answer Relevancy | 0.8314 | 0 |
| Context Prcision | 1.0000 | 0 |
| Context Recall | 1.0000 | 0 |
| Answer Correctness | 0.8241 | 0 |

逐题原始结果见 [eval_result.csv](eval_result.csv)，可复核的模型及重排引用见 [eval_result.summary.json](eval_result.summary.json)。已核对 UTF-8 BOM、九个输出列头、输入顺序、参考答案一致性、上下文 JSON 数组与分数范围。

“参加课程需要准备什么”这条问题的 Answer Correctness 为 `0.570748`，实际回答除电脑和 Python 3.12 外还包含课程安排、答疑及退款信息，可作为进一步人工复核和改进回答简洁性的样例。该结果仅反映本次小型演示集；同一模型参与生成及判分，后续应扩充题集并使用独立评判模型复测。

参考：[Ragas 模型适配](https://docs.ragas.io/en/v0.2.8/getstarted/rag_evaluation/)、[Ragas 0.4.3 评估实现](https://github.com/vibrantlabsai/ragas/blob/v0.4.3/src/ragas/evaluation.py)。
