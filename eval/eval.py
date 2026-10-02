"""使用真实 RAG 图状态与 Ragas 评估 UTF-8 BOM 问答测试集。"""

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

# 禁止评估遥测；评估资料仅发送给明确配置的模型接口。
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
os.environ.setdefault("LANGSMITH_TRACING", "false")

ROOT = Path(__file__).resolve().parents[1]
METRIC_COLUMNS = {
    "Faithfulness": "faithfulness",
    "Answer Relevancy": "answer_relevancy",
    "Context Prcision": "context_precision",  # 按要求保留输出列名中的拼写。
    "Context Recall": "context_recall",
    "Answer Correctness": "answer_correctness",
}
OUTPUT_COLUMNS = ["question", "contexts", "answer", "ground_truth", *METRIC_COLUMNS]


def step_1_load_configuration(argv=None):
    """读取命令行和私有环境配置，不打印连接口令或模型密钥。"""
    from knowledge_service.core.config import Settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "eval/qa.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/eval_result.csv")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.remote")
    parser.add_argument("--user-id", default="reader")
    parser.add_argument("--kb-ids", nargs="+", default=["kb_education"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if (
        args.timeout < 1
        or args.max_workers < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("timeout、max-workers 和 limit 必须为正整数")
    settings = Settings(_env_file=args.env_file if args.env_file.is_file() else None)
    if not settings.llm_api_key.get_secret_value():
        raise ValueError("未配置 LLM_API_KEY")
    if not settings.bge_model_revision:
        raise ValueError("未配置真实 BGE_MODEL_REVISION")
    return args, settings


def step_2_load_test_dataset(path, limit=None):
    """校验 UTF-8 BOM、两列列头及非空问题和参考答案，保留原始顺序。"""
    if not path.is_file():
        raise FileNotFoundError(f"测试集不存在：{path}")
    if not path.read_bytes().startswith(b"\xef\xbb\xbf"):
        raise ValueError("qa.csv 必须采用 UTF-8 BOM 编码")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["question", "ground_truth"]:
            raise ValueError("CSV 列头必须依次为 question,ground_truth")
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(
                not isinstance(row.get(k), str) or not row[k].strip()
                for k in ("question", "ground_truth")
            ):
                raise ValueError(f"CSV 第 {number} 条记录为空或列数不正确")
            rows.append({k: row[k].strip() for k in ("question", "ground_truth")})
    if not rows:
        raise ValueError("评估测试集没有数据")
    return rows[:limit] if limit else rows


async def step_3_collect_rag_samples(runtime, test_rows, user_id, kb_ids, timeout):
    """调用项目原 RAG 图，从 state.reranked_docs 解析实际有序证据并读取真实回答。"""
    from knowledge_service.rag import answer_graph

    graph = answer_graph(runtime)
    samples, traces = [], []
    for index, row in enumerate(test_rows, 1):
        chat = await asyncio.to_thread(
            runtime.domain.create_conversation,
            user_id,
            "eval-chat-" + uuid.uuid4().hex,
            kb_ids,
        )
        message = await asyncio.to_thread(
            runtime.domain.reserve_answer,
            user_id,
            chat["_id"],
            "eval-answer-" + uuid.uuid4().hex,
            row["question"],
        )
        try:
            async with asyncio.timeout(timeout):
                state = await graph.ainvoke(
                    {"message_id": message["_id"], "user_id": user_id},
                    {"configurable": {"thread_id": message["thread_id"]}},
                )
            # 只使用图真实返回的重排顺序，不重新召回，不用 ground_truth 填写 contexts。
            ids = state["reranked_docs"]
            if ids != state["selected_ids"] or len(set(ids)) != len(ids):
                raise ValueError("图状态中的实际生成证据与重排顺序不一致")
            docs = await asyncio.to_thread(
                runtime.domain.readable_chunks, user_id, ids, True
            )
            lookup = {doc["_id"]: doc for doc in docs}
            if set(lookup) != set(ids):
                raise ValueError("评估证据已不可访问")
            stored = await asyncio.to_thread(
                runtime.store.read, lambda tx: tx.get("messages", message["_id"])
            )
            if stored["status"] != "completed" or not stored["answer"].strip():
                raise ValueError("RAG 生成未完成或资料生成策略未开启")
            samples.append(
                {
                    **row,
                    "contexts": [lookup[key]["text"] for key in ids],
                    "answer": stored["answer"],
                }
            )
            traces.append(
                {
                    "message_id": message["_id"],
                    "thread_id": message["thread_id"],
                    "reranked_docs": ids,
                    "degraded": state.get("degraded", []),
                }
            )
            print(f"RAG {index}/{len(test_rows)}：{len(ids)} 个重排片段", flush=True)
        finally:
            # 异常、中断也释放任务配额；完成的回答不会被再次覆盖。
            await asyncio.to_thread(
                runtime.domain.finish_answer, message["_id"], "interrupted"
            )
    return samples, traces


def step_4_build_ragas_models(settings, timeout):
    """配置 Ragas 的真实 Qwen 判分模型及 BGE-M3 1024 维稠密嵌入适配器。"""
    import httpx
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    from knowledge_service.adapters.models import Models

    class BgeM3Embeddings(Embeddings):
        """为 Ragas 语义指标复用原 BGE 接口、快照与向量校验。"""

        def embed_documents(self, texts):
            """在独立线程桥接同步嵌入，兼容 Ragas 在运行中事件循环内的同步调用。"""
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, self.aembed_documents(texts)).result()

        def embed_query(self, text):
            """同步计算一条文本的 BGE-M3 稠密向量。"""
            return self.embed_documents([text])[0]

        async def aembed_documents(self, texts):
            """按服务允许的批量大小调用 BGE，并安全关闭当前事件循环的客户端。"""
            models = Models(settings)
            vectors = []
            try:
                for offset in range(0, len(texts), 32):
                    dense, _, _ = await models.embed(texts[offset : offset + 32])
                    vectors.extend(dense)
            finally:
                await models.close()
            return vectors

        async def aembed_query(self, text):
            """异步计算一条文本的 BGE-M3 稠密向量。"""
            return (await self.aembed_documents([text]))[0]

    # 可指定独立判分模型；默认复用已有自建 Qwen，无隐含 OpenAI 默认服务。
    api_base = os.environ.get("EVAL_LLM_BASE_URL") or settings.llm_base_url
    key = os.environ.get("EVAL_LLM_API_KEY") or settings.llm_api_key.get_secret_value()
    model = os.environ.get("EVAL_LLM_MODEL") or settings.llm_model
    extra_body = {}
    if (
        api_base == settings.llm_base_url
        and settings.llm_provider == "vllm"
        and settings.llm_disable_thinking
    ):
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    http_client = httpx.Client(timeout=timeout)
    async_client = httpx.AsyncClient(timeout=timeout)
    llm = ChatOpenAI(
        model=model,
        base_url=api_base,
        api_key=key,
        temperature=0,
        max_tokens=1024,
        max_retries=2,
        timeout=timeout,
        extra_body=extra_body,
        use_responses_api=False,
        http_client=http_client,
        http_async_client=async_client,
    )
    run_config = RunConfig(timeout=timeout, max_retries=2, max_workers=1)
    return (
        LangchainLLMWrapper(llm, run_config=run_config),
        LangchainEmbeddingsWrapper(BgeM3Embeddings(), run_config=run_config),
        http_client,
        async_client,
        model,
    )


async def step_5_evaluate_with_ragas(samples, llm, embeddings, timeout, max_workers):
    """由 Ragas 计算五项原始指标，不将失败的 NaN 改成零分或满分。"""
    from ragas import EvaluationDataset, aevaluate
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": row["question"],
                "retrieved_contexts": row["contexts"],
                "response": row["answer"],
                "reference": row["ground_truth"],
            }
            for row in samples
        ]
    )
    # Answer Correctness 使用其默认事实判定/语义相似度权重，不替换成其他指标。
    result = await aevaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            AnswerCorrectness(),
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=timeout, max_retries=2, max_workers=max_workers),
        raise_exceptions=False,
        show_progress=True,
    )
    scores = result.scores
    if len(scores) != len(samples):
        raise ValueError("Ragas 返回的评估行数与测试集不一致")
    return [
        {
            **sample,
            **{label: float(score[name]) for label, name in METRIC_COLUMNS.items()},
        }
        for sample, score in zip(samples, scores, strict=True)
    ]


def step_6_write_result_csv(rows, path):
    """以 UTF-8 BOM 原子保存结果，contexts 采用 JSON 数组保留顺序及换行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for row in rows:
                output = {
                    **row,
                    "contexts": json.dumps(row["contexts"], ensure_ascii=False),
                }
                for name in METRIC_COLUMNS:
                    value = row[name]
                    output[name] = (
                        format(value, ".6f") if math.isfinite(value) else "NaN"
                    )
                writer.writerow(output)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def step_7_write_summary(rows, traces, args, settings, judge_model):
    """保存可追溯的模型、测试集摘要、重排引用及缺失指标数量，不记录密钥。"""
    means, failures = {}, {}
    for name in METRIC_COLUMNS:
        valid = [row[name] for row in rows if math.isfinite(row[name])]
        means[name] = sum(valid) / len(valid) if valid else None
        failures[name] = len(rows) - len(valid)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "questions": len(rows),
        "dataset_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "user_id": args.user_id,
        "kb_ids": args.kb_ids,
        "ragas_version": "0.4.3",
        "generation_model": settings.llm_model,
        "judge_model": judge_model,
        "embedding_model": settings.bge_model,
        "embedding_revision": settings.bge_model_revision,
        "metric_means": means,
        "missing_scores": failures,
        "traces": traces,
        "limitations": [
            "小型教育演示测试集不代表生产效果",
            "默认判分与生成使用同一模型，可能存在偏差",
            "Context Prcision 是输出列名；实际指标为 Ragas Context Precision",
            "NaN 表示评估未产生有效分数；均值只统计有效分数",
        ],
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "questions": len(rows),
                "metric_means": means,
                "missing_scores": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return any(failures.values())


async def collect_samples(args, settings, test_rows):
    """管理项目 Runtime 的生命周期，保证实际模型和数据库连接及时关闭。"""
    from knowledge_service.runtime import Runtime

    runtime = Runtime(settings)
    try:
        return await step_3_collect_rag_samples(
            runtime, test_rows, args.user_id, args.kb_ids, args.timeout
        )
    finally:
        await runtime.close()


async def evaluate_samples(args, settings, samples):
    """在同一事件循环完成判分与客户端清理，避免已关闭循环上的连接泄漏。"""
    llm, embeddings, http_client, async_client, judge = step_4_build_ragas_models(
        settings, args.timeout
    )
    try:
        rows = await step_5_evaluate_with_ragas(
            samples, llm, embeddings, args.timeout, args.max_workers
        )
        return rows, judge
    finally:
        http_client.close()
        await async_client.aclose()


def main(argv=None):
    """顺序执行七个核心步骤；模型或数据错误仅输出脱敏类型，不伪造评估结果。"""
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    try:
        args, settings = step_1_load_configuration(argv)
        test_rows = step_2_load_test_dataset(args.input, args.limit)
        samples, traces = asyncio.run(collect_samples(args, settings, test_rows))
        rows, judge = asyncio.run(evaluate_samples(args, settings, samples))
        step_6_write_result_csv(rows, args.output)
        missing = step_7_write_summary(rows, traces, args, settings, judge)
        return 2 if missing else 0
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "code": getattr(error, "code", "EVALUATION_ERROR"),
                    "missing_module": getattr(error, "name", None)
                    if isinstance(error, ModuleNotFoundError)
                    else None,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
