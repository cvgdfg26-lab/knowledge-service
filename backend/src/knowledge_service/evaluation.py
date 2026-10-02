"""Retrieval measurements from actual runs; generation correctness remains a human review."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

from knowledge_service.core.errors import require
from knowledge_service.core.security import now
from knowledge_service.rag import search
from knowledge_service.schemas import SearchInput


def evidence_matches(chunk, evidence):
    if chunk["version_id"] != evidence["version_id"]:
        return False
    location = chunk["locator"]
    if "page_number" in evidence:
        return location.get("page_number") == evidence["page_number"]
    if "line_start" in evidence:
        return (
            location.get("line_start", 10**9) <= evidence["line_end"]
            and location.get("line_end", 0) >= evidence["line_start"]
        )
    return evidence.get("block_index") == location.get("block_index")


async def evaluate(runtime, config_path, split, output):
    require(config_path and output, "EVAL_CONFIG", "需要 --config 和 --output")
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes)
    data_bytes = Path(config["dataset"]).read_bytes()
    questions = [json.loads(line) for line in data_bytes.decode("utf-8").splitlines() if line.strip()]
    questions = [q for q in questions if q["split"] == split]
    require(questions, "EVAL_EMPTY", "该数据集切分无问题")
    require(len({q["id"] for q in questions}) == len(questions), "EVAL_IDS", "问题 ID 重复")
    results = []
    for question in questions:
        started = perf_counter()
        retrieved = await search(
            runtime,
            config["user_id"],
            SearchInput(
                kb_ids=config["kb_ids"],
                question=question["question"],
                mode=config["mode"],
                rerank=config["rerank"],
                limit=5,
            ),
        )
        evidence = question.get("evidence", [])
        hits = sum(any(evidence_matches(c, e) for c in retrieved["candidates"]) for e in evidence)
        results.append(
            {
                "question_id": question["id"],
                "hit_at_5": bool(hits),
                "recall_at_5": hits / len(evidence) if evidence else None,
                "answerable": bool(evidence),
                "seconds": perf_counter() - started,
                "candidate_ids": [c["_id"] for c in retrieved["candidates"]],
                "degraded": retrieved["degraded"],
                "timings": retrieved["timings"],
            }
        )
    answerable = [r for r in results if r["answerable"]]
    require(answerable, "EVAL_NO_EVIDENCE", "至少需要一道人工标注来源的问题")
    latencies = sorted(r["seconds"] for r in results)
    payload = {
        "created_at": now().isoformat(),
        "split": split,
        "config": config,
        "dataset_hash": hashlib.sha256(data_bytes).hexdigest(),
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "model_revision": runtime.s.bge_model_revision,
        "collection": runtime.s.milvus_collection,
        "generation": runtime.s.generation,
        "metrics": {
            "questions": len(results),
            "answerable": len(answerable),
            "hit_at_5": sum(r["hit_at_5"] for r in answerable) / len(answerable),
            "recall_at_5": sum(r["recall_at_5"] for r in answerable) / len(answerable),
            "latency_p95_seconds": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        },
        "results": results,
        "limitations": ["不调用生成 API", "无答案正确率结论", "无资料题单独计数", "延迟为当前环境实测"],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False))


def compare(paths):
    require(paths, "COMPARE_INPUT", "需要 --runs")
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    require(
        len({(r["dataset_hash"], r["split"]) for r in runs}) == 1, "COMPARE_DATASET", "只能比较同一数据集切分"
    )
    print(
        json.dumps(
            [
                {
                    "config": r["config"],
                    "metrics": r["metrics"],
                    "degraded_questions": sum(bool(q["degraded"]) for q in r["results"]),
                }
                for r in runs
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
