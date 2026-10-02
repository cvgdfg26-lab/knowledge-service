import asyncio
import re
from collections import defaultdict
from time import perf_counter
from typing import TypedDict

import httpx
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from knowledge_service.core.errors import ServiceError, require
from knowledge_service.schemas import SearchInput


async def search(runtime, user_id, query: SearchInput):
    versions = await asyncio.to_thread(runtime.domain.scope, user_id, query.kb_ids)
    if not versions:
        return {"candidates": [], "degraded": [], "timings": {}}
    groups = defaultdict(list)
    for v in versions:
        groups[v["collection"]].append(v["_id"])
    dense, sparse, degraded, timings = None, None, [], {}
    mode = query.mode
    if mode != "bm25":
        try:
            d, sp, elapsed = await runtime.models.embed([query.question])
            dense, sparse = d[0], sp[0]
            timings["embedding_seconds"] = elapsed
        except (httpx.HTTPError, ServiceError):
            mode = "bm25"
            degraded.append("embedding_unavailable_bm25")
    ranks = {}
    start = perf_counter()
    for collection, ids in groups.items():
        manifests = await asyncio.to_thread(
            runtime.store.read, lambda tx, group=ids: [tx.get("index_manifests", vid) for vid in group]
        )
        group_mode = mode
        if mode != "bm25" and any(
            not m or m.get("model_revision") != runtime.s.bge_model_revision for m in manifests
        ):
            group_mode = "bm25"
            if "legacy_revision_bm25" not in degraded:
                degraded.append("legacy_revision_bm25")
        rows = await asyncio.to_thread(
            runtime.vectors.search, collection, ids, query.question, dense, sparse, group_mode
        )
        for rank, row in enumerate(rows, 1):
            ranks[row["chunk_id"]] = ranks.get(row["chunk_id"], 0) + 1 / (60 + rank)
    timings["retrieval_seconds"] = perf_counter() - start
    candidates = await asyncio.to_thread(
        runtime.domain.readable_chunks, user_id, sorted(ranks, key=ranks.get, reverse=True)[:30]
    )
    # Authoritative chunk text is read from MongoDB only after current authorization.
    if candidates and query.rerank:
        try:
            start = perf_counter()
            candidates = await runtime.models.rerank(query.question, candidates)
            timings["rerank_seconds"] = perf_counter() - start
        except (httpx.HTTPError, ServiceError):
            degraded.append("rerank_unavailable")
    safe = await asyncio.to_thread(runtime.domain.readable_chunks, user_id, [c["_id"] for c in candidates])
    lookup = {c["_id"]: c for c in safe}
    return {
        "candidates": [lookup[c["_id"]] for c in candidates if c["_id"] in lookup][: query.limit],
        "degraded": degraded,
        "timings": timings,
    }


def validate_citations(text, count):
    labels = [int(n) for n in re.findall(r"\[S(\d+)\]", text)]
    require(
        labels and all(1 <= n <= count for n in labels),
        "INVALID_CITATIONS",
        "回答未提供有效引用，结果标为不完整",
        422,
    )


class AnswerState(TypedDict, total=False):
    message_id: str
    user_id: str
    selected_ids: list[str]
    # Keep ordered references rather than plaintext in persisted checkpoints.
    reranked_docs: list[str]
    degraded: list[str]


def answer_graph(runtime):
    async def retrieve(state):
        message = await asyncio.to_thread(
            runtime.store.read, lambda tx: tx.get("messages", state["message_id"])
        )
        await asyncio.to_thread(runtime.domain.conversation, state["user_id"], message["session_id"])
        result = await search(
            runtime, state["user_id"], SearchInput(kb_ids=message["kb_ids"], question=message["question"])
        )
        selected = result["candidates"]
        await asyncio.to_thread(
            runtime.store.transaction,
            lambda tx: tx.patch(
                "messages",
                message["_id"],
                {
                    "status": "generating",
                    "dependency_versions": sorted({c["version_id"] for c in selected}),
                    "timings": result["timings"],
                    "degraded": result["degraded"],
                },
                {"status": "pending"},
            ),
        )
        get_stream_writer()({"event": "status", "stage": "retrieval_complete"})
        ids = [c["_id"] for c in selected]
        return {"selected_ids": ids, "reranked_docs": ids, "degraded": result["degraded"]}

    async def generate(state):
        emit = get_stream_writer()
        message = await asyncio.to_thread(
            runtime.store.read, lambda tx: tx.get("messages", state["message_id"])
        )
        user_id, message_id = state["user_id"], message["_id"]
        selected = await asyncio.to_thread(
            runtime.domain.readable_chunks, user_id, state["selected_ids"], True
        )
        require(len(selected) == len(state["selected_ids"]), "ACCESS_REVOKED", "资料访问权限已变化", 403)
        if not selected:
            answer = "当前授权资料不足以回答这个问题，请补充资料或明确查询范围。"
            await asyncio.to_thread(runtime.domain.finish_answer, message_id, "completed", answer)
            emit({"event": "delta", "text": answer})
            emit({"event": "done", "status": "completed", "behavior": "evidence_insufficient"})
            return {}
        sources = [
            {
                "label": f"S{i}",
                "chunk_id": c["_id"],
                "version_id": c["version_id"],
                "locator": c["locator"],
                "heading": c["heading"],
            }
            for i, c in enumerate(selected, 1)
        ]
        emit({"event": "sources", "sources": sources})
        for flag in state["degraded"]:
            emit({"event": "warning", "code": flag})
        for c in selected:
            kb = await asyncio.to_thread(
                runtime.store.read, lambda tx, row=c: runtime.domain.access(tx, user_id, row["kb_id"])
            )
            if not kb["allow_llm_egress"]:
                await asyncio.to_thread(
                    runtime.domain.finish_answer, message_id, "retrieval_only", "", sources
                )
                emit(
                    {
                        "event": "warning",
                        "code": "llm_egress_blocked",
                        "message": "资料禁止外发，仅提供授权来源",
                    }
                )
                emit({"event": "done", "status": "retrieval_only"})
                return {}
        context = "\n\n".join(
            f"[S{i}] 版本 {c['version_id']} / {c['heading']}\n{c['text']}" for i, c in enumerate(selected, 1)
        )
        history = await asyncio.to_thread(runtime.domain.conversation, user_id, message["session_id"])
        prior = [m for m in history["messages"] if m.get("status") == "completed" and m["_id"] != message_id][
            -2:
        ]
        # History can contain earlier evidence from the same scope; re-check its provider policy too.
        for m in prior:
            for version_id in m.get("dependency_versions", []):
                v = await asyncio.to_thread(
                    runtime.store.read,
                    lambda tx, key=version_id: runtime.domain.version(tx, user_id, key, True),
                )
                kb = await asyncio.to_thread(
                    runtime.store.read, lambda tx, row=v: runtime.domain.access(tx, user_id, row["kb_id"])
                )
                require(kb["allow_llm_egress"], "LLM_EGRESS", "历史资料禁止外发", 403)
        # Only previous questions guide follow-ups; previous generated facts do not become evidence.
        history_prompt = [
            {"role": "user", "content": "此前问题（仅用于理解追问）：" + m["question"]} for m in prior
        ]
        dependencies = sorted(
            {c["version_id"] for c in selected} | {v for m in prior for v in m.get("dependency_versions", [])}
        )
        await asyncio.to_thread(
            runtime.store.transaction,
            lambda tx: tx.patch(
                "messages", message_id, {"dependency_versions": dependencies}, {"status": "generating"}
            ),
        )

        def check_provider_scope(tx):
            for version_id in dependencies:
                v = runtime.domain.version(tx, user_id, version_id, True)
                kb = runtime.domain.access(tx, user_id, v["kb_id"])
                require(kb["allow_llm_egress"], "LLM_EGRESS", "资料禁止继续外发", 403)

        answer = ""
        try:
            await asyncio.to_thread(runtime.store.read, check_provider_scope)
            async for text in runtime.models.generate(message["question"], context, history_prompt):
                await asyncio.to_thread(runtime.store.read, check_provider_scope)
                fresh = await asyncio.to_thread(runtime.store.read, lambda tx: tx.get("messages", message_id))
                require(not fresh["cancel_requested"], "CANCELLED", "生成已取消", 409)
                safe = await asyncio.to_thread(
                    runtime.domain.readable_chunks, user_id, state["selected_ids"], True
                )
                require(len(safe) == len(selected), "ACCESS_REVOKED", "资料访问权限已变化", 403)
                await asyncio.to_thread(runtime.domain.conversation, user_id, message["session_id"])
                answer += text
                require(len(answer) <= 12000, "ANSWER_LIMIT", "生成长度超过限制", 503)
                emit({"event": "delta", "text": text})
            validate_citations(answer, len(sources))
            safe = await asyncio.to_thread(
                runtime.domain.readable_chunks, user_id, state["selected_ids"], True
            )
            require(len(safe) == len(selected), "ACCESS_REVOKED", "资料访问权限已变化", 403)
            await asyncio.to_thread(runtime.domain.finish_answer, message_id, "completed", answer, sources)
            emit({"event": "done", "status": "completed"})
        except (ServiceError, httpx.HTTPError) as error:
            code = error.code if isinstance(error, ServiceError) else "LLM_UNAVAILABLE"
            revoked = code in {"ACCESS_REVOKED", "NOT_FOUND", "UNAUTHENTICATED", "LLM_EGRESS"}
            cancelled = code == "CANCELLED"
            status = "access_revoked" if revoked else "cancelled" if cancelled else "interrupted"
            await asyncio.to_thread(
                runtime.domain.finish_answer, message_id, status, "" if revoked else answer, sources
            )
            emit(
                {
                    "event": "access_revoked" if revoked else "cancelled" if cancelled else "error",
                    "code": code,
                }
            )
        return {}

    graph = StateGraph(AnswerState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile(checkpointer=runtime.checkpointer)
