from datetime import timedelta

import pytest

from knowledge_service.core.errors import ServiceError
from knowledge_service.core.security import digest, now
from knowledge_service.rag import answer_graph, search
from knowledge_service.schemas import SearchInput
from knowledge_service.worker import Worker


def assert_code(code, operation):
    with pytest.raises(ServiceError) as error:
        operation()
    assert error.value.code == code


def test_permission_scope(published):
    assert len(published.domain.scope("reader", ["kb"])) == 1
    assert_code("NOT_FOUND", lambda: published.domain.scope("outsider", ["kb"]))
    published.domain.acl("admin", "doc", "acl", ["admin"])
    assert published.domain.scope("reader", ["kb"]) == []
    assert published.domain.readable_chunks("reader", ["old_0"], True) == []


def test_transaction_rolls_back_and_idempotency(runtime):
    def operation(tx):
        tx.insert("audit_events", {"_id": "rollback"})
        raise ServiceError("FAIL", "failure")

    assert_code("FAIL", lambda: runtime.store.transaction(operation))
    assert not runtime.store.db.audit_events.find_one({"_id": "rollback"})
    args = ("maintainer", "kb", "same", "a.md", "source", "hash", {})
    first = runtime.domain.upload(*args)
    assert runtime.domain.upload(*args) == first
    assert runtime.store.db.document_versions.count_documents({}) == 1
    assert_code(
        "IDEMPOTENCY_CONFLICT",
        lambda: runtime.domain.upload("maintainer", "kb", "same", "b.md", "x", "x", {}),
    )


@pytest.mark.asyncio
async def test_markdown_parse_review_publish(runtime):
    source = "# 课程说明\n\n开课前可申请退款。\n".encode()
    runtime.objects.put(runtime.s.originals_bucket, "source", source)
    result = runtime.domain.upload("maintainer", "kb", "upload", "course.md", "source", digest(source), {})
    worker = Worker(runtime)
    await worker.execute(worker.claim())
    preview = runtime.domain.preview("reviewer", result["version_id"])
    assert preview["version"]["processing_status"] == "success"
    assert preview["chunks"][-1]["locator"]["line_start"] == 3
    runtime.domain.submit("maintainer", result["version_id"], "submit")
    runtime.domain.review("reviewer", result["version_id"], "review", "approved", "核对原文")
    runtime.domain.publish_request("reviewer", result["version_id"], "publish")
    await worker.execute(worker.claim())
    assert runtime.domain.scope("reader", ["kb"])[0]["_id"] == result["version_id"]
    assert runtime.vectors.writes


def test_submitter_cannot_review(runtime):
    result = runtime.domain.upload("admin", "kb", "upload", "x.md", "x", "h", {})
    runtime.store.db.document_versions.update_one(
        {"_id": result["version_id"]}, {"$set": {"processing_status": "success", "preview_hash": "h"}}
    )
    runtime.domain.submit("admin", result["version_id"], "submit")
    assert_code(
        "SELF_REVIEW", lambda: runtime.domain.review("admin", result["version_id"], "review", "approved", "")
    )


def prepare_index(r):
    r.store.db.document_versions.insert_one(
        {
            "_id": "new",
            "document_id": "doc",
            "kb_id": "kb",
            "state": "approved",
            "preview_hash": "h",
            "submitted_hash": "h",
            "graph_snapshot_key": "snapshot",
        }
    )
    r.store.db.chunks.insert_one(
        {
            "_id": "new_0",
            "version_id": "new",
            "kb_id": "kb",
            "ordinal": 0,
            "heading": "新政策",
            "text": "新政策",
            "locator": {"type": "text", "line_start": 1, "line_end": 1},
        }
    )
    r.domain.publish_request("reviewer", "new", "publish")
    worker = Worker(r)
    return worker, worker.claim()


@pytest.mark.asyncio
async def test_index_failure_keeps_old_version(published):
    worker, task = prepare_index(published)
    published.vectors.failure = ServiceError("SEARCH_NOT_READY", "not ready", 503)
    await worker.execute(task)
    assert published.domain.scope("reader", ["kb"])[0]["_id"] == "old"
    assert published.store.db.processing_tasks.find_one({"_id": task["_id"]})["status"] == "failed"


def test_lease_takeover_rejects_old_attempt(published):
    worker, task = prepare_index(published)
    published.store.db.processing_tasks.update_one(
        {"_id": task["_id"]}, {"$set": {"lease_until": now() - timedelta(seconds=1)}}
    )
    new_worker = Worker(published)
    replacement = new_worker.claim()
    assert replacement["fencing_token"] == task["fencing_token"] + 1
    assert_code("LEASE_LOST", lambda: worker.mark(task, "success"))
    assert published.store.db.documents.find_one({"_id": "doc"})["active_version_id"] == "old"


def test_revocation_hides_existing_history(published):
    d = published.domain
    chat = d.create_conversation("reader", "chat", ["kb"])
    msg = d.reserve_answer("reader", chat["_id"], "answer", "秘密问题")
    published.store.db.messages.update_one({"_id": msg["_id"]}, {"$set": {"dependency_versions": ["old"]}})
    d.finish_answer(msg["_id"], "completed", "秘密回答")
    d.acl("admin", "doc", "acl", ["admin"])
    hidden = d.conversation("reader", chat["_id"])["messages"][0]
    assert hidden == {"_id": msg["_id"], "status": "access_revoked"}
    assert_code("NOT_FOUND", lambda: d.conversation("outsider", chat["_id"]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowed,response,status",
    [
        (False, "unused", "retrieval_only"),
        (True, "退款政策。[S1]", "completed"),
        (True, "编造。[S99]", "interrupted"),
    ],
)
async def test_answer_policies_and_citations(published, allowed, response, status):
    r = published
    r.store.db.knowledge_bases.update_one({"_id": "kb"}, {"$set": {"allow_llm_egress": allowed}})
    r.models.response = response
    chat = r.domain.create_conversation("reader", "chat", ["kb"])
    msg = r.domain.reserve_answer("reader", chat["_id"], "answer", "退款规则？")
    events = [
        e
        async for e in answer_graph(r).astream(
            {"user_id": "reader", "message_id": msg["_id"]},
            {"configurable": {"thread_id": msg["thread_id"]}},
            stream_mode="custom",
        )
    ]
    assert events
    assert r.store.db.messages.find_one({"_id": msg["_id"]})["status"] == status
    assert r.models.generate_calls == int(allowed)
    assert r.store.db.quota_leases.find_one({"_id": msg["slot_id"]})["request_id"] is None


@pytest.mark.asyncio
async def test_embedding_failure_falls_back_to_bm25(published):
    async def failing(texts):
        raise ServiceError("BGE_NOT_CONFIGURED", "unavailable", 503)

    published.models.embed = failing
    result = await search(published, "reader", SearchInput(kb_ids=["kb"], question="退款"))
    assert result["candidates"][0]["_id"] == "old_0"
    assert result["degraded"] == ["embedding_unavailable_bm25"]


@pytest.mark.asyncio
async def test_housekeeping_releases_crashed_generation(runtime):
    chat = runtime.domain.create_conversation("reader", "chat", ["kb"])
    m = runtime.domain.reserve_answer("reader", chat["_id"], "answer", "question")
    runtime.store.db.messages.update_one(
        {"_id": m["_id"]}, {"$set": {"created_at": now() - timedelta(minutes=6)}}
    )
    await Worker(runtime).housekeeping()
    assert runtime.store.db.messages.find_one({"_id": m["_id"]})["status"] == "interrupted"
    assert runtime.store.db.quota_leases.find_one({"_id": m["slot_id"]})["request_id"] is None


@pytest.mark.parametrize("fault", ["dense_ready", "sparse_ready", "bm25_ready", "graph_ready"])
def test_publish_rejects_incomplete_manifest(published, fault):
    worker, task = prepare_index(published)
    index = {k: True for k in ("dense_ready", "sparse_ready", "bm25_ready", "search_ready")}
    graph = {"graph_ready": True}
    (graph if fault == "graph_ready" else index)[fault] = False
    assert_code(
        "NOT_READY",
        lambda: worker.guarded(task, lambda tx: published.domain.publish_commit(tx, task, index, graph)),
    )
    assert published.store.db.documents.find_one({"_id": "doc"})["active_version_id"] == "old"
    assert published.store.db.document_versions.find_one({"_id": "old"})["state"] == "published"


def test_publish_cas_and_replay_after_revocation(published):
    worker, task = prepare_index(published)
    published.domain.acl("admin", "doc", "acl", None)
    index = {k: True for k in ("dense_ready", "sparse_ready", "bm25_ready", "search_ready")}
    assert_code(
        "PUBLISH_CONFLICT",
        lambda: worker.guarded(
            task, lambda tx: published.domain.publish_commit(tx, task, index, {"graph_ready": True})
        ),
    )
    published.domain.member("admin", "kb", "reviewer", None, "revoke")
    assert_code("NOT_FOUND", lambda: published.domain.publish_request("reviewer", "new", "publish"))


@pytest.mark.asyncio
async def test_outbox_physically_redacts_revoked_history(published):
    r = published
    chat = r.domain.create_conversation("reader", "chat", ["kb"])
    m = r.domain.reserve_answer("reader", chat["_id"], "answer", "private question")
    r.store.db.messages.update_one({"_id": m["_id"]}, {"$set": {"dependency_versions": ["old"]}})
    r.domain.finish_answer(m["_id"], "completed", "private answer")
    r.store.db.feedback_tickets.insert_one(
        {"_id": "feedback", "message_id": m["_id"], "detail": "private feedback"}
    )
    r.domain.acl("admin", "doc", "acl", ["admin"])
    await Worker(r).housekeeping()
    stored = r.store.db.messages.find_one({"_id": m["_id"]})
    assert stored["question"] == stored["answer"] == "" and stored["sources"] == []
    assert stored["status"] == "access_revoked"
    assert r.store.db.feedback_tickets.find_one({"_id": "feedback"})["detail"] == ""
    assert r.store.db.idempotency_records.count_documents({"result._id": m["_id"]}) == 0
    assert r.store.db.outbox_events.count_documents({"status": "pending"}) == 0


@pytest.mark.asyncio
async def test_permission_removed_during_generation_clears_answer(published):
    r = published
    r.store.db.knowledge_bases.update_one({"_id": "kb"}, {"$set": {"allow_llm_egress": True}})

    async def revoking(*args):
        yield "开头。[S1]"
        r.domain.acl("admin", "doc", "acl", ["admin"])
        yield "不得展示。"

    r.models.generate = revoking
    chat = r.domain.create_conversation("reader", "chat", ["kb"])
    m = r.domain.reserve_answer("reader", chat["_id"], "answer", "退款？")
    events = [
        e
        async for e in answer_graph(r).astream(
            {"user_id": "reader", "message_id": m["_id"]},
            {"configurable": {"thread_id": m["thread_id"]}},
            stream_mode="custom",
        )
    ]
    assert events[-1]["event"] == "access_revoked"
    assert not any(e.get("text") == "不得展示。" for e in events)
    stored = r.store.db.messages.find_one({"_id": m["_id"]})
    assert stored["status"] == "access_revoked" and stored["answer"] == ""


@pytest.mark.asyncio
async def test_orphan_cleanup_preserves_linked_artifacts(published):
    r = published
    created = now() - timedelta(minutes=11)
    for key in ("linked", "orphan"):
        r.objects.put(r.s.derived_bucket, key, b"bytes")
        r.store.db.orphan_candidates.insert_one(
            {"_id": key, "bucket": r.s.derived_bucket, "key": key, "created_at": created}
        )
    r.store.db.document_versions.update_one(
        {"_id": "old"}, {"$set": {"artifact_keys": [{"key": "linked", "bucket": r.s.derived_bucket}]}}
    )
    await Worker(r).housekeeping()
    assert (r.s.derived_bucket, "linked") in r.objects.data
    assert (r.s.derived_bucket, "orphan") not in r.objects.data
