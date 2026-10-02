import asyncio
import json
import logging
import mimetypes
import os
from datetime import timedelta
from pathlib import Path
from typing import TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from knowledge_service.core.errors import ServiceError, require
from knowledge_service.core.security import digest, now, uid
from knowledge_service.ingestion import split_text

log = logging.getLogger("knowledge.worker")


class TaskState(TypedDict):
    task_id: str
    fencing_token: int
    kind: str


class Worker:
    def __init__(self, runtime):
        self.r = runtime
        self.owner = f"worker-{os.getpid()}-{uid('run')}"

    def claim(self):
        def operation(tx):
            tasks = tx.find(
                "processing_tasks",
                {
                    "$or": [
                        {"status": {"$in": ["pending", "waiting_external"]}, "next_run_at": {"$lte": now()}},
                        {"status": "running", "lease_until": {"$lte": now()}},
                    ]
                },
                limit=1,
            )
            if not tasks:
                return None
            task = tasks[0]
            return tx.patch(
                "processing_tasks",
                task["_id"],
                {
                    "status": "running",
                    "owner": self.owner,
                    "fencing_token": task["fencing_token"] + 1,
                    "lease_until": now() + timedelta(seconds=self.r.s.task_lease_seconds),
                    "attempts": task["attempts"] + (task["status"] != "waiting_external"),
                },
                {"status": task["status"], "fencing_token": task["fencing_token"]},
            )

        return self.r.store.transaction(operation)

    def guarded(self, task, callback):
        def operation(tx):
            self.r.domain.assert_lease(tx, task)
            return callback(tx)

        return self.r.store.transaction(operation)

    async def heartbeat(self, task):
        while True:
            await asyncio.sleep(self.r.s.task_lease_seconds / 3)
            await asyncio.to_thread(
                self.guarded,
                task,
                lambda tx: tx.patch(
                    "processing_tasks",
                    task["_id"],
                    {"lease_until": now() + timedelta(seconds=self.r.s.task_lease_seconds)},
                ),
            )

    def mark(self, task, status, **extra):
        return self.guarded(
            task, lambda tx: tx.patch("processing_tasks", task["_id"], {"status": status, **extra})
        )

    async def parse(self, task):
        r, s = self.r, self.r.s
        v = await asyncio.to_thread(r.store.read, lambda tx: tx.get("document_versions", task["version_id"]))
        require(v and v["state"] == "draft", "PARSE_STATE", "版本不允许解析")
        if v["processing_status"] == "success":
            await asyncio.to_thread(self.mark, task, "success")
            return
        extension = Path(v["filename"]).suffix.lower()
        source = await asyncio.to_thread(r.objects.get, s.originals_bucket, v["source_key"])
        require(digest(source) == v["source_hash"], "SOURCE_HASH", "原文摘要不一致")
        files, blocks = {}, None
        if extension in {".md", ".txt"}:
            text = source.decode("utf-8-sig")
        else:

            def check_parser(tx):
                active = r.domain.document(tx, v["created_by"], v["document_id"], "maintain")
                current_kb = r.domain.access(tx, v["created_by"], active["kb_id"], "maintain")
                require(
                    not active["blocked_at"] and current_kb["allow_parser_egress"],
                    "PARSER_EGRESS",
                    "该资料不允许外部解析",
                    403,
                )

            await asyncio.to_thread(self.guarded, task, check_parser)
            submission = await asyncio.to_thread(
                r.store.read, lambda tx: tx.get("parser_submissions", v["_id"])
            )
            if not submission:
                await asyncio.to_thread(
                    self.guarded,
                    task,
                    lambda tx: tx.insert(
                        "parser_submissions",
                        {
                            "_id": v["_id"],
                            "status": "submitting",
                            "created_at": now(),
                            "data_id": v["_id"],
                        },
                    ),
                )
                await asyncio.to_thread(self.guarded, task, check_parser)
                try:
                    batch_id, url = await r.mineru.submit(v["filename"], v["_id"])
                except Exception:
                    await asyncio.to_thread(
                        self.guarded,
                        task,
                        lambda tx: tx.patch("parser_submissions", v["_id"], {"status": "submission_unknown"}),
                    )
                    raise ServiceError(
                        "SUBMISSION_UNKNOWN", "远程提交结果未知，需要核对，禁止自动重复提交", 503
                    )
                submission = await asyncio.to_thread(
                    self.guarded,
                    task,
                    lambda tx: tx.patch(
                        "parser_submissions",
                        v["_id"],
                        {"status": "upload_pending", "batch_id": batch_id, "upload_url": url},
                    ),
                )
            require(
                submission["status"] not in {"submitting", "submission_unknown"},
                "SUBMISSION_UNKNOWN",
                "远程提交结果未知，需要人工核对",
                503,
            )
            require(
                submission["created_at"] + timedelta(seconds=s.mineru_task_timeout_seconds) > now(),
                "PARSER_TIMEOUT",
                "解析任务超过总等待时间",
                503,
            )
            if submission["status"] == "upload_pending":
                await asyncio.to_thread(self.guarded, task, check_parser)
                await r.mineru.upload(submission["upload_url"], source)
                await asyncio.to_thread(
                    self.guarded,
                    task,
                    lambda tx: tx.patch(
                        "parser_submissions", v["_id"], {"status": "polling", "upload_url": None}
                    ),
                )
            result = await r.mineru.poll(submission["batch_id"])
            require(result.get("data_id") == v["_id"], "PARSER_MAPPING", "解析结果不匹配资料版本", 503)
            require(
                result.get("state") in {"waiting-file", "pending", "running", "converting", "failed", "done"},
                "PARSER_RESPONSE",
                "解析任务状态不识别",
                503,
            )
            require(result["state"] != "failed", "PARSER_FAILED", "远程解析失败", 422)
            if result["state"] != "done":
                await asyncio.to_thread(
                    self.mark,
                    task,
                    "waiting_external",
                    next_run_at=now() + timedelta(seconds=s.mineru_poll_interval_seconds),
                )
                return
            files, text, blocks = await r.mineru.download(result["full_zip_url"])
        chunks = split_text(text, v["_id"], v["kb_id"], blocks, source_format=extension.lstrip("."))
        require(len(chunks) <= 5000, "CHUNK_LIMIT", "单文档片段超限，需拆分后导入", 422)
        artifact_keys = []
        base = f"{v['kb_id']}/{v['_id']}/{task['fencing_token']}"
        for filename, data in {**files, "normalized.md": text.encode()}.items():
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            image = content_type.startswith("image/") and not filename.endswith(".svg")
            bucket = s.images_bucket if image else s.derived_bucket
            key = base + "/" + filename
            await asyncio.to_thread(
                self.guarded,
                task,
                lambda tx, b=bucket, k=key: tx.insert(
                    "orphan_candidates", {"_id": uid("orphan"), "bucket": b, "key": k, "created_at": now()}
                ),
            )
            await asyncio.to_thread(r.objects.put, bucket, key, data, content_type)
            artifact_keys.append(
                {
                    "bucket": bucket,
                    "key": key,
                    "hash": digest(data),
                    "asset_id": digest(v["_id"] + filename) if image else None,
                    "content_type": content_type,
                }
            )
        preview_hash = digest(
            json.dumps(
                {"source_hash": v["source_hash"], "chunks": chunks, "metadata": v["metadata"]},
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        facts_key = base + "/graph-facts.json"
        await asyncio.to_thread(
            self.guarded,
            task,
            lambda tx: tx.insert(
                "orphan_candidates",
                {"_id": uid("orphan"), "bucket": s.derived_bucket, "key": facts_key, "created_at": now()},
            ),
        )
        await asyncio.to_thread(
            r.objects.put,
            s.derived_bucket,
            facts_key,
            json.dumps(chunks, ensure_ascii=False).encode(),
            "application/json",
        )
        artifact_keys.append({"bucket": s.derived_bucket, "key": facts_key})

        def persist(tx):
            version = tx.get("document_versions", v["_id"])
            require(version["state"] == "draft", "PARSE_STATE", "版本已改变")
            tx.delete("chunks", {"version_id": v["_id"]})
            for chunk in chunks:
                tx.insert("chunks", chunk)
            tx.patch(
                "document_versions",
                v["_id"],
                {
                    "processing_status": "success",
                    "preview_hash": preview_hash,
                    "artifact_keys": artifact_keys,
                    "graph_snapshot_key": facts_key,
                    "split_config": {"unit": "characters", "max": 800, "overlap": 60},
                },
            )
            tx.patch("processing_tasks", task["_id"], {"status": "success"})

        await asyncio.to_thread(self.guarded, task, persist)

    async def index(self, task):
        r, s = self.r, self.r.s
        v = await asyncio.to_thread(r.store.read, lambda tx: tx.get("document_versions", task["version_id"]))
        require(v and v["state"] == "approved", "INDEX_STATE", "版本不允许准备索引")
        chunks = await asyncio.to_thread(
            r.store.read, lambda tx: tx.find("chunks", {"version_id": v["_id"]}, 5000)
        )

        def register_copy(tx):
            current = tx.get("document_versions", v["_id"])
            doc = r.domain.document(tx, task["requested_by"], v["document_id"], "review")
            require(current["state"] == "approved" and not doc["blocked_at"], "INDEX_STATE", "版本已被阻断")
            copies = current.get("index_copies", [])
            if s.milvus_collection not in copies:
                tx.patch("document_versions", v["_id"], {"index_copies": [*copies, s.milvus_collection]})

        await asyncio.to_thread(self.guarded, task, register_copy)
        dense, sparse = [], []
        for offset in range(0, len(chunks), 16):
            await asyncio.to_thread(
                self.guarded, task, lambda tx: r.domain.access(tx, task["requested_by"], v["kb_id"], "review")
            )
            d, sp, _ = await r.models.embed([c["text"] for c in chunks[offset : offset + 16]])
            dense.extend(d)
            sparse.extend(sp)
        await asyncio.to_thread(r.vectors.write, chunks, dense, sparse)
        await asyncio.to_thread(r.relations.write, v, chunks)
        index_manifest = {
            "version_id": v["_id"],
            "dense_ready": True,
            "sparse_ready": True,
            "bm25_ready": True,
            "search_ready": True,
            "collection": s.milvus_collection,
            "generation": s.generation,
            "model_revision": s.bge_model_revision,
            "count": len(chunks),
        }
        graph_manifest = {
            "version_id": v["_id"],
            "graph_ready": True,
            "generation": s.generation,
            "snapshot_key": v["graph_snapshot_key"],
            "reviewed_hash": v["submitted_hash"],
            "fact_count": len(chunks),
        }

        def commit(tx):
            r.domain.publish_commit(tx, task, index_manifest, graph_manifest)
            tx.patch("processing_tasks", task["_id"], {"status": "success"})

        await asyncio.to_thread(self.guarded, task, commit)

    async def cleanup(self, task):
        r, s = self.r, self.r.s
        v = await asyncio.to_thread(r.store.read, lambda tx: tx.get("document_versions", task["version_id"]))
        messages = await asyncio.to_thread(
            r.store.read, lambda tx: tx.find("messages", {"dependency_versions": v["_id"]})
        )
        for message in messages:
            await r.checkpointer.adelete_thread(message["thread_id"])
            await asyncio.to_thread(self.guarded, task, lambda tx: None)
            await asyncio.to_thread(r.domain.redact_message, message["_id"])
        if task["kind"] == "delete":
            for collection in {
                *v.get("index_copies", []),
                v.get("collection", s.milvus_collection),
                s.milvus_collection,
            }:
                await asyncio.to_thread(r.vectors.delete_version, collection, v["_id"])
            await asyncio.to_thread(r.relations.delete_version, v["_id"])
            await asyncio.to_thread(r.objects.remove, s.originals_bucket, v["source_key"])
            for asset in v.get("artifact_keys", []):
                await asyncio.to_thread(r.objects.remove, asset["bucket"], asset["key"])
            await asyncio.to_thread(
                self.guarded, task, lambda tx: tx.delete("chunks", {"version_id": v["_id"]})
            )
        await asyncio.to_thread(self.mark, task, "success")

    async def execute(self, task):
        heartbeat = asyncio.create_task(self.heartbeat(task))
        try:

            async def parse_node(state):
                await self.parse(task)
                return {}

            async def index_node(state):
                await self.index(task)
                return {}

            async def cleanup_node(state):
                await self.cleanup(task)
                return {}

            graph = StateGraph(TaskState)
            graph.add_node("parse", parse_node).add_node("index", index_node).add_node(
                "cleanup", cleanup_node
            )
            graph.add_conditional_edges(
                START,
                lambda state: state["kind"],
                {"parse": "parse", "index": "index", "delete": "cleanup", "invalidate": "cleanup"},
            )
            for node in ("parse", "index", "cleanup"):
                graph.add_edge(node, END)
            await graph.compile(checkpointer=self.r.checkpointer).ainvoke(
                {"task_id": task["_id"], "fencing_token": task["fencing_token"], "kind": task["kind"]},
                {"configurable": {"thread_id": f"{task['_id']}:{task['fencing_token']}"}},
            )
        except Exception as error:
            code = error.code if isinstance(error, ServiceError) else type(error).__name__
            log.warning("task_failed task_id=%s code=%s", task["_id"], code)
            retry = isinstance(error, (httpx.TransportError, httpx.HTTPStatusError)) and task["attempts"] < 3
            try:
                await asyncio.to_thread(
                    self.mark,
                    task,
                    "pending" if retry else "failed",
                    error_code=code,
                    next_run_at=now() + timedelta(seconds=15 * task["attempts"]),
                )
            except ServiceError:
                pass  # Lost lease or already committed terminal state; never overwrite another attempt.
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            # Worker checkpoints contain IDs only and do not need indefinite retention.
            await self.r.checkpointer.adelete_thread(f"{task['_id']}:{task['fencing_token']}")

    async def housekeeping(self):
        def drain(tx):
            stale = tx.find(
                "messages",
                {
                    "status": {"$in": ["pending", "generating"]},
                    "created_at": {"$lt": now() - timedelta(seconds=300)},
                },
            )
            for m in stale:
                tx.patch(
                    "messages", m["_id"], {"status": "interrupted", "answer": ""}, {"status": m["status"]}
                )
                slot = tx.get("quota_leases", m["slot_id"])
                if slot and slot.get("request_id") == m["_id"]:
                    tx.patch("quota_leases", slot["_id"], {"expires_at": now(), "request_id": None})
            return tx.find(
                "orphan_candidates", {"created_at": {"$lt": now() - timedelta(minutes=10)}}, limit=50
            )

        candidates = await asyncio.to_thread(self.r.store.transaction, drain)
        for item in candidates:
            linked = await asyncio.to_thread(
                self.r.store.read,
                lambda tx, obj=item: tx.find(
                    "document_versions",
                    {"$or": [{"source_key": obj["key"]}, {"artifact_keys.key": obj["key"]}]},
                    limit=1,
                ),
            )
            if not linked:
                await asyncio.to_thread(self.r.objects.remove, item["bucket"], item["key"])
            await asyncio.to_thread(
                self.r.store.transaction,
                lambda tx, obj=item: tx.delete("orphan_candidates", {"_id": obj["_id"]}),
            )
        events = await asyncio.to_thread(
            self.r.store.read, lambda tx: tx.find("outbox_events", {"status": "pending"}, limit=100)
        )
        for event in events:
            if event["event_type"] in {"membership_changed", "acl_changed", "withdrawn", "deleted"}:
                if event["event_type"] == "membership_changed":
                    query = {"kb_ids": event["aggregate_id"]}
                else:
                    versions = await asyncio.to_thread(
                        self.r.store.read,
                        lambda tx: tx.find(
                            "document_versions", {"document_id": event["aggregate_id"]}, limit=10000
                        ),
                    )
                    query = {"dependency_versions": {"$in": [v["_id"] for v in versions]}}
                messages = await asyncio.to_thread(
                    self.r.store.read, lambda tx: tx.find("messages", query, limit=10000)
                )
                require(
                    len(messages) < 10000, "REDACTION_PAGE_LIMIT", "撤权清理需分页扩容，事件暂不确认", 503
                )
                for m in messages:

                    def check(tx):
                        for kb_id in m["kb_ids"]:
                            self.r.domain.access(tx, m["owner_id"], kb_id)
                        for version_id in m.get("dependency_versions", []):
                            self.r.domain.version(tx, m["owner_id"], version_id, True)

                    try:
                        await asyncio.to_thread(self.r.store.read, check)
                    except ServiceError:
                        await self.r.checkpointer.adelete_thread(m["thread_id"])
                        await asyncio.to_thread(self.r.domain.redact_message, m["_id"])
            await asyncio.to_thread(
                self.r.store.transaction,
                lambda tx, e=event: tx.patch(
                    "outbox_events",
                    e["_id"],
                    {"status": "processed", "processed_at": now()},
                    {"status": "pending"},
                ),
            )
        # Reconcile deletion repeatedly: a lease fences Mongo commits, not an in-flight remote write.
        deleted = await asyncio.to_thread(
            self.r.store.read,
            lambda tx: tx.find(
                "document_versions",
                {
                    "state": "deleted",
                    "$or": [
                        {"delete_verified_at": {"$exists": False}},
                        {"delete_verified_at": {"$lt": now() - timedelta(minutes=5)}},
                    ],
                },
                limit=50,
            ),
        )
        for v in deleted:
            for collection in {
                *v.get("index_copies", []),
                v.get("collection", self.r.s.milvus_collection),
                self.r.s.milvus_collection,
            }:
                await asyncio.to_thread(self.r.vectors.delete_version, collection, v["_id"])
            await asyncio.to_thread(self.r.relations.delete_version, v["_id"])
            await asyncio.to_thread(
                self.r.store.transaction,
                lambda tx, key=v["_id"]: tx.patch(
                    "document_versions", key, {"delete_verified_at": now()}, {"state": "deleted"}
                ),
            )

    async def run(self):
        while True:
            try:
                await self.housekeeping()
            except Exception as error:
                log.warning("housekeeping_failed exception_type=%s", type(error).__name__)
            task = await asyncio.to_thread(self.claim)
            if task:
                await self.execute(task)
            else:
                await asyncio.sleep(self.r.s.worker_poll_seconds)
