import json
import secrets
from datetime import timedelta

from knowledge_service.core.errors import ServiceError, require
from knowledge_service.core.security import digest, now, uid, verify_password

ROLES = {
    "read": {"reader", "maintainer", "reviewer", "admin"},
    "maintain": {"maintainer", "admin"},
    "review": {"reviewer", "admin"},
    "manage": {"admin"},
}


class Domain:
    def __init__(self, store, settings):
        self.store, self.s = store, settings

    def actor(self, tx, user_id):
        user = tx.get("users", user_id)
        require(user and user.get("status") == "active", "UNAUTHENTICATED", "请重新登录", 401)
        return user

    def access(self, tx, user_id, kb_id, action="read"):
        self.actor(tx, user_id)
        kb = tx.get("knowledge_bases", kb_id)
        member = tx.get("kb_members", kb_id + ":" + user_id)
        require(kb and member and member["role"] in ROLES[action], "NOT_FOUND", "资源不可访问", 404)
        return kb

    def document(self, tx, user_id, document_id, action="read"):
        doc = tx.get("documents", document_id)
        require(doc and not doc.get("deleted_at"), "NOT_FOUND", "资源不可访问", 404)
        self.access(tx, user_id, doc["kb_id"], action)
        if action == "read":
            readers = doc.get("reader_ids")
            require(
                not doc.get("blocked_at") and (readers is None or user_id in readers),
                "NOT_FOUND",
                "资源不可访问",
                404,
            )
            require(not doc.get("valid_until") or doc["valid_until"] > now(), "EXPIRED", "资料已失效", 404)
        return doc

    def version(self, tx, user_id, version_id, historical=False):
        version = tx.get("document_versions", version_id)
        require(version, "NOT_FOUND", "资源不可访问", 404)
        doc = self.document(tx, user_id, version["document_id"])
        require(
            version["state"] in ({"published", "archived"} if historical else {"published"}),
            "NOT_FOUND",
            "资源不可访问",
            404,
        )
        if not historical:
            require(doc.get("active_version_id") == version_id, "NOT_FOUND", "资源不可访问", 404)
        return version

    def audit(self, tx, actor, action, resource):
        tx.insert(
            "audit_events",
            {"_id": uid("audit"), "actor": actor, "action": action, "resource": resource, "timestamp": now()},
        )

    def event(self, tx, actor, action, resource):
        self.audit(tx, actor, action, resource)
        tx.insert(
            "outbox_events",
            {
                "_id": uid("event"),
                "event_type": action,
                "aggregate_id": resource,
                "status": "pending",
                "created_at": now(),
            },
        )

    def task(self, tx, version_id, kind, **extra):
        key = version_id + ":" + kind + ":" + self.s.generation
        existing = tx.find("processing_tasks", {"business_key": key}, limit=1)
        if existing:
            return existing[0]
        return tx.insert(
            "processing_tasks",
            {
                "_id": uid("task"),
                "version_id": version_id,
                "kind": kind,
                "business_key": key,
                "status": "pending",
                "attempts": 0,
                "fencing_token": 0,
                "next_run_at": now(),
                **extra,
            },
        )

    def write(self, user_id, route, key, body, callback):
        require(key and len(key) <= 128, "IDEMPOTENCY_REQUIRED", "写操作需要 Idempotency-Key", 422)
        record_id = digest(user_id + ":" + route + ":" + key)
        body_hash = digest(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str))

        def operation(tx):
            self.actor(tx, user_id)
            # Replaying a cached write result must still enforce current resource membership.
            action, _, resource = route.partition(":")
            if action in {"submit", "review", "publish"}:
                v = tx.get("document_versions", resource)
                require(v, "NOT_FOUND", "资源不可访问", 404)
                self.access(tx, user_id, v["kb_id"], "maintain" if action == "submit" else "review")
            elif action in {"acl", "block", "rollback"}:
                doc = tx.get("documents", resource)
                require(doc, "NOT_FOUND", "资源不可访问", 404)
                self.access(tx, user_id, doc["kb_id"], "manage" if action == "acl" else "review")
            previous = tx.get("idempotency_records", record_id)
            if previous:
                require(previous["body_hash"] == body_hash, "IDEMPOTENCY_CONFLICT", "同键请求内容不同", 409)
                return previous["result"]
            result = callback(tx)
            tx.insert(
                "idempotency_records",
                {
                    "_id": record_id,
                    "body_hash": body_hash,
                    "result": result,
                    "actor_id": user_id,
                    "expires_at": now() + timedelta(days=7),
                },
            )
            return result

        return self.store.transaction(operation)

    def login(self, login, password, client="local"):
        window = int(now().timestamp()) // 300
        attempt_id = digest(client + ":" + login + ":" + str(window))

        def count(tx):
            old = tx.get("auth_attempts", attempt_id)
            require(not old or old["count"] < 10, "LOGIN_LIMIT", "登录尝试过多，请稍后重试", 429)
            if old:
                tx.patch("auth_attempts", attempt_id, {"count": old["count"] + 1}, {"count": old["count"]})
            else:
                tx.insert(
                    "auth_attempts",
                    {"_id": attempt_id, "count": 1, "expires_at": now() + timedelta(minutes=10)},
                )

        self.store.transaction(count)
        users = self.store.read(lambda tx: tx.find("users", {"login": login}, limit=1))
        require(
            users and users[0]["status"] == "active" and verify_password(users[0]["password_hash"], password),
            "LOGIN_FAILED",
            "账号或密码错误",
            401,
        )
        user = users[0]
        token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        self.store.transaction(
            lambda tx: tx.insert(
                "auth_sessions",
                {
                    "_id": digest(token),
                    "user_id": user["_id"],
                    "auth_epoch": user["auth_epoch"],
                    "csrf_hash": digest(csrf),
                    "expires_at": now() + timedelta(seconds=self.s.session_ttl_seconds),
                },
            )
        )
        return token, csrf

    def authenticate(self, token, csrf=None, write=False):
        def check(tx):
            session = tx.get("auth_sessions", digest(token or ""))
            require(
                session and not session.get("revoked_at") and session["expires_at"] > now(),
                "UNAUTHENTICATED",
                "请重新登录",
                401,
            )
            user = self.actor(tx, session["user_id"])
            require(user["auth_epoch"] == session["auth_epoch"], "UNAUTHENTICATED", "会话已失效", 401)
            if write:
                require(
                    csrf and secrets.compare_digest(digest(csrf), session["csrf_hash"]),
                    "CSRF",
                    "请求校验失败",
                    403,
                )
            return {"_id": user["_id"], "login": user["login"], "session_id": session["_id"]}

        return self.store.read(check)

    def logout(self, user):
        self.store.transaction(
            lambda tx: tx.patch("auth_sessions", user["session_id"], {"revoked_at": now()})
        )

    def list_kbs(self, user_id):
        def query(tx):
            self.actor(tx, user_id)
            members = tx.find("kb_members", {"user_id": user_id})
            return [{**tx.get("knowledge_bases", m["kb_id"]), "role": m["role"]} for m in members]

        return self.store.read(query)

    def create_kb(self, user_id, key, data):
        def create(tx):
            require(self.actor(tx, user_id).get("system_admin"), "FORBIDDEN", "需要平台管理权限", 403)
            kb = tx.insert(
                "knowledge_bases",
                {"_id": uid("kb"), **data, "policy_epoch": 1, "embedding_profile": self.s.embedding_profile},
            )
            tx.insert(
                "kb_members",
                {"_id": kb["_id"] + ":" + user_id, "kb_id": kb["_id"], "user_id": user_id, "role": "admin"},
            )
            self.event(tx, user_id, "kb_created", kb["_id"])
            return kb

        return self.write(user_id, "create-kb", key, data, create)

    def member(self, user_id, kb_id, target, role, key):
        # Authorization is checked even when replaying an old idempotency result.
        self.store.read(lambda tx: self.access(tx, user_id, kb_id, "manage"))

        def change(tx):
            kb = self.access(tx, user_id, kb_id, "manage")
            self.actor(tx, target)
            member_id = kb_id + ":" + target
            old = tx.get("kb_members", member_id)
            if role:
                if old:
                    tx.patch("kb_members", member_id, {"role": role})
                else:
                    tx.insert(
                        "kb_members", {"_id": member_id, "kb_id": kb_id, "user_id": target, "role": role}
                    )
            else:
                tx.delete("kb_members", {"_id": member_id})
            tx.patch(
                "knowledge_bases",
                kb_id,
                {"policy_epoch": kb["policy_epoch"] + 1},
                {"policy_epoch": kb["policy_epoch"]},
            )
            tx.insert(
                "security_ledger",
                {
                    "_id": uid("security"),
                    "event_type": "membership_changed",
                    "kb_id": kb_id,
                    "user_id": target,
                    "timestamp": now(),
                },
            )
            self.event(tx, user_id, "membership_changed", kb_id)
            return {"kb_id": kb_id, "user_id": target, "role": role}

        return self.write(user_id, "member:" + kb_id + ":" + target, key, {"role": role}, change)

    def upload(self, user_id, kb_id, key, filename, source_key, source_hash, metadata, document_id=None):
        self.store.read(lambda tx: self.access(tx, user_id, kb_id, "maintain"))

        def create(tx):
            self.access(tx, user_id, kb_id, "maintain")
            if document_id:
                doc = self.document(tx, user_id, document_id, "maintain")
                require(doc["kb_id"] == kb_id, "NOT_FOUND", "资源不可访问", 404)
                revision = len(tx.find("document_versions", {"document_id": document_id})) + 1
                tx.patch(
                    "documents", document_id, {"revision": doc["revision"] + 1}, {"revision": doc["revision"]}
                )
            else:
                doc = tx.insert(
                    "documents",
                    {
                        "_id": uid("doc"),
                        "kb_id": kb_id,
                        "owner_id": user_id,
                        "title": metadata.get("title") or filename,
                        "revision": 1,
                        "active_version_id": None,
                        "blocked_at": None,
                        "deleted_at": None,
                    },
                )
                revision = 1
            version = tx.insert(
                "document_versions",
                {
                    "_id": uid("ver"),
                    "document_id": doc["_id"],
                    "kb_id": kb_id,
                    "revision": revision,
                    "source_key": source_key,
                    "source_hash": source_hash,
                    "filename": filename,
                    "metadata": metadata,
                    "state": "draft",
                    "processing_status": "pending",
                    "created_by": user_id,
                    "created_at": now(),
                },
            )
            task = self.task(tx, version["_id"], "parse", requested_by=user_id)
            self.event(tx, user_id, "uploaded", version["_id"])
            return {"document_id": doc["_id"], "version_id": version["_id"], "task_id": task["_id"]}

        body = {"filename": filename, "hash": source_hash, "metadata": metadata, "document_id": document_id}
        return self.write(user_id, "upload:" + kb_id, key, body, create)

    def preview(self, user_id, version_id):
        def query(tx):
            version = tx.get("document_versions", version_id)
            require(version, "NOT_FOUND", "资源不可访问", 404)
            self.access(tx, user_id, version["kb_id"])
            member = tx.get("kb_members", version["kb_id"] + ":" + user_id)
            action = (
                "read"
                if version["state"] == "published"
                else ("review" if member["role"] == "reviewer" else "maintain")
            )
            self.document(tx, user_id, version["document_id"], action)
            return {"version": version, "chunks": tx.find("chunks", {"version_id": version_id})}

        return self.store.read(query)

    def submit(self, user_id, version_id, key):
        def change(tx):
            v = tx.get("document_versions", version_id)
            require(v, "NOT_FOUND", "资源不可访问", 404)
            self.document(tx, user_id, v["document_id"], "maintain")
            require(
                v["processing_status"] == "success" and v["state"] in {"draft", "rejected"},
                "NOT_READY",
                "解析完成后才能提交审核",
            )
            tx.patch(
                "document_versions",
                version_id,
                {"state": "pending_review", "submitted_by": user_id, "submitted_hash": v["preview_hash"]},
                {"state": v["state"]},
            )
            self.event(tx, user_id, "submitted", version_id)
            return {"version_id": version_id, "state": "pending_review"}

        return self.write(user_id, "submit:" + version_id, key, {}, change)

    def review(self, user_id, version_id, key, decision, reason):
        def change(tx):
            v = tx.get("document_versions", version_id)
            require(v, "NOT_FOUND", "资源不可访问", 404)
            self.document(tx, user_id, v["document_id"], "review")
            require(v["state"] == "pending_review", "REVIEW_STATE", "版本不在待审核状态")
            require(v["submitted_by"] != user_id, "SELF_REVIEW", "提交者不能审核自己的内容", 403)
            require(v["submitted_hash"] == v["preview_hash"], "REVIEW_HASH", "审核摘要已变化")
            tx.insert(
                "review_records",
                {
                    "_id": uid("review"),
                    "version_id": version_id,
                    "reviewer_id": user_id,
                    "decision": decision,
                    "reason": reason,
                    "reviewed_hash": v["submitted_hash"],
                },
            )
            tx.patch("document_versions", version_id, {"state": decision}, {"state": "pending_review"})
            self.event(tx, user_id, "reviewed", version_id)
            return {"version_id": version_id, "state": decision}

        return self.write(
            user_id, "review:" + version_id, key, {"decision": decision, "reason": reason}, change
        )

    def publish_request(self, user_id, version_id, key):
        def change(tx):
            v = tx.get("document_versions", version_id)
            require(v, "NOT_FOUND", "资源不可访问", 404)
            doc = self.document(tx, user_id, v["document_id"], "review")
            require(v["state"] == "approved", "PUBLISH_STATE", "版本尚未审核通过")
            task = self.task(
                tx,
                version_id,
                "index",
                expected_revision=doc["revision"],
                expected_active_version=doc["active_version_id"],
                requested_by=user_id,
            )
            return {"version_id": version_id, "task_id": task["_id"]}

        return self.write(user_id, "publish:" + version_id, key, {}, change)

    def publish_commit(self, tx, task, index_manifest, graph_manifest):
        self.assert_lease(tx, task)
        v = tx.get("document_versions", task["version_id"])
        doc = self.document(tx, task["requested_by"], v["document_id"], "review")
        require(
            v["state"] == "approved" and v["submitted_hash"] == v["preview_hash"],
            "PUBLISH_STATE",
            "发布条件已变化",
        )
        require(
            all(index_manifest.get(k) for k in ("dense_ready", "sparse_ready", "bm25_ready", "search_ready"))
            and graph_manifest.get("graph_ready"),
            "NOT_READY",
            "检索/基础图谱未就绪",
        )
        require(
            doc["revision"] == task["expected_revision"]
            and doc["active_version_id"] == task["expected_active_version"]
            and not doc.get("blocked_at"),
            "PUBLISH_CONFLICT",
            "文档版本发生变化，不能发布",
        )
        old = doc["active_version_id"]
        if old:
            tx.patch("document_versions", old, {"state": "archived"})
        tx.patch(
            "documents",
            doc["_id"],
            {"active_version_id": v["_id"], "revision": doc["revision"] + 1},
            {"revision": doc["revision"], "active_version_id": old},
        )
        tx.patch(
            "document_versions",
            v["_id"],
            {"state": "published", "collection": self.s.milvus_collection, "generation": self.s.generation},
            {"state": "approved"},
        )
        for name, value in (("index_manifests", index_manifest), ("graph_manifests", graph_manifest)):
            existing = tx.get(name, v["_id"])
            if existing:
                tx.patch(name, v["_id"], value)
            else:
                tx.insert(name, {"_id": v["_id"], **value})
        self.event(tx, task["requested_by"], "published", v["_id"])

    def block(self, user_id, document_id, key, delete=False):
        def change(tx):
            doc = self.document(tx, user_id, document_id, "review")
            tx.patch(
                "documents",
                document_id,
                {
                    "blocked_at": now(),
                    "deleted_at": now() if delete else None,
                    "revision": doc["revision"] + 1,
                },
                {"revision": doc["revision"]},
            )
            tasks = []
            for v in tx.find("document_versions", {"document_id": document_id}):
                tx.patch("document_versions", v["_id"], {"state": "deleted" if delete else "withdrawn"})
                tasks.append(self.task(tx, v["_id"], "delete" if delete else "invalidate")["_id"])
            tx.insert(
                "security_ledger",
                {
                    "_id": uid("security"),
                    "document_id": document_id,
                    "event_type": "deleted" if delete else "withdrawn",
                    "timestamp": now(),
                },
            )
            self.event(tx, user_id, "deleted" if delete else "withdrawn", document_id)
            return {
                "document_id": document_id,
                "state": "deleted" if delete else "withdrawn",
                "task_ids": tasks,
            }

        return self.write(user_id, "block:" + document_id, key, {"delete": delete}, change)

    def scope(self, user_id, kb_ids):
        def query(tx):
            for kb in kb_ids:
                self.access(tx, user_id, kb)
            docs = tx.find(
                "documents", {"kb_id": {"$in": kb_ids}, "deleted_at": None, "blocked_at": None}, limit=10000
            )
            versions = []
            for doc in docs:
                try:
                    if doc["active_version_id"]:
                        versions.append(self.version(tx, user_id, doc["active_version_id"]))
                except ServiceError as error:
                    if error.status != 404:
                        raise
            return versions

        return self.store.read(query)

    def readable_chunks(self, user_id, ids, historical=False):
        def query(tx):
            result = []
            for chunk_id in ids:
                c = tx.get("chunks", chunk_id)
                if not c:
                    continue
                try:
                    self.version(tx, user_id, c["version_id"], historical)
                    result.append(c)
                except ServiceError as error:
                    if error.status != 404:
                        raise
            return result

        return self.store.read(query)

    def assert_lease(self, tx, task):
        current = tx.get("processing_tasks", task["_id"])
        require(
            current
            and current["status"] == "running"
            and current["fencing_token"] == task["fencing_token"]
            and current["owner"] == task["owner"]
            and current["lease_until"] > now(),
            "LEASE_LOST",
            "任务租约已失效",
        )

    def policy(self, user_id, kb_id, key, values):
        self.store.read(lambda tx: self.access(tx, user_id, kb_id, "manage"))

        def change(tx):
            kb = self.access(tx, user_id, kb_id, "manage")
            require(
                values and all(v is not None for v in values.values()), "POLICY_INPUT", "策略值不能为空", 422
            )
            result = tx.patch(
                "knowledge_bases",
                kb_id,
                {**values, "policy_epoch": kb["policy_epoch"] + 1},
                {"policy_epoch": kb["policy_epoch"]},
            )
            self.event(tx, user_id, "policy_changed", kb_id)
            return result

        return self.write(user_id, "policy:" + kb_id, key, values, change)

    def acl(self, user_id, doc_id, key, readers):
        def change(tx):
            doc = self.document(tx, user_id, doc_id, "manage")
            if readers is not None:
                for target in readers:
                    self.access(tx, target, doc["kb_id"])
            tx.patch(
                "documents",
                doc_id,
                {"reader_ids": readers, "revision": doc["revision"] + 1},
                {"revision": doc["revision"]},
            )
            tx.insert(
                "security_ledger",
                {
                    "_id": uid("security"),
                    "document_id": doc_id,
                    "event_type": "acl_changed",
                    "timestamp": now(),
                },
            )
            self.event(tx, user_id, "acl_changed", doc_id)
            return {"document_id": doc_id, "reader_ids": readers}

        return self.write(user_id, "acl:" + doc_id, key, {"reader_ids": readers}, change)

    def rollback(self, user_id, doc_id, key, version_id, expected_revision):
        def change(tx):
            doc = self.document(tx, user_id, doc_id, "review")
            require(doc["revision"] == expected_revision, "ROLLBACK_CONFLICT", "探测期间文档发生变化")
            require(not doc.get("blocked_at"), "ROLLBACK_BLOCKED", "下架资料不可通过回滚重新公开")
            v = tx.get("document_versions", version_id)
            require(
                v and v["document_id"] == doc_id and v["state"] in {"archived", "published"},
                "ROLLBACK_VERSION",
                "回滚版本不可用",
            )
            index = tx.get("index_manifests", version_id)
            graph = tx.get("graph_manifests", version_id)
            require(
                index
                and all(index.get(k) for k in ("dense_ready", "sparse_ready", "bm25_ready", "search_ready"))
                and graph
                and graph.get("graph_ready"),
                "ROLLBACK_NOT_READY",
                "旧副本未就绪",
            )
            old = doc["active_version_id"]
            if old and old != version_id:
                tx.patch("document_versions", old, {"state": "archived"})
            tx.patch("document_versions", version_id, {"state": "published"})
            tx.patch(
                "documents",
                doc_id,
                {"active_version_id": version_id, "revision": doc["revision"] + 1},
                {"revision": doc["revision"], "active_version_id": old},
            )
            self.event(tx, user_id, "rolled_back", version_id)
            return {"document_id": doc_id, "active_version_id": version_id}

        return self.write(user_id, "rollback:" + doc_id, key, {"version_id": version_id}, change)

    def retry(self, user_id, task_id, key):
        def change(tx):
            task = tx.get("processing_tasks", task_id)
            require(task, "NOT_FOUND", "资源不可访问", 404)
            v = tx.get("document_versions", task["version_id"])
            self.access(tx, user_id, v["kb_id"], "maintain" if task["kind"] == "parse" else "review")
            require(task["status"] == "failed", "TASK_STATE", "仅失败任务可重试")
            submission = tx.get("parser_submissions", v["_id"])
            require(
                not submission or submission["status"] not in {"submitting", "submission_unknown"},
                "SUBMISSION_UNKNOWN",
                "请先核对未知远程任务，不可重复提交",
            )
            if task["kind"] == "index":
                doc = tx.get("documents", v["document_id"])
                require(v["state"] == "approved" and not doc["blocked_at"], "PUBLISH_STATE", "发布条件不满足")
                extra = {
                    "expected_revision": doc["revision"],
                    "expected_active_version": doc["active_version_id"],
                    "requested_by": user_id,
                }
            else:
                extra = {}
            return tx.patch(
                "processing_tasks",
                task_id,
                {"status": "pending", "next_run_at": now(), "attempts": 0, "error_code": None, **extra},
                {"status": "failed"},
            )

        return self.write(user_id, "retry:" + task_id, key, {}, change)

    def create_conversation(self, user_id, key, kb_ids):
        self.scope(user_id, kb_ids)

        def create(tx):
            for kb in kb_ids:
                self.access(tx, user_id, kb)
            return tx.insert(
                "chat_sessions",
                {
                    "_id": uid("chat"),
                    "owner_id": user_id,
                    "kb_ids": kb_ids,
                    "created_at": now(),
                    "revision": 1,
                },
            )

        return self.write(user_id, "conversation", key, {"kb_ids": kb_ids}, create)

    def conversation(self, user_id, conversation_id):
        def query(tx):
            chat = tx.get("chat_sessions", conversation_id)
            require(chat and chat["owner_id"] == user_id, "NOT_FOUND", "资源不可访问", 404)
            for kb in chat["kb_ids"]:
                self.access(tx, user_id, kb)
            messages = tx.find("messages", {"session_id": conversation_id})
            safe = []
            for m in messages:
                try:
                    for version in m.get("dependency_versions", []):
                        self.version(tx, user_id, version, historical=True)
                    safe.append(m)
                except ServiceError:
                    safe.append({"_id": m["_id"], "status": "access_revoked"})
            return {"session": chat, "messages": safe}

        return self.store.read(query)

    def reserve_answer(self, user_id, chat_id, key, question):
        chat = self.conversation(user_id, chat_id)["session"]

        def reserve(tx):
            active_chat = tx.get("chat_sessions", chat_id)
            tx.patch(
                "chat_sessions",
                chat_id,
                {"revision": active_chat["revision"] + 1},
                {"revision": active_chat["revision"]},
            )
            active = tx.find(
                "messages", {"session_id": chat_id, "status": {"$in": ["pending", "generating"]}}
            )
            require(not active, "CHAT_BUSY", "该会话仍有生成任务", 409)
            slots = tx.find("quota_leases", {"expires_at": {"$lte": now()}}, limit=1)
            require(slots, "QUOTA", "生成并发已满，请稍后重试", 429)
            message_id = uid("msg")
            slot = slots[0]
            tx.patch(
                "quota_leases",
                slot["_id"],
                {"request_id": message_id, "expires_at": now() + timedelta(seconds=300)},
                {"expires_at": slot["expires_at"]},
            )
            message = tx.insert(
                "messages",
                {
                    "_id": message_id,
                    "request_id": message_id,
                    "owner_id": user_id,
                    "session_id": chat_id,
                    "question": question,
                    "kb_ids": chat["kb_ids"],
                    "status": "pending",
                    "answer": "",
                    "dependency_versions": [],
                    "created_at": now(),
                    "slot_id": slot["_id"],
                    "cancel_requested": False,
                    "thread_id": user_id + ":" + chat_id + ":" + message_id,
                },
            )
            return message

        return self.write(user_id, "answer:" + chat_id, key, {"question": question}, reserve)

    def finish_answer(self, message_id, status, answer="", sources=None):
        def finish(tx):
            m = tx.get("messages", message_id)
            if m["status"] not in {"pending", "generating"}:
                return
            tx.patch(
                "messages",
                message_id,
                {"status": status, "answer": answer, "sources": sources or []},
                {"status": m["status"]},
            )
            slot = tx.get("quota_leases", m["slot_id"])
            if slot and slot.get("request_id") == message_id:
                tx.patch("quota_leases", slot["_id"], {"expires_at": now(), "request_id": None})

        self.store.transaction(finish)

    def redact_message(self, message_id):
        def redact(tx):
            m = tx.get("messages", message_id)
            tx.patch(
                "messages",
                message_id,
                {"status": "access_revoked", "question": "", "answer": "", "sources": []},
            )
            slot = tx.get("quota_leases", m["slot_id"])
            if slot and slot.get("request_id") == message_id:
                tx.patch("quota_leases", slot["_id"], {"expires_at": now(), "request_id": None})
            tickets = tx.find("feedback_tickets", {"message_id": message_id})
            for f in tickets:
                tx.patch(
                    "feedback_tickets", f["_id"], {"detail": "", "resolution": "", "status": "access_revoked"}
                )
            ids = [message_id, *[f["_id"] for f in tickets]]
            tx.delete(
                "idempotency_records",
                {"$or": [{"result._id": {"$in": ids}}, {"result.message_id": message_id}]},
            )

        self.store.transaction(redact)
