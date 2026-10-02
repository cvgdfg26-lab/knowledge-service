import asyncio
import json
import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from knowledge_service.core.config import get_settings
from knowledge_service.core.errors import ServiceError, require
from knowledge_service.core.security import digest, now, uid
from knowledge_service.rag import answer_graph, search
from knowledge_service.runtime import Runtime
from knowledge_service.schemas import (
    AclInput,
    AnswerInput,
    ConversationInput,
    FeedbackInput,
    FeedbackPatch,
    KnowledgeBaseInput,
    KnowledgeBasePatch,
    Login,
    MembershipInput,
    ReviewInput,
    RollbackInput,
    SearchInput,
)

log = logging.getLogger("knowledge.api")
REQUESTS = Counter("knowledge_http_requests_total", "HTTP requests", ["role", "method", "status"])


def create_app(role, settings=None, runtime=None):
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime or Runtime(settings)
        yield
        if runtime is None:
            await app.state.runtime.close()

    app = FastAPI(title=f"Knowledge {role} API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["Content-Type", "X-CSRF-Token", "Idempotency-Key", "X-Request-ID"],
    )

    @app.middleware("http")
    async def context(request, call_next):
        request.state.request_id = uid("req")
        origin = request.headers.get("origin")
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > settings.max_upload_bytes + 1024 * 1024):
            return JSONResponse({"error": {"code": "UPLOAD_SIZE", "message": "请求体超限"}}, status_code=413)
        if (
            request.method in {"POST", "PATCH", "PUT", "DELETE"}
            and origin
            and origin not in settings.allowed_origins
        ):
            return JSONResponse(
                {
                    "request_id": request.state.request_id,
                    "data": None,
                    "error": {"code": "ORIGIN", "message": "请求来源不允许"},
                },
                status_code=403,
            )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self' "
            + " ".join(settings.allowed_origins)
            + "; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'"
        )
        REQUESTS.labels(role, request.method, response.status_code).inc()
        return response

    @app.exception_handler(ServiceError)
    async def business_error(request, error):
        return JSONResponse(
            {
                "request_id": request.state.request_id,
                "data": None,
                "error": {"code": error.code, "message": error.message},
            },
            status_code=error.status,
        )

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        log.error(
            "request_failed request_id=%s exception_type=%s", request.state.request_id, type(error).__name__
        )
        return JSONResponse(
            {
                "request_id": request.state.request_id,
                "data": None,
                "error": {"code": "DEPENDENCY_ERROR", "message": "服务暂不可用"},
            },
            status_code=503,
        )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        # Do not echo passwords, tokens, or source text from Pydantic's input field.
        return JSONResponse(
            {
                "request_id": request.state.request_id,
                "data": None,
                "error": {"code": "VALIDATION", "message": "请求参数不合法"},
            },
            status_code=422,
        )

    def r(request: Request):
        return request.app.state.runtime

    def user(request: Request, rt=Depends(r)):
        return rt.domain.authenticate(
            request.cookies.get("kb_session"),
            request.headers.get("x-csrf-token"),
            write=request.method in {"POST", "PATCH", "PUT", "DELETE"},
        )

    def key(request: Request):
        return request.headers.get("idempotency-key", "")

    def ok(request, data):
        return {"request_id": request.state.request_id, "data": data, "error": None}

    @app.get("/api/v1/health/live")
    def live(request: Request):
        return ok(request, {"service": "kb-" + role, "status": "live"})

    @app.get("/api/v1/health/ready")
    def ready(request: Request, rt=Depends(r)):
        checks = {}
        for name, factory in (
            ("mongodb", lambda: rt.store),
            ("minio", lambda: rt.objects),
            ("milvus", lambda: rt.vectors),
            ("neo4j", lambda: rt.relations),
        ):
            try:
                factory().ping()
                checks[name] = "ready"
            except Exception:
                checks[name] = "unavailable"
        checks["embedding_config"] = (
            "ready"
            if (settings.bge_m3_endpoint or settings.bge_m3_service_url) and settings.bge_model_revision
            else "missing"
        )
        good = all(v == "ready" for v in checks.values())
        return JSONResponse(
            ok(request, {"service": "kb-" + role, "checks": checks}), status_code=200 if good else 503
        )

    @app.get("/api/v1/auth/me")
    def me(request: Request, current=Depends(user)):
        return ok(request, {"user_id": current["_id"], "login": current["login"]})

    @app.get("/api/v1/knowledge-bases")
    def kbs(request: Request, current=Depends(user), rt=Depends(r)):
        return ok(request, rt.domain.list_kbs(current["_id"]))

    @app.get("/api/v1/knowledge-bases/{kb_id}")
    def kb_detail(kb_id: str, request: Request, current=Depends(user), rt=Depends(r)):
        return ok(request, rt.store.read(lambda tx: rt.domain.access(tx, current["_id"], kb_id)))

    @app.get("/api/v1/metrics", include_in_schema=False)
    def metrics(current=Depends(user), rt=Depends(r)):
        actor = rt.store.read(lambda tx: rt.domain.actor(tx, current["_id"]))
        require(actor.get("system_admin"), "FORBIDDEN", "需要运维权限", 403)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    if role == "import":

        @app.patch("/api/v1/knowledge-bases/{kb_id}")
        def policy(
            kb_id: str, body: KnowledgeBasePatch, request: Request, current=Depends(user), rt=Depends(r)
        ):
            return ok(
                request,
                rt.domain.policy(current["_id"], kb_id, key(request), body.model_dump(exclude_unset=True)),
            )

        @app.patch("/api/v1/documents/{doc_id}/acl")
        def acl(doc_id: str, body: AclInput, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.acl(current["_id"], doc_id, key(request), body.reader_ids))

        @app.post("/api/v1/documents/{doc_id}/rollback")
        async def rollback(
            doc_id: str, body: RollbackInput, request: Request, current=Depends(user), rt=Depends(r)
        ):
            def prepare(tx):
                doc = rt.domain.document(tx, current["_id"], doc_id, "review")
                v = tx.get("document_versions", body.version_id)
                manifest = tx.get("index_manifests", body.version_id)
                require(
                    v
                    and v["document_id"] == doc_id
                    and manifest
                    and manifest["model_revision"] == settings.bge_model_revision
                    and v.get("generation") == settings.generation,
                    "ROLLBACK_PROFILE",
                    "回滚副本不匹配当前模型与索引代次",
                )
                return doc, v, tx.find("chunks", {"version_id": v["_id"]}, 5000)

            doc, v, chunks = await asyncio.to_thread(rt.store.read, prepare)
            require(chunks, "ROLLBACK_NOT_READY", "回滚副本无片段")
            dense, sparse, _ = await rt.models.embed([chunks[0]["text"]])
            await asyncio.to_thread(rt.vectors.verify, v["collection"], chunks, dense[0], sparse[0])
            await asyncio.to_thread(rt.relations.verify, v, chunks)
            result = await asyncio.to_thread(
                rt.domain.rollback, current["_id"], doc_id, key(request), body.version_id, doc["revision"]
            )
            return ok(request, result)

        @app.post("/api/v1/tasks/{task_id}/retry", status_code=202)
        def retry(task_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.retry(current["_id"], task_id, key(request)))

        @app.post("/api/v1/knowledge-bases", status_code=201)
        def create_kb(body: KnowledgeBaseInput, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.create_kb(current["_id"], key(request), body.model_dump()))

        @app.put("/api/v1/knowledge-bases/{kb_id}/members/{target}")
        def member(
            kb_id: str,
            target: str,
            body: MembershipInput,
            request: Request,
            current=Depends(user),
            rt=Depends(r),
        ):
            return ok(request, rt.domain.member(current["_id"], kb_id, target, body.role, key(request)))

        @app.delete("/api/v1/knowledge-bases/{kb_id}/members/{target}")
        def revoke(kb_id: str, target: str, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.member(current["_id"], kb_id, target, None, key(request)))

        @app.get("/api/v1/knowledge-bases/{kb_id}/documents")
        def documents(kb_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            def query(tx):
                rt.domain.access(tx, current["_id"], kb_id)
                member = tx.get("kb_members", kb_id + ":" + current["_id"])
                docs = tx.find("documents", {"kb_id": kb_id, "deleted_at": None})
                if member["role"] == "reader":
                    allowed = {v["document_id"] for v in rt.domain.scope(current["_id"], [kb_id])}
                    docs = [d for d in docs if d["_id"] in allowed]
                return docs

            return ok(request, rt.store.read(query))

        async def upload_source(kb_id, request, current, rt, file, metadata, document_id=None):
            await asyncio.to_thread(
                rt.store.read, lambda tx: rt.domain.access(tx, current["_id"], kb_id, "maintain")
            )
            filename = file.filename or ""
            require(
                filename and "/" not in filename and "\\" not in filename and len(filename) <= 200,
                "FILENAME",
                "文件名不合法",
                422,
            )
            require(
                Path(filename).suffix.lower() in {".md", ".txt", ".pdf", ".docx"},
                "FILE_TYPE",
                "仅支持 Markdown、TXT、PDF、DOCX",
                422,
            )
            require(key(request), "IDEMPOTENCY_REQUIRED", "上传需要 Idempotency-Key", 422)
            data = await file.read(settings.max_upload_bytes + 1)
            require(0 < len(data) <= settings.max_upload_bytes, "UPLOAD_SIZE", "文件为空或超过大小限制", 413)
            try:
                meta = json.loads(metadata)
                require(
                    isinstance(meta, dict)
                    and len(metadata) <= 4000
                    and set(meta) <= {"title", "course", "city", "year", "age_range"}
                    and all(isinstance(v, (str, int)) for v in meta.values()),
                    "METADATA",
                    "资料字段不合法",
                    422,
                )
            except ValueError:
                raise ServiceError("METADATA", "元数据需要 JSON 对象", 422) from None
            source_key = kb_id + "/uploads/" + uid("source") + Path(filename).suffix.lower()
            await asyncio.to_thread(
                rt.objects.put,
                settings.originals_bucket,
                source_key,
                data,
                mimetypes.guess_type(filename)[0] or "application/octet-stream",
            )
            try:
                result = await asyncio.to_thread(
                    rt.domain.upload,
                    current["_id"],
                    kb_id,
                    key(request),
                    filename,
                    source_key,
                    digest(data),
                    meta,
                    document_id,
                )
                linked = await asyncio.to_thread(
                    rt.store.read, lambda tx: tx.get("document_versions", result["version_id"])
                )
                if linked["source_key"] != source_key:  # Replayed request: remove the unreferenced object.
                    await asyncio.to_thread(rt.objects.remove, settings.originals_bucket, source_key)
                return ok(request, result)
            except Exception:
                # Do not remove a file whose transaction might have committed; record for reconciliation.
                await asyncio.to_thread(
                    rt.store.transaction,
                    lambda tx: tx.insert(
                        "orphan_candidates",
                        {
                            "_id": uid("orphan"),
                            "bucket": settings.originals_bucket,
                            "key": source_key,
                            "created_at": now(),
                        },
                    ),
                )
                raise

        @app.post("/api/v1/knowledge-bases/{kb_id}/documents", status_code=202)
        async def upload(
            kb_id: str,
            request: Request,
            file: UploadFile = File(...),
            metadata: str = Form("{}"),
            current=Depends(user),
            rt=Depends(r),
        ):
            return await upload_source(kb_id, request, current, rt, file, metadata)

        @app.post("/api/v1/documents/{doc_id}/versions", status_code=202)
        async def new_version(
            doc_id: str,
            request: Request,
            file: UploadFile = File(...),
            metadata: str = Form("{}"),
            current=Depends(user),
            rt=Depends(r),
        ):
            doc = await asyncio.to_thread(
                rt.store.read, lambda tx: rt.domain.document(tx, current["_id"], doc_id, "maintain")
            )
            return await upload_source(doc["kb_id"], request, current, rt, file, metadata, doc_id)

        @app.get("/api/v1/documents/{doc_id}/versions")
        def versions(doc_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            def query(tx):
                existing = tx.get("documents", doc_id)
                require(existing, "NOT_FOUND", "资源不可访问", 404)
                rt.domain.access(tx, current["_id"], existing["kb_id"])
                membership = tx.get("kb_members", existing["kb_id"] + ":" + current["_id"])
                action = (
                    "read"
                    if membership["role"] == "reader"
                    else ("review" if membership["role"] == "reviewer" else "maintain")
                )
                doc = rt.domain.document(tx, current["_id"], doc_id, action)
                member = tx.get("kb_members", doc["kb_id"] + ":" + current["_id"])
                rows = tx.find("document_versions", {"document_id": doc_id})
                return (
                    rows
                    if member["role"] != "reader"
                    else [v for v in rows if v["state"] in {"published", "archived"}]
                )

            return ok(request, rt.store.read(query))

        @app.get("/api/v1/versions/{version_id}/preview")
        def preview(version_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.preview(current["_id"], version_id))

        @app.post("/api/v1/versions/{version_id}/submit")
        def submit(version_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.submit(current["_id"], version_id, key(request)))

        @app.post("/api/v1/versions/{version_id}/reviews")
        def review(
            version_id: str, body: ReviewInput, request: Request, current=Depends(user), rt=Depends(r)
        ):
            return ok(
                request,
                rt.domain.review(current["_id"], version_id, key(request), body.decision, body.reason),
            )

        @app.post("/api/v1/versions/{version_id}/publish", status_code=202)
        def publish(version_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.publish_request(current["_id"], version_id, key(request)))

        @app.post("/api/v1/documents/{doc_id}/withdraw")
        def withdraw(doc_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.block(current["_id"], doc_id, key(request)))

        @app.delete("/api/v1/documents/{doc_id}", status_code=202)
        def delete(doc_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.block(current["_id"], doc_id, key(request), True))

        @app.get("/api/v1/tasks/{task_id}")
        def task_state(task_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            def query(tx):
                task = tx.get("processing_tasks", task_id)
                require(task, "NOT_FOUND", "资源不可访问", 404)
                v = tx.get("document_versions", task["version_id"])
                rt.domain.access(
                    tx, current["_id"], v["kb_id"], "maintain" if task["kind"] == "parse" else "review"
                )
                return {
                    k: task.get(k) for k in ("_id", "kind", "status", "attempts", "error_code", "version_id")
                }

            return ok(request, rt.store.read(query))

    if role == "query":

        @app.post("/api/v1/auth/login")
        def login(body: Login, request: Request, response: Response, rt=Depends(r)):
            token, csrf = rt.domain.login(
                body.login, body.password, request.client.host if request.client else "unknown"
            )
            response.set_cookie(
                "kb_session",
                token,
                httponly=True,
                secure=settings.cookie_secure,
                samesite="lax",
                path="/",
                max_age=settings.session_ttl_seconds,
            )
            return ok(request, {"csrf_token": csrf})

        @app.post("/api/v1/auth/logout")
        def logout(request: Request, response: Response, current=Depends(user), rt=Depends(r)):
            rt.domain.logout(current)
            response.delete_cookie("kb_session", path="/")
            return ok(request, {"status": "logged_out"})

        @app.post("/api/v1/search")
        async def retrieve(body: SearchInput, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, await search(rt, current["_id"], body))

        @app.post("/api/v1/conversations", status_code=201)
        def conversation(body: ConversationInput, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.create_conversation(current["_id"], key(request), body.kb_ids))

        @app.get("/api/v1/conversations/{chat_id}")
        def history(chat_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.domain.conversation(current["_id"], chat_id))

        @app.post("/api/v1/conversations/{chat_id}/answers")
        async def answer(
            chat_id: str, body: AnswerInput, request: Request, current=Depends(user), rt=Depends(r)
        ):
            message = await asyncio.to_thread(
                rt.domain.reserve_answer, current["_id"], chat_id, key(request), body.question
            )
            require(
                message["status"] == "pending", "ALREADY_STARTED", "本次请求已有记录，请读取会话状态", 409
            )
            # CAS before stream creation prevents duplicate billing if two identical requests race.
            await asyncio.to_thread(
                rt.store.transaction,
                lambda tx: tx.patch(
                    "messages",
                    message["_id"],
                    {"stream_claimed": True},
                    {"status": "pending", "stream_claimed": {"$ne": True}},
                ),
            )

            async def events():
                sequence = 0

                def frame(value):
                    nonlocal sequence
                    sequence += 1
                    event = value.pop("event")
                    payload = {
                        "request_id": message["_id"],
                        "message_id": message["_id"],
                        "sequence": sequence,
                        **value,
                    }
                    return (
                        f"event: {event}\nid: {sequence}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    )

                try:
                    yield frame({"event": "accepted"})
                    async with asyncio.timeout(240):
                        graph = answer_graph(rt)
                        async for value in graph.astream(
                            {"message_id": message["_id"], "user_id": current["_id"]},
                            {"configurable": {"thread_id": message["thread_id"]}},
                            stream_mode="custom",
                        ):
                            await asyncio.to_thread(rt.domain.authenticate, request.cookies.get("kb_session"))
                            await asyncio.to_thread(rt.domain.conversation, current["_id"], chat_id)

                            def check_dependencies(tx):
                                m = tx.get("messages", message["_id"])
                                for version_id in m.get("dependency_versions", []):
                                    rt.domain.version(tx, current["_id"], version_id, True)

                            await asyncio.to_thread(rt.store.read, check_dependencies)
                            if await request.is_disconnected():
                                break
                            yield frame(value)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    code = error.code if isinstance(error, ServiceError) else "GENERATION_INTERRUPTED"
                    yield frame(
                        {
                            "event": "access_revoked"
                            if code in {"NOT_FOUND", "UNAUTHENTICATED"}
                            else "error",
                            "code": code,
                        }
                    )
                finally:
                    await asyncio.to_thread(rt.domain.finish_answer, message["_id"], "interrupted")

            return StreamingResponse(
                events(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
            )

        @app.post("/api/v1/requests/{message_id}/cancel")
        def cancel(message_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            def operation(tx):
                m = tx.get("messages", message_id)
                require(m and m["owner_id"] == current["_id"], "NOT_FOUND", "资源不可访问", 404)
                tx.patch("messages", message_id, {"cancel_requested": True})
                return {"message_id": message_id, "cancel_requested": True}

            return ok(
                request, rt.domain.write(current["_id"], "cancel:" + message_id, key(request), {}, operation)
            )

        @app.get("/api/v1/citations/{chunk_id}")
        def citation(chunk_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            chunks = rt.domain.readable_chunks(current["_id"], [chunk_id], True)
            require(chunks, "NOT_FOUND", "资源不可访问", 404)
            return ok(request, chunks[0])

        @app.get("/api/v1/versions/{version_id}/source")
        def source(version_id: str, current=Depends(user), rt=Depends(r)):
            v = rt.store.read(lambda tx: rt.domain.version(tx, current["_id"], version_id, True))
            data = rt.objects.get(settings.originals_bucket, v["source_key"])
            rt.store.read(lambda tx: rt.domain.version(tx, current["_id"], version_id, True))
            return Response(
                data,
                media_type="application/octet-stream",
                headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(v["filename"])},
            )

        @app.get("/api/v1/assets/{asset_id}")
        def asset(asset_id: str, current=Depends(user), rt=Depends(r)):
            def locate(tx):
                versions = tx.find("document_versions", {"artifact_keys.asset_id": asset_id}, limit=1)
                require(versions, "NOT_FOUND", "资源不可访问", 404)
                v = rt.domain.version(tx, current["_id"], versions[0]["_id"], True)
                return v, next(a for a in v["artifact_keys"] if a.get("asset_id") == asset_id)

            v, a = rt.store.read(locate)
            require(
                a["content_type"] in {"image/png", "image/jpeg", "image/webp"},
                "ASSET_TYPE",
                "资产不可内联预览",
                422,
            )
            data = rt.objects.get(a["bucket"], a["key"])
            rt.store.read(lambda tx: rt.domain.version(tx, current["_id"], v["_id"], True))
            return Response(data, media_type=a["content_type"])

        @app.post("/api/v1/feedback", status_code=201)
        def feedback(body: FeedbackInput, request: Request, current=Depends(user), rt=Depends(r)):
            def operation(tx):
                m = tx.get("messages", body.message_id)
                require(m and m["owner_id"] == current["_id"], "NOT_FOUND", "资源不可访问", 404)
                for version_id in m.get("dependency_versions", []):
                    rt.domain.version(tx, current["_id"], version_id, True)
                return tx.insert(
                    "feedback_tickets",
                    {
                        "_id": uid("feedback"),
                        "owner_id": current["_id"],
                        **body.model_dump(),
                        "status": "open",
                        "created_at": now(),
                    },
                )

            return ok(
                request,
                rt.domain.write(current["_id"], "feedback", key(request), body.model_dump(), operation),
            )

        def feedback_access(tx, current, feedback_id, rt, maintain=False):
            f = tx.get("feedback_tickets", feedback_id)
            require(f, "NOT_FOUND", "资源不可访问", 404)
            m = tx.get("messages", f["message_id"])
            require(m, "NOT_FOUND", "资源不可访问", 404)
            if maintain or f["owner_id"] != current["_id"]:
                for kb_id in m["kb_ids"]:
                    rt_access = tx.get("kb_members", kb_id + ":" + current["_id"])
                    require(
                        rt_access and rt_access["role"] in {"maintainer", "reviewer", "admin"},
                        "NOT_FOUND",
                        "资源不可访问",
                        404,
                    )
            for version_id in m.get("dependency_versions", []):
                rt.domain.version(tx, current["_id"], version_id, True)
            return f

        @app.get("/api/v1/feedback/{feedback_id}")
        def feedback_detail(feedback_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            return ok(request, rt.store.read(lambda tx: feedback_access(tx, current, feedback_id, rt)))

        @app.patch("/api/v1/feedback/{feedback_id}")
        def feedback_update(
            feedback_id: str, body: FeedbackPatch, request: Request, current=Depends(user), rt=Depends(r)
        ):
            def operation(tx):
                feedback_access(tx, current, feedback_id, rt, True)
                require(
                    body.status not in {"resolved", "dismissed"} or body.resolution,
                    "RESOLUTION_REQUIRED",
                    "关闭反馈需填写处理和回归说明",
                    422,
                )
                rt.domain.audit(tx, current["_id"], "feedback_updated", feedback_id)
                return tx.patch("feedback_tickets", feedback_id, body.model_dump())

            return ok(
                request,
                rt.domain.write(
                    current["_id"], "feedback:" + feedback_id, key(request), body.model_dump(), operation
                ),
            )

        @app.get("/api/v1/debug/requests/{message_id}")
        def debug(message_id: str, request: Request, current=Depends(user), rt=Depends(r)):
            def query(tx):
                m = tx.get("messages", message_id)
                require(m and m["owner_id"] == current["_id"], "NOT_FOUND", "资源不可访问", 404)
                for kb_id in m["kb_ids"]:
                    rt.domain.access(tx, current["_id"], kb_id, "manage")
                for version_id in m.get("dependency_versions", []):
                    rt.domain.version(tx, current["_id"], version_id, True)
                return {
                    k: m.get(k)
                    for k in ("_id", "status", "timings", "degraded", "dependency_versions", "sources")
                }

            return ok(request, rt.store.read(query))

        if settings.frontend_path.exists():
            app.mount("/", StaticFiles(directory=settings.frontend_path, html=True), name="frontend")
    return app
