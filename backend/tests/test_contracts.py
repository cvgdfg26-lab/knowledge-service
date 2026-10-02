import io
import json
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient

from knowledge_service.adapters.mineru import MinerU, unpack_artifacts
from knowledge_service.adapters.models import Models
from knowledge_service.api import create_app
from knowledge_service.core.config import Settings
from knowledge_service.core.errors import ServiceError
from knowledge_service.evaluation import evidence_matches
from knowledge_service.ingestion import split_text


@pytest.mark.parametrize("filename", ["../secrets", "/absolute", "C:/secret", "..\\secret"])
def test_zip_traversal_rejected(filename):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(filename, "data")
    with pytest.raises(ServiceError, match="ARCHIVE_PATH"):
        unpack_artifacts(buffer.getvalue(), 10000)


def test_zip_expansion_limit():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("full.md", "x" * 1000)
    with pytest.raises(ServiceError):
        unpack_artifacts(buffer.getvalue(), 100)


def test_pdf_real_page_and_docx_block_locator():
    c = split_text("ignored", "v", "kb", [{"text": "原文", "page_idx": 4, "bbox": [1, 2, 3, 4]}])[0]
    assert c["locator"]["page_number"] == 5
    with pytest.raises(ServiceError):
        split_text("ignored", "v", "kb", [{"text": "原文"}])
    c = split_text("ignored", "v", "kb", [{"text": "原文"}], source_format="docx")[0]
    assert c["locator"]["block_index"] == 0 and "page_number" not in c["locator"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dimension,revision,expected",
    [(1024, "rev", None), (1536, "rev", "BGE_DIMENSION"), (1024, "wrong", "BGE_REVISION")],
)
async def test_gpu_embedding_contract(dimension, revision, expected):
    def handler(request):
        assert request.headers["authorization"] == "Bearer gpu-test-token"
        assert json.loads(request.content)["return_sparse"] is True
        return httpx.Response(
            200,
            json={
                "model_revision": revision,
                "dense_vectors": [[1.0] + [0.0] * (dimension - 1)],
                "sparse_vectors": [{"8": 0.3}],
            },
        )

    s = Settings(
        _env_file=None,
        bge_m3_service_url="https://gpu.test",
        bge_model_revision="rev",
        bge_service_token="gpu-test-token",
    )
    models = Models(s, httpx.MockTransport(handler))
    try:
        if expected:
            with pytest.raises(ServiceError) as error:
                await models.embed(["测试"])
            assert error.value.code == expected
        else:
            d, sparse, _ = await models.embed(["测试"])
            assert len(d[0]) == 1024 and sparse == [{8: 0.3}]
    finally:
        await models.close()


@pytest.mark.asyncio
async def test_rerank_rejects_duplicate_ids():
    models = Models(
        Settings(_env_file=None, reranker_service_url="https://gpu.test"),
        httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "model": "BAAI/bge-reranker-large",
                    "results": [{"candidate_id": "one", "score": 0.5}, {"candidate_id": "one", "score": 0.4}],
                },
            )
        ),
    )
    with pytest.raises(ServiceError) as error:
        await models.rerank("问题", [{"_id": "one", "text": "one"}, {"_id": "two", "text": "two"}])
    assert error.value.code == "RERANK_RESPONSE"
    await models.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url", ["http://example.com/file", "https://127.0.0.1/file", "https://example.com:8080/file"]
)
async def test_mineru_disallows_untrusted_transfers(url):
    parser = MinerU(Settings(_env_file=None))
    with pytest.raises(ServiceError):
        await parser.validate_transfer(url)
    await parser.close()


def test_http_auth_csrf_and_sanitized_validation(runtime):
    with TestClient(create_app("query", runtime.s, runtime)) as client:
        assert client.get("/api/v1/health/live").status_code == 200
        assert client.get("/api/v1/knowledge-bases").status_code == 401
        login = client.post("/api/v1/auth/login", json={"login": "reader", "password": "test-password-12345"})
        csrf = login.json()["data"]["csrf_token"]
        assert client.get("/api/v1/knowledge-bases").status_code == 200
        assert client.post("/api/v1/conversations", json={"kb_ids": ["kb"]}).status_code == 403
        headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "chat"}
        assert (
            client.post("/api/v1/conversations", json={"kb_ids": ["kb"]}, headers=headers).status_code == 201
        )
        invalid = client.post(
            "/api/v1/auth/login", json={"login": "reader", "password": {"secret": "sensitive"}}
        )
        assert invalid.status_code == 422 and "sensitive" not in invalid.text
        assert (
            client.post("/api/v1/auth/login", json={}, headers={"Origin": "https://evil.test"}).status_code
            == 403
        )


def test_http_sse_duplicate_does_not_generate(runtime):
    with TestClient(create_app("query", runtime.s, runtime)) as client:
        csrf = client.post(
            "/api/v1/auth/login", json={"login": "reader", "password": "test-password-12345"}
        ).json()["data"]["csrf_token"]
        headers = {"X-CSRF-Token": csrf, "Idempotency-Key": "chat"}
        chat = client.post("/api/v1/conversations", json={"kb_ids": ["kb"]}, headers=headers).json()["data"][
            "_id"
        ]
        headers["Idempotency-Key"] = "answer"
        url = f"/api/v1/conversations/{chat}/answers"
        response = client.post(url, json={"question": "无资料的问题"}, headers=headers)
        assert response.status_code == 200 and "event: done" in response.text
        assert "evidence_insufficient" in response.text
        assert client.post(url, json={"question": "无资料的问题"}, headers=headers).status_code == 409
        assert runtime.models.generate_calls == 0


def test_evaluation_uses_source_location_not_chunk_ids():
    chunk = {"version_id": "v", "locator": {"line_start": 3, "line_end": 9}}
    assert evidence_matches(chunk, {"version_id": "v", "line_start": 8, "line_end": 10})
    assert not evidence_matches(chunk, {"version_id": "v2", "line_start": 8, "line_end": 10})


def test_feedback_can_be_resolved_and_requires_ownership(published):
    r = published
    chat = r.domain.create_conversation("reader", "chat", ["kb"])
    m = r.domain.reserve_answer("reader", chat["_id"], "answer", "question")
    r.domain.finish_answer(m["_id"], "completed", "answer")
    with TestClient(create_app("query", r.s, r)) as client:

        def login(user):
            return client.post(
                "/api/v1/auth/login", json={"login": user, "password": "test-password-12345"}
            ).json()["data"]["csrf_token"]

        headers = {"X-CSRF-Token": login("reader"), "Idempotency-Key": "feedback"}
        response = client.post(
            "/api/v1/feedback",
            json={"message_id": m["_id"], "category": "answer_error", "detail": "核对失败"},
            headers=headers,
        )
        assert response.status_code == 201
        ticket = response.json()["data"]["_id"]
        headers = {"X-CSRF-Token": login("maintainer"), "Idempotency-Key": "resolve"}
        assert (
            client.patch(
                f"/api/v1/feedback/{ticket}",
                json={"status": "resolved", "resolution": "补充原文，回归通过"},
                headers=headers,
            ).status_code
            == 200
        )
        login("outsider")
        assert client.get(f"/api/v1/feedback/{ticket}").status_code == 404
