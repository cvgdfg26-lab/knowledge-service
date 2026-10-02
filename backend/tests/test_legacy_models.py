import json

import httpx
import pytest

from knowledge_service.adapters.models import Models
from knowledge_service.core.config import Settings
from knowledge_service.core.errors import ServiceError


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", ["snapshot", None, "other"])
async def test_legacy_embedding_requires_verified_snapshot_header(revision):
    def handler(request):
        assert json.loads(request.content) == {"embedding_documents": ["probe"]}
        headers = {"X-Model-Name": "BAAI/bge-m3"}
        if revision:
            headers["X-Model-Revision"] = revision
        return httpx.Response(
            200, headers=headers, json={"dense": [[1.0] + [0.0] * 1023], "sparse": [{"2": 0.5}]}
        )

    models = Models(
        Settings(
            _env_file=None,
            bge_m3_endpoint="http://gpu/embeddings",
            bge_api_format="legacy_bge",
            bge_model_revision="snapshot",
        ),
        httpx.MockTransport(handler),
    )
    try:
        if revision == "snapshot":
            dense, sparse, _ = await models.embed(["probe"])
            assert len(dense[0]) == 1024 and sparse == [{2: 0.5}]
        else:
            with pytest.raises(ServiceError, match="BGE_REVISION"):
                await models.embed(["probe"])
    finally:
        await models.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scores", [[0.2, 0.8], [0.2], [float("inf"), 0.8]])
async def test_legacy_rerank_preserves_input_ids_and_validates_scores(scores):
    def handler(request):
        assert json.loads(request.content) == {"query": "probe", "documents": ["a", "b"], "normalize": True}
        # Use literal response body to simulate a broken service returning Infinity.
        return httpx.Response(
            200,
            headers={"X-Model-Name": "BAAI/bge-reranker-large", "X-Model-Revision": "snapshot"},
            content=json.dumps({"scores": scores}).encode(),
        )

    models = Models(
        Settings(
            _env_file=None,
            reranker_endpoint="http://gpu/rerank",
            reranker_api_format="legacy_bge",
            reranker_model_revision="snapshot",
        ),
        httpx.MockTransport(handler),
    )
    try:
        candidates = [{"_id": "first", "text": "a"}, {"_id": "second", "text": "b"}]
        if scores == [0.2, 0.8]:
            assert [c["_id"] for c in await models.rerank("probe", candidates)] == ["second", "first"]
        else:
            with pytest.raises(ServiceError, match="RERANK_RESPONSE"):
                await models.rerank("probe", candidates)
    finally:
        await models.close()
