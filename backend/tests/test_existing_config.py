import json
from pathlib import Path

import httpx
import pytest
import yaml
from dotenv import dotenv_values

from knowledge_service.adapters.models import Models
from knowledge_service.core.config import Settings, get_settings
from knowledge_service.core.env_import import normalize_env, prepare


def test_legacy_configuration_maps_without_fabricating_revision():
    source = {
        "MONGO_URL": "mongodb://127.0.0.1:27017",
        "MONGO_DB_NAME": "supplied",
        "MINIO_BUCKET_NAME": "existing",
        "NEO4J_USERNAME": "existing-user",
        "BGE_M3_URL": "http://127.0.0.1:6008/embeddings",
        "BGE_RERANKER_URL": "http://127.0.0.1:6009/v1/rerank",
        "TEXT_LLM_OPENAI_API_BASE": "http://127.0.0.1:6006/v1",
        "TEXT_LLM_DEFAULT_MODEL": "qwen3-8b",
        "VLLM_ENABLE": "1",
        "MINERU_MODEL_SOURCE": "modelscope",
    }
    mapped = normalize_env(source)
    assert mapped["MONGODB_DATABASE"] == "supplied"
    assert mapped["GRAPH_CHECKPOINT_DATABASE"] == "supplied_checkpoints"
    assert mapped["ORIGINALS_BUCKET"] == mapped["IMAGES_BUCKET"] == mapped["DERIVED_BUCKET"] == "existing"
    assert mapped["LLM_PROVIDER"] == "vllm" and mapped["LLM_MODEL"] == "qwen3-8b"
    assert mapped["BGE_M3_ENDPOINT"].endswith("/embeddings")
    assert "BGE_MODEL_REVISION" not in mapped and "MINERU_MODEL_VERSION" not in mapped


def test_import_preserves_credentials_no_interpolation_and_no_overwrite(tmp_path):
    source = tmp_path / "source.env"
    dest = tmp_path / "target.env"
    source.write_text(
        "MINERU_BASE_API_TOKEN='test$literal'\nMINIO_SECRET_KEY='test\\'quote'\n"
        "MINERU_TRANSFER_HOSTS='[\"files.example.com\"]'\n"
        "# unrelated shell command remains comment data\n",
        encoding="utf-8",
    )
    report = prepare(source, dest)
    parsed = dotenv_values(dest, interpolate=False)
    assert parsed["MINERU_API_TOKEN"] == "test$literal"
    assert parsed["MINIO_SECRET_KEY"] == "test'quote"
    assert "test$literal" not in json.dumps(report)
    assert "MINERU_TRANSFER_HOSTS" not in report["pending"]
    original = dest.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(source, dest)
    assert dest.read_bytes() == original


def test_explicit_environment_file_takes_priority(monkeypatch, tmp_path):
    path = tmp_path / ".env.existing"
    path.write_text("MONGODB_DATABASE='selected'\nLLM_MODEL='qwen3-8b'\n", encoding="utf-8")
    monkeypatch.setenv("KB_ENV_FILE", str(path))
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.mongodb_database == "selected" and settings.llm_model == "qwen3-8b"
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_existing_model_paths_are_not_appended_twice():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        assert "authorization" not in request.headers
        if request.url.path == "/embeddings":
            return httpx.Response(
                200,
                json={
                    "model_revision": "real-snapshot",
                    "dense_vectors": [[1.0] + [0.0] * 1023],
                    "sparse_vectors": [{"1": 0.5}],
                },
            )
        return httpx.Response(
            200, json={"model": "BAAI/bge-reranker-large", "results": [{"candidate_id": "c", "score": 0.6}]}
        )

    models = Models(
        Settings(
            _env_file=None,
            bge_m3_endpoint="http://gpu.test/embeddings",
            bge_model_revision="real-snapshot",
            reranker_endpoint="http://gpu.test/v1/rerank",
        ),
        httpx.MockTransport(handler),
    )
    try:
        await models.embed(["probe"])
        await models.rerank("probe", [{"_id": "c", "text": "probe"}])
        assert paths == ["/embeddings", "/v1/rerank"]
    finally:
        await models.close()


def test_reuse_profile_only_starts_business_services_on_cpu_host_network():
    root = Path(__file__).resolve().parents[2]
    services = yaml.safe_load((root / "infra/compose.existing.yml").read_text(encoding="utf-8"))["services"]
    assert set(services) == {"kb-import", "kb-query", "kb-worker"}
    for service in services.values():
        assert service["network_mode"] == "host" and "ports" not in service
    assert "LLM_API_KEY" not in services["kb-worker"]["environment"]
    assert "MINERU_API_TOKEN" not in services["kb-query"]["environment"]
    script = (root / "infra/scripts/gpu-tunnel.sh").read_text(encoding="utf-8")
    for port in (6006, 6008, 6009):
        assert f"127.0.0.1:{port}:127.0.0.1:{port}" in script
