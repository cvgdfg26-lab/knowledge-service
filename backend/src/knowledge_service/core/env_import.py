"""Translate supplied configuration; never execute comment text or invent model metadata."""

import json
from pathlib import Path

from dotenv import dotenv_values

from knowledge_service.core.config import Settings

ALIASES = {
    "NEO4J_USERNAME": "NEO4J_USER",
    "MONGO_URL": "MONGODB_URI",
    "MONGO_DB_NAME": "MONGODB_DATABASE",
    "MINERU_BASE_API_TOKEN": "MINERU_API_TOKEN",
    "MINERU_BASE_URL": "MINERU_API_BASE_URL",
    "TEXT_LLM_OPENAI_API_KEY": "LLM_API_KEY",
    "TEXT_LLM_OPENAI_API_BASE": "LLM_BASE_URL",
    "TEXT_LLM_DEFAULT_MODEL": "LLM_MODEL",
    "BGE_M3_URL": "BGE_M3_ENDPOINT",
    "BGE_RERANKER_URL": "RERANKER_ENDPOINT",
}
IGNORED = {
    "MINERU_MODEL_SOURCE",
    "WEB_MCP_ENABLE",
    "MCP_DASHSCOPE_BASE_URL",
    "MCP_DASHSCOPE_BASE_URL_STREAMABLE",
}


def normalize_env(source):
    canonical = {k.upper() for k in Settings.model_fields}
    result = {k.upper(): str(v) for k, v in source.items() if v is not None and k.upper() in canonical}
    for old, new in ALIASES.items():
        if new not in result and source.get(old) is not None:
            result[new] = str(source[old])
    if source.get("MINIO_BUCKET_NAME"):
        for name in ("ORIGINALS_BUCKET", "DERIVED_BUCKET", "IMAGES_BUCKET"):
            result.setdefault(name, str(source["MINIO_BUCKET_NAME"]))
    if source.get("VLLM_ENABLE") in {"1", "true", "True"}:
        result.setdefault("LLM_PROVIDER", "vllm")
    result.setdefault("APP_ENV", "dev")
    result.setdefault("COOKIE_SECURE", "false")
    if result.get("MONGODB_DATABASE"):
        result.setdefault("GRAPH_CHECKPOINT_DATABASE", result["MONGODB_DATABASE"] + "_checkpoints")
    return result


def prepare(source_path, destination):
    source_path, destination = Path(source_path), Path(destination)
    if not source_path.is_file():
        raise ValueError("SOURCE_MISSING")
    source = dotenv_values(source_path, encoding="utf-8-sig", interpolate=False)
    values = normalize_env(source)
    # Construct for validation, but never print Pydantic's potentially sensitive input errors.
    inputs = {k.lower(): v for k, v in values.items()}
    for key in ("allowed_origins", "trusted_hosts", "mineru_transfer_hosts"):
        if key in inputs:
            inputs[key] = json.loads(inputs[key])
    settings = Settings(_env_file=None, **inputs)
    if any("\n" in v or "\r" in v for v in values.values()):
        raise ValueError("MULTILINE_ENV_UNSUPPORTED")
    with destination.open("x", encoding="utf-8") as output:
        output.write("# Generated from supplied configuration; secrets must stay local.\n")
        for key, value in sorted(values.items()):
            escaped = value.replace("'", "\\'")
            output.write(key + "='" + escaped + "'\n")
    report = {
        "status": "prepared",
        "fields": sorted(values),
        "pending": [
            k
            for k in ("BGE_MODEL_REVISION", "RERANKER_MODEL_REVISION", "MILVUS_URI", "NEO4J_URI")
            if not values.get(k)
        ]
        + ([] if settings.mineru_transfer_hosts else ["MINERU_TRANSFER_HOSTS"]),
        "ignored_keys": sorted(k for k in source if k in IGNORED),
    }
    return report
