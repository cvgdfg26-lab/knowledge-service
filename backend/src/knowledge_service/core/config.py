import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.cpu", extra="ignore", case_sensitive=False)

    app_env: Literal["dev", "test", "staging", "prod"] = "dev"
    service_role: str = "query"
    mongodb_uri: SecretStr = SecretStr("mongodb://localhost:27017/?replicaSet=rs0")
    mongodb_database: str = "knowledge_dev"
    mongodb_replica_set: str = "rs0"
    graph_checkpoint_database: str = "knowledge_checkpoints_dev"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: SecretStr = SecretStr("")
    minio_secret_key: SecretStr = SecretStr("")
    minio_use_tls: bool = False
    originals_bucket: str = "kb-originals"
    derived_bucket: str = "kb-derived"
    images_bucket: str = "kb-images"
    milvus_uri: str = "http://localhost:19530"
    milvus_token: SecretStr = SecretStr("")
    milvus_collection: str = "kb_chunks_g001_bge1024"
    generation: str = "g001"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    neo4j_database: str = "neo4j"
    embedding_profile: Literal["bge1024_bge_sparse"] = "bge1024_bge_sparse"
    bge_model: Literal["BAAI/bge-m3"] = "BAAI/bge-m3"
    bge_dense_dimension: Literal[1024] = 1024
    bge_model_revision: str = ""
    bge_m3_service_url: str = ""
    bge_m3_endpoint: str = ""
    bge_api_format: Literal["native", "legacy_bge"] = "native"
    bge_service_token: SecretStr = SecretStr("")
    reranker_service_url: str = ""
    reranker_endpoint: str = ""
    reranker_api_format: Literal["native", "legacy_bge"] = "native"
    reranker_service_token: SecretStr = SecretStr("")
    reranker_model: Literal["BAAI/bge-reranker-large"] = "BAAI/bge-reranker-large"
    reranker_model_revision: str = ""
    llm_provider: str = "dashscope"
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "qwen-flash"
    llm_disable_thinking: bool = False
    mineru_api_base_url: str = "https://mineru.net/api/v4"
    mineru_api_token: SecretStr = SecretStr("")
    mineru_model_version: Literal["pipeline", "vlm"] = "pipeline"
    mineru_transfer_hosts: list[str] = Field(default_factory=list)
    mineru_poll_interval_seconds: int = Field(default=15, ge=5)
    mineru_task_timeout_seconds: int = Field(default=1800, ge=60)
    model_timeout_seconds: int = Field(default=45, ge=1, le=300)
    task_lease_seconds: int = Field(default=120, ge=30)
    worker_poll_seconds: float = Field(default=2, ge=0.1)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_archive_bytes: int = Field(default=100 * 1024 * 1024, ge=1)
    max_active_generations: int = Field(default=20, ge=1)
    session_ttl_seconds: int = Field(default=28800, ge=60)
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8000", "http://localhost:8001", "http://localhost:8080"]
    )
    trusted_hosts: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "testserver"])
    cookie_secure: bool = False
    frontend_path: Path = Path(__file__).resolve().parents[4] / "frontend"
    graph_revision: str = "rag-v1"

    @model_validator(mode="after")
    def production_cookie(self):
        if self.app_env in {"staging", "prod"} and not self.cookie_secure:
            raise ValueError("COOKIE_SECURE must be true in staging/prod (HTTPS required)")
        if self.mongodb_database == self.graph_checkpoint_database:
            raise ValueError("checkpoint database must be isolated from business database")
        from urllib.parse import urlsplit

        for origin in self.allowed_origins:
            value = urlsplit(origin)
            if (
                value.scheme not in {"http", "https"}
                or not value.hostname
                or value.username
                or value.password
                or value.path not in {"", "/"}
                or value.query
                or value.fragment
                or ";" in origin
                or " " in origin
            ):
                raise ValueError("ALLOWED_ORIGINS must contain exact HTTP(S) origins")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings(_env_file=os.environ.get("KB_ENV_FILE", ".env.cpu"))
