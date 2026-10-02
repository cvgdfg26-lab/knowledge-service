"""Generate local-only deployment secrets without displaying their values or overwriting files."""

import base64
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / ".env.cpu"
SECRET_NAMES = {
    "MONGO_ROOT_PASSWORD",
    "MONGO_IMPORT_PASSWORD",
    "MONGO_QUERY_PASSWORD",
    "MONGO_WORKER_PASSWORD",
    "MINIO_ROOT_PASSWORD",
    "MINIO_APP_PASSWORD",
    "MINIO_QUERY_PASSWORD",
    "MINIO_MILVUS_PASSWORD",
    "NEO4J_PASSWORD",
    "BGE_SERVICE_TOKEN",
    "RERANKER_SERVICE_TOKEN",
    "SEED_ADMIN_PASSWORD",
    "MILVUS_ROOT_PASSWORD",
    "MILVUS_IMPORT_PASSWORD",
    "MILVUS_QUERY_PASSWORD",
    "MILVUS_WORKER_PASSWORD",
    "SEED_MAINTAINER_PASSWORD",
    "SEED_REVIEWER_PASSWORD",
    "SEED_READER_PASSWORD",
    "SEED_OUTSIDER_PASSWORD",
}


def main():
    if TARGET.exists():
        raise SystemExit(".env.cpu already exists; refusing to overwrite")
    lines = []
    for line in (ROOT / ".env.cpu.example").read_text(encoding="utf-8").splitlines():
        name = line.partition("=")[0]
        lines.append(
            name + "=" + secrets.token_hex(24) if name in SECRET_NAMES else line
        )
    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8")
    key = ROOT / "infra/secrets/mongo-key"
    key.parent.mkdir(parents=True, exist_ok=True)
    if not key.exists():
        key.write_bytes(base64.b64encode(secrets.token_bytes(512)))
    print(
        "Created .env.cpu and replica-set key. Read local file for demo passwords; no secrets printed."
    )


if __name__ == "__main__":
    main()
