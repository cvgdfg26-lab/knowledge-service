"""Pin the existing BGE services to the actual local weight snapshot."""

import hashlib
import importlib
import os
import sys
from pathlib import Path

kind = os.environ["KB_MODEL_KIND"]
root = Path(os.environ["KB_MODEL_PATH"])
digest = hashlib.sha256()
files = sorted(
    p
    for p in root.rglob("*")
    if p.is_file()
    and not any(part.startswith(".") for part in p.relative_to(root).parts)
)
for path in files:
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(b"\0")
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
revision = "sha256:" + digest.hexdigest()
Path(os.environ["KB_REVISION_FILE"]).write_text(revision)
sys.path.insert(0, "/root/autodl-tmp/apps/model_services/0525")
module = "bge_m3_service" if kind == "embedding" else "bge_reranker_service"
app = importlib.import_module(module).app
name = "BAAI/bge-m3" if kind == "embedding" else "BAAI/bge-reranker-large"


@app.middleware("http")
async def metadata(request, call_next):
    response = await call_next(request)
    response.headers["X-Model-Revision"] = revision
    response.headers["X-Model-Name"] = name
    return response
