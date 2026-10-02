"""Real GPU inference only. Model files are pinned and mounted read-only by the operator."""

import asyncio
import math
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

ROLE = os.getenv("MODEL_ROLE", "embedding")
MODEL = "BAAI/bge-m3" if ROLE == "embedding" else "BAAI/bge-reranker-large"
REVISION = os.getenv("MODEL_REVISION", "")
TOKEN = os.getenv("SERVICE_TOKEN", "")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/model"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "2048" if ROLE == "embedding" else "512"))
gate = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    import torch
    from FlagEmbedding import BGEM3FlagModel, FlagReranker

    if (
        ROLE not in {"embedding", "reranker"}
        or len(TOKEN) < 24
        or not REVISION
        or not MODEL_PATH.is_dir()
    ):
        raise RuntimeError(
            "MODEL_ROLE, pinned MODEL_PATH/MODEL_REVISION and SERVICE_TOKEN>=24 are required"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required; there is no synthetic/CPU fallback")
    model_class = BGEM3FlagModel if ROLE == "embedding" else FlagReranker
    app.state.model = model_class(str(MODEL_PATH), use_fp16=True, devices=["cuda:0"])
    app.state.revision = REVISION
    yield


app = FastAPI(title=f"{MODEL} inference", lifespan=lifespan)


def authorized(authorization: Annotated[str | None, Header()] = None):
    if not TOKEN or not secrets.compare_digest(authorization or "", "Bearer " + TOKEN):
        raise HTTPException(401, "unauthorized")


class Embeddings(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=16)
    return_dense: bool = True
    return_sparse: bool = True


class Candidate(BaseModel):
    candidate_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=12000)


class Rerank(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    candidates: list[Candidate] = Field(min_length=1, max_length=30)
    top_n: int = Field(ge=1, le=30)


async def infer(callback):
    if gate.locked():
        raise HTTPException(429, "GPU busy; retry with backoff")
    async with gate:
        # A disconnect must not release the lock while the CUDA thread is still executing.
        work = asyncio.create_task(asyncio.to_thread(callback))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            await work
            raise
        except Exception:  # noqa: BLE001 -- sanitized public response, no model internals
            raise HTTPException(503, "inference failed") from None


@app.get("/health/live")
def live():
    return {"status": "live", "role": ROLE}


@app.get("/health/ready", dependencies=[Depends(authorized)])
def ready():
    return {
        "ready": hasattr(app.state, "model"),
        "model": MODEL,
        "model_revision": REVISION,
    }


@app.post("/v1/embeddings", dependencies=[Depends(authorized)])
async def embedding(body: Embeddings):
    if ROLE != "embedding":
        raise HTTPException(404)
    if (
        not body.return_dense
        or not body.return_sparse
        or any(not t.strip() or len(t) > 12000 for t in body.texts)
    ):
        raise HTTPException(422, "dense+sparse required; text size invalid")
    token_lengths = [
        len(app.state.model.tokenizer(t, truncation=False)["input_ids"])
        for t in body.texts
    ]
    if any(n > MAX_LENGTH for n in token_lengths):
        raise HTTPException(
            422, "embedding token limit; split the document more finely"
        )
    result = await infer(
        lambda: app.state.model.encode(
            body.texts,
            batch_size=len(body.texts),
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
    )
    dense = result["dense_vecs"].tolist()
    sparse = [
        {str(k): float(v) for k, v in row.items() if float(v) > 0}
        for row in result["lexical_weights"]
    ]
    if any(
        len(v) != 1024 or not all(math.isfinite(x) for x in v) for v in dense
    ) or any(not v for v in sparse):
        raise HTTPException(503, "model contract invalid")
    return {
        "model": MODEL,
        "model_revision": REVISION,
        "dense_vectors": dense,
        "sparse_vectors": sparse,
        "dimension": 1024,
        "input_tokens": token_lengths,
        "truncated": False,
    }


@app.post("/v1/rerank", dependencies=[Depends(authorized)])
async def rerank(body: Rerank):
    if ROLE != "reranker":
        raise HTTPException(404)
    if len({c.candidate_id for c in body.candidates}) != len(
        body.candidates
    ) or body.top_n > len(body.candidates):
        raise HTTPException(422, "candidate ids/top_n invalid")
    pairs = [[body.query, c.text] for c in body.candidates]
    scores = await infer(
        lambda: app.state.model.compute_score(
            pairs, batch_size=min(8, len(pairs)), max_length=MAX_LENGTH, normalize=True
        )
    )
    if isinstance(scores, (float, int)):
        scores = [scores]
    rows = [
        {"candidate_id": c.candidate_id, "score": float(s)}
        for c, s in zip(body.candidates, scores, strict=True)
    ]
    if not all(math.isfinite(row["score"]) for row in rows):
        raise HTTPException(503, "invalid scores")
    return {
        "model": MODEL,
        "model_revision": REVISION,
        "results": sorted(rows, key=lambda row: row["score"], reverse=True)[
            : body.top_n
        ],
        "max_length": MAX_LENGTH,
        "truncation_policy": "tokenizer longest_first",
    }
