"""Test doubles only: production always uses real MongoDB transactions and real models."""

import copy
import threading
from datetime import timedelta
from types import SimpleNamespace

import mongomock
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from knowledge_service.core.config import Settings
from knowledge_service.core.security import now, passwords
from knowledge_service.repositories.mongo import UnitOfWork
from knowledge_service.services import Domain


class MemoryStore:
    def __init__(self):
        self.db = mongomock.MongoClient(tz_aware=True).knowledge_test
        self.lock = threading.RLock()

    def read(self, callback):
        with self.lock:
            return callback(UnitOfWork(self.db))

    def transaction(self, callback):
        with self.lock:
            snapshot = {n: copy.deepcopy(list(self.db[n].find())) for n in self.db.list_collection_names()}
            try:
                return callback(UnitOfWork(self.db))
            except Exception:
                for name in self.db.list_collection_names():
                    self.db[name].delete_many({})
                for name, documents in snapshot.items():
                    if documents:
                        self.db[name].insert_many(documents)
                raise


class ObjectsDouble:
    def __init__(self):
        self.data = {}

    def put(self, bucket, key, data, *args):
        self.data[bucket, key] = data

    def get(self, bucket, key):
        return self.data[bucket, key]

    def remove(self, bucket, key):
        self.data.pop((bucket, key), None)


class ModelsDouble:
    def __init__(self):
        self.generate_calls = 0
        self.response = "学员可在开课前申请退款。[S1]"

    async def embed(self, texts):
        return [[1.0] + [0.0] * 1023 for _ in texts], [{1: 0.8} for _ in texts], 0.01

    async def rerank(self, question, candidates):
        return candidates

    async def generate(self, *args):
        self.generate_calls += 1
        yield self.response


class VectorsDouble:
    def __init__(self, store):
        self.store = store
        self.failure = None
        self.writes = []

    def write(self, chunks, dense, sparse):
        if self.failure:
            raise self.failure
        self.writes.extend(c["_id"] for c in chunks)

    def search(self, collection, versions, *args):
        return [
            {"chunk_id": c["_id"], "score": 1}
            for c in self.store.read(lambda tx: tx.find("chunks", {"version_id": {"$in": versions}}))
        ]

    def delete_version(self, *args):
        pass


class RelationsDouble:
    def write(self, *args):
        pass

    def delete_version(self, *args):
        pass


@pytest.fixture
def runtime(tmp_path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        frontend_path=tmp_path,
        bge_model_revision="test-fixture-not-real-model",
    )
    store = MemoryStore()
    password_hash = passwords.hash("test-password-12345")
    for login in ("admin", "maintainer", "reviewer", "reader", "outsider"):
        store.db.users.insert_one(
            {
                "_id": login,
                "login": login,
                "password_hash": password_hash,
                "auth_epoch": 1,
                "status": "active",
                "system_admin": login == "admin",
            }
        )
    store.db.knowledge_bases.insert_one(
        {
            "_id": "kb",
            "name": "教育",
            "domain": "education",
            "policy_epoch": 1,
            "allow_llm_egress": False,
            "allow_parser_egress": False,
        }
    )
    for login in ("admin", "maintainer", "reviewer", "reader"):
        store.db.kb_members.insert_one({"_id": "kb:" + login, "kb_id": "kb", "user_id": login, "role": login})
    for n in range(2):
        store.db.quota_leases.insert_one({"_id": str(n), "expires_at": now() - timedelta(days=1)})
    return SimpleNamespace(
        s=settings,
        store=store,
        domain=Domain(store, settings),
        objects=ObjectsDouble(),
        models=ModelsDouble(),
        vectors=VectorsDouble(store),
        relations=RelationsDouble(),
        checkpointer=InMemorySaver(),
    )


@pytest.fixture
def published(runtime):
    db = runtime.store.db
    db.documents.insert_one(
        {
            "_id": "doc",
            "kb_id": "kb",
            "revision": 2,
            "active_version_id": "old",
            "blocked_at": None,
            "deleted_at": None,
        }
    )
    db.document_versions.insert_one(
        {
            "_id": "old",
            "document_id": "doc",
            "kb_id": "kb",
            "state": "published",
            "collection": "test",
            "generation": "g001",
        }
    )
    db.index_manifests.insert_one({"_id": "old", "model_revision": runtime.s.bge_model_revision})
    db.chunks.insert_one(
        {
            "_id": "old_0",
            "version_id": "old",
            "kb_id": "kb",
            "text": "开课前可退款。",
            "heading": "退款政策",
            "ordinal": 0,
            "locator": {"type": "text", "line_start": 1, "line_end": 2},
        }
    )
    return runtime
