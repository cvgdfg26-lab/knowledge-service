"""Run explicitly against a disposable Mongo replica set: RUN_INTEGRATION=1."""

import os

import pytest

from knowledge_service.core.config import Settings
from knowledge_service.repositories.mongo import MongoStore


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1", reason="requires configured disposable Mongo replica set"
)
def test_real_mongo_transaction_abort():
    s = Settings()
    assert s.mongodb_database.endswith("_test"), "integration tests require a *_test database"
    store = MongoStore(s)
    store.ping()
    key = "integration_transaction_abort"
    try:

        def abort(tx):
            tx.insert("audit_events", {"_id": key})
            raise RuntimeError("abort")

        with pytest.raises(RuntimeError):
            store.transaction(abort)
        assert store.read(lambda tx: tx.get("audit_events", key)) is None
    finally:
        store.close()
