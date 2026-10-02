from datetime import UTC, datetime

from knowledge_service.core.security import now

INDEXES = {
    "users": [([("login", 1)], {"unique": True})],
    "auth_sessions": [([("expires_at", 1)], {"expireAfterSeconds": 0})],
    "auth_attempts": [([("expires_at", 1)], {"expireAfterSeconds": 0})],
    "kb_members": [([("kb_id", 1), ("user_id", 1)], {"unique": True})],
    "document_versions": [
        ([("document_id", 1), ("revision", 1)], {"unique": True}),
        ([("artifact_keys.asset_id", 1)], {}),
    ],
    "documents": [([("kb_id", 1), ("deleted_at", 1), ("blocked_at", 1)], {})],
    "chunks": [([("version_id", 1), ("ordinal", 1)], {"unique": True})],
    "processing_tasks": [
        ([("business_key", 1)], {"unique": True}),
        ([("status", 1), ("next_run_at", 1), ("lease_until", 1)], {}),
    ],
    "messages": [
        ([("request_id", 1)], {"unique": True}),
        ([("session_id", 1), ("created_at", 1)], {}),
        ([("dependency_versions", 1)], {}),
    ],
    "idempotency_records": [([("expires_at", 1)], {"expireAfterSeconds": 0})],
    "outbox_events": [([("status", 1), ("created_at", 1)], {})],
}
COLLECTIONS = set(INDEXES) | {
    "knowledge_bases",
    "business_domains",
    "review_records",
    "index_manifests",
    "graph_manifests",
    "parser_submissions",
    "chat_sessions",
    "citations",
    "feedback_tickets",
    "audit_events",
    "security_ledger",
    "quota_leases",
    "orphan_candidates",
    "schema_revisions",
}


def migrate(runtime):
    runtime.store.ping()
    db = runtime.store.db
    for name in sorted(COLLECTIONS):
        validator = {
            "$jsonSchema": {
                "bsonType": "object",
                "required": ["_id"],
                "properties": {"_id": {"bsonType": "string"}},
            }
        }
        if name not in db.list_collection_names():
            db.create_collection(name, validator=validator)
        else:
            db.command("collMod", name, validator=validator, validationLevel="strict")
        for keys, options in INDEXES.get(name, []):
            db[name].create_index(keys, **options)
    for index in range(runtime.s.max_active_generations):
        db.quota_leases.update_one(
            {"_id": f"slot_{index}"},
            {"$setOnInsert": {"expires_at": datetime(2000, 1, 1, tzinfo=UTC), "request_id": None}},
            upsert=True,
        )
    runtime.objects.initialize()
    runtime.vectors.initialize()
    runtime.relations.initialize()
    # Constructing the saver creates its version-specific checkpoint indexes.
    _ = runtime.checkpointer
    db.schema_revisions.update_one(
        {"_id": "schema"}, {"$set": {"revision": 1, "updated_at": now()}}, upsert=True
    )
