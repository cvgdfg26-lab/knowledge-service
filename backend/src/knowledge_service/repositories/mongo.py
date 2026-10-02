from pymongo import MongoClient, ReturnDocument
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern

from knowledge_service.core.errors import ServiceError


class UnitOfWork:
    def __init__(self, db, session=None):
        self.db = db
        self.session = session

    def get(self, name, key):
        return self.db[name].find_one({"_id": key}, session=self.session)

    def find(self, name, query=None, limit=1000):
        return list(self.db[name].find(query or {}, session=self.session).limit(limit))

    def insert(self, name, value):
        self.db[name].insert_one(value, session=self.session)
        return value

    def patch(self, name, key, values, expected=None):
        result = self.db[name].find_one_and_update(
            {"_id": key, **(expected or {})},
            {"$set": values},
            return_document=ReturnDocument.AFTER,
            session=self.session,
        )
        if result is None:
            raise ServiceError("CONFLICT", "资源状态已变化，请刷新后重试", 409)
        return result

    def delete(self, name, query):
        return self.db[name].delete_many(query, session=self.session).deleted_count


class MongoStore:
    def __init__(self, settings):
        self.client = MongoClient(
            settings.mongodb_uri.get_secret_value(),
            tz_aware=True,
            serverSelectionTimeoutMS=4000,
            connectTimeoutMS=4000,
        )
        self.db = self.client.get_database(
            settings.mongodb_database,
            read_concern=ReadConcern("majority"),
            write_concern=WriteConcern("majority"),
        )

    def read(self, callback):
        return callback(UnitOfWork(self.db))

    def transaction(self, callback):
        with self.client.start_session() as session:
            return session.with_transaction(
                lambda current: callback(UnitOfWork(self.db, current)),
                read_concern=ReadConcern("snapshot"),
                write_concern=WriteConcern("majority"),
            )

    def ping(self):
        result = self.client.admin.command("hello")
        if not result.get("setName"):
            raise ServiceError("REPLICA_SET_REQUIRED", "MongoDB 必须配置副本集", 503)
        # Probe a real transaction; hello alone does not validate transaction access.
        self.transaction(lambda tx: tx.find("knowledge_bases", limit=1))

    def close(self):
        self.client.close()
