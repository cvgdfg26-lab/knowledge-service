import io
import json

from minio import Minio
from minio.deleteobjects import DeleteObject
from neo4j import GraphDatabase
from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

from knowledge_service.core.errors import require


def scope_filter(version_ids):
    # JSON encoding is for the Milvus expression grammar, never a shell command.
    require(bool(version_ids), "EMPTY_SCOPE", "无可用资料", 404)
    return "version_id in " + json.dumps(list(version_ids), ensure_ascii=True)


class Objects:
    def __init__(self, s):
        self.s = s
        self.client = Minio(
            s.minio_endpoint,
            access_key=s.minio_access_key.get_secret_value(),
            secret_key=s.minio_secret_key.get_secret_value(),
            secure=s.minio_use_tls,
        )

    def put(self, bucket, key, data, content_type="application/octet-stream"):
        self.client.put_object(bucket, key, io.BytesIO(data), len(data), content_type=content_type)

    def get(self, bucket, key):
        response = self.client.get_object(bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def remove(self, bucket, key):
        versions = [
            DeleteObject(o.object_name, version_id=o.version_id)
            for o in self.client.list_objects(bucket, prefix=key, include_version=True)
            if o.object_name == key
        ]
        if versions:
            errors = list(self.client.remove_objects(bucket, versions))
            require(not errors, "OBJECT_DELETE", "部分对象版本未清理", 503)
        else:
            self.client.remove_object(bucket, key)

    def initialize(self):
        for bucket in (self.s.originals_bucket, self.s.derived_bucket, self.s.images_bucket):
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)

    def ping(self):
        for bucket in (self.s.originals_bucket, self.s.derived_bucket, self.s.images_bucket):
            require(self.client.bucket_exists(bucket), "BUCKET_MISSING", "存储未初始化", 503)


class Vectors:
    def __init__(self, s):
        self.s = s
        self.client = MilvusClient(uri=s.milvus_uri, token=s.milvus_token.get_secret_value(), timeout=10)

    def initialize(self):
        c = self.client
        name = self.s.milvus_collection
        if c.has_collection(name):
            self.ping()
            return
        schema = c.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=100, is_primary=True)
        for field in ("version_id", "kb_id", "generation"):
            schema.add_field(field, DataType.VARCHAR, max_length=100)
        schema.add_field(
            "content",
            DataType.VARCHAR,
            max_length=12000,
            enable_analyzer=True,
            analyzer_params={"tokenizer": "jieba"},
        )
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_bge", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="content_bm25",
                input_field_names=["content"],
                output_field_names=["sparse_bm25"],
                function_type=FunctionType.BM25,
            )
        )
        indexes = c.prepare_index_params()
        indexes.add_index(
            "dense", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 128}
        )
        indexes.add_index("sparse_bge", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        indexes.add_index("sparse_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        c.create_collection(name, schema=schema, index_params=indexes, consistency_level="Strong")

    def ping(self):
        description = self.client.describe_collection(self.s.milvus_collection)
        dense = next(f for f in description["fields"] if f["name"] == "dense")
        require(int(dense["params"]["dim"]) == 1024, "VECTOR_SCHEMA", "向量库维度不匹配", 503)

    def write(self, chunks, dense, sparse):
        rows = [
            {
                "chunk_id": c["_id"],
                "version_id": c["version_id"],
                "kb_id": c["kb_id"],
                "generation": self.s.generation,
                "content": c["text"],
                "dense": d,
                "sparse_bge": sp,
            }
            for c, d, sp in zip(chunks, dense, sparse, strict=True)
        ]
        self.client.upsert(self.s.milvus_collection, rows)
        self.client.flush(self.s.milvus_collection)
        self.client.load_collection(self.s.milvus_collection)
        self.verify(self.s.milvus_collection, chunks, dense[0], sparse[0])

    def verify(self, collection, chunks, dense, sparse):
        require(chunks, "VECTOR_EMPTY", "没有可用片段", 503)
        ids = [c["_id"] for c in chunks]
        found = self.client.query(
            collection,
            filter="chunk_id in " + json.dumps(ids),
            output_fields=["chunk_id"],
            consistency_level="Strong",
            limit=len(ids),
        )
        require({r["chunk_id"] for r in found} == set(ids), "VECTOR_NOT_READY", "向量尚未就绪", 503)
        # Verify all three actual search paths, not only flush/query visibility.
        scope = [chunks[0]["version_id"]]
        for mode in ("dense", "sparse", "bm25"):
            require(
                self.search(collection, scope, chunks[0]["text"], dense, sparse, mode),
                "SEARCH_NOT_READY",
                "检索通道未就绪",
                503,
            )

    def search(self, collection, versions, question, dense=None, sparse=None, mode="hybrid"):
        expression = scope_filter(versions)
        fields = ["chunk_id", "version_id", "kb_id"]
        requests = []
        if mode in {"dense", "hybrid"}:
            requests.append(
                AnnSearchRequest(
                    [dense],
                    "dense",
                    {"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=30,
                    expr=expression,
                )
            )
        if mode in {"sparse", "hybrid"}:
            requests.append(
                AnnSearchRequest(
                    [sparse], "sparse_bge", {"metric_type": "IP", "params": {}}, limit=30, expr=expression
                )
            )
        if mode == "bm25":
            requests.append(
                AnnSearchRequest(
                    [question],
                    "sparse_bm25",
                    {"metric_type": "BM25", "params": {}},
                    limit=30,
                    expr=expression,
                )
            )
        result = self.client.hybrid_search(
            collection,
            reqs=requests,
            ranker=RRFRanker(60),
            limit=30,
            output_fields=fields,
            consistency_level="Strong",
        )
        return [{"chunk_id": r["entity"]["chunk_id"], "score": r["distance"]} for r in result[0]]

    def delete_version(self, collection, version):
        self.client.delete(collection, filter=scope_filter([version]))

    def close(self):
        self.client.close()


class Relations:
    def __init__(self, s):
        self.s = s
        self.driver = GraphDatabase.driver(
            s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password.get_secret_value())
        )

    def initialize(self):
        for label, field in (
            ("DocumentVersion", "version_id"),
            ("Chunk", "chunk_id"),
            ("Entity", "entity_id"),
            ("Assertion", "assertion_id"),
        ):
            self.driver.execute_query(
                f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{field} IS UNIQUE",
                database_=self.s.neo4j_database,
            )

    def write(self, version, chunks):
        rows = [
            {
                "chunk_id": c["_id"],
                "entity_id": c["kb_id"] + ":" + c["heading"],
                "assertion_id": c["_id"] + ":topic",
                "name": c["heading"],
                "ordinal": c["ordinal"],
            }
            for c in chunks
        ]
        self.driver.execute_query(
            "MERGE (v:DocumentVersion {version_id:$version}) SET v.kb_id=$kb, v.generation=$gen "
            "WITH v UNWIND $rows AS row MERGE (c:Chunk {chunk_id:row.chunk_id}) "
            "SET c.version_id=$version, c.kb_id=$kb, c.generation=$gen "
            "MERGE (v)-[:CONTAINS]->(c) MERGE (e:Entity {entity_id:row.entity_id}) "
            "SET e.kb_id=$kb, e.name=row.name MERGE (a:Assertion {assertion_id:row.assertion_id}) "
            "SET a.kb_id=$kb, a.version_id=$version, a.generation=$gen, a.relation_type='HAS_SECTION', "
            "a.reviewed_hash=$hash MERGE (a)-[:SUBJECT]->(v) MERGE (a)-[:OBJECT]->(e) "
            "MERGE (a)-[:EVIDENCED_BY]->(c)",
            version=version["_id"],
            kb=version["kb_id"],
            gen=self.s.generation,
            hash=version["submitted_hash"],
            rows=rows,
            database_=self.s.neo4j_database,
        )
        self.verify(version, chunks)

    def verify(self, version, chunks):
        records, _, _ = self.driver.execute_query(
            "MATCH (a:Assertion {version_id:$version, generation:$gen, reviewed_hash:$hash})-[:EVIDENCED_BY]->(c:Chunk) "
            "RETURN c.chunk_id AS chunk_id",
            version=version["_id"],
            gen=self.s.generation,
            hash=version["submitted_hash"],
            database_=self.s.neo4j_database,
        )
        require(
            {r["chunk_id"] for r in records} == {c["_id"] for c in chunks},
            "GRAPH_NOT_READY",
            "基础图谱尚未就绪",
            503,
        )

    def delete_version(self, version):
        for label in ("Assertion", "Chunk", "DocumentVersion"):
            self.driver.execute_query(
                f"MATCH (n:{label} {{version_id:$version}}) DETACH DELETE n",
                version=version,
                database_=self.s.neo4j_database,
            )
        self.driver.execute_query(
            "MATCH (e:Entity) WHERE NOT (e)--() DELETE e", database_=self.s.neo4j_database
        )

    def ping(self):
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()
