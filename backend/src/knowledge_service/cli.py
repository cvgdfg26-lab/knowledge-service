import argparse
import asyncio
import json
import logging
import os
import sys

from pymongo import MongoClient

from knowledge_service.core.config import get_settings
from knowledge_service.core.errors import require
from knowledge_service.core.security import passwords
from knowledge_service.migrations import migrate
from knowledge_service.runtime import Runtime
from knowledge_service.worker import Worker


def init_mongo(runtime):
    s = runtime.s
    bootstrap = os.environ.get("MONGODB_BOOTSTRAP_URI")
    require(bootstrap, "BOOTSTRAP_URI", "初始化需配置管理连接 MONGODB_BOOTSTRAP_URI")
    with MongoClient(bootstrap, serverSelectionTimeoutMS=5000) as client:
        if not client.admin.command("hello").get("setName"):
            client.admin.command(
                "replSetInitiate",
                {
                    "_id": s.mongodb_replica_set,
                    "members": [{"_id": 0, "host": os.environ.get("MONGODB_REPLICA_HOST", "mongodb:27017")}],
                },
            )
    # The replica host must match the name reachable by every CPU container.
    import time

    for attempt in range(30):
        try:
            runtime.store.ping()
            break
        except Exception:
            if attempt == 29:
                raise
            time.sleep(1)
    db = runtime.store.client[s.mongodb_database]
    query_writes = [
        "auth_sessions",
        "auth_attempts",
        "messages",
        "chat_sessions",
        "quota_leases",
        "idempotency_records",
        "feedback_tickets",
        "audit_events",
    ]
    role = "knowledge_query"
    privileges = [{"resource": {"db": s.mongodb_database, "collection": ""}, "actions": ["find"]}]
    privileges.extend(
        {
            "resource": {"db": s.mongodb_database, "collection": name},
            "actions": ["find", "insert", "update", "remove"],
        }
        for name in query_writes
    )
    role_spec = {"privileges": privileges, "roles": []}
    if db.command("rolesInfo", role)["roles"]:
        db.command("updateRole", role, **role_spec)
    else:
        db.command("createRole", role, **role_spec)
    for name in ("import", "query", "worker"):
        password = os.environ.get("MONGO_" + name.upper() + "_PASSWORD", "")
        require(len(password) >= 24, "SERVICE_PASSWORD", "服务账号密码至少24位")
        roles = [
            {"role": role if name == "query" else "readWrite", "db": s.mongodb_database},
            {"role": "readWrite", "db": s.graph_checkpoint_database},
        ]
        user = "kb_" + name
        command = "updateUser" if db.command("usersInfo", user)["users"] else "createUser"
        db.command(command, user, pwd=password, roles=roles)


def init_infra(runtime):
    init_mongo(runtime)
    runtime.objects.initialize()
    from knowledge_service.bootstrap import init_milvus

    init_milvus(runtime.s)


def seed_demo(runtime):
    require(runtime.s.app_env in {"dev", "test"}, "DEMO_ENV", "演示数据仅允许开发/测试环境")
    db = runtime.store.db
    for login in ("admin", "maintainer", "reviewer", "reader", "outsider"):
        password = os.environ.get("SEED_" + login.upper() + "_PASSWORD", "")
        require(len(password) >= 12, "DEMO_PASSWORD", f"请配置 SEED_{login.upper()}_PASSWORD（至少12位）")
        db.users.update_one(
            {"_id": login},
            {
                "$setOnInsert": {
                    "login": login,
                    "password_hash": passwords.hash(password),
                    "status": "active",
                    "auth_epoch": 1,
                    "system_admin": login == "admin",
                }
            },
            upsert=True,
        )
    for code, name in (
        ("education", "教育"),
        ("listening", "听书"),
        ("travel", "旅游"),
        ("parenting", "育儿"),
        ("finance", "金融"),
    ):
        db.business_domains.update_one(
            {"_id": code}, {"$setOnInsert": {"code": code, "name": name}}, upsert=True
        )
        kb_id = "kb_" + code
        db.knowledge_bases.update_one(
            {"_id": kb_id},
            {
                "$setOnInsert": {
                    "name": name + "演示库",
                    "domain": code,
                    "policy_epoch": 1,
                    "embedding_profile": "bge1024_bge_sparse",
                    "allow_parser_egress": False,
                    "allow_llm_egress": False,
                }
            },
            upsert=True,
        )
        for login, role in (
            ("admin", "admin"),
            ("maintainer", "maintainer"),
            ("reviewer", "reviewer"),
            ("reader", "reader"),
        ):
            db.kb_members.update_one(
                {"_id": kb_id + ":" + login},
                {"$setOnInsert": {"kb_id": kb_id, "user_id": login, "role": role}},
                upsert=True,
            )
    print("演示用户和五领域知识库已初始化；外发默认关闭。")


async def doctor(runtime, models=False):
    checks = {}
    for name, factory in (
        ("mongodb", lambda: runtime.store),
        ("minio", lambda: runtime.objects),
        ("milvus", lambda: runtime.vectors),
        ("neo4j", lambda: runtime.relations),
    ):
        try:
            await asyncio.to_thread(factory().ping)
            checks[name] = "ready"
        except Exception as error:
            checks[name] = {"status": "unavailable", "error_type": type(error).__name__}
    checks["embedding_config"] = (
        "ready"
        if (runtime.s.bge_m3_endpoint or runtime.s.bge_m3_service_url) and runtime.s.bge_model_revision
        else "missing"
    )
    if models:
        try:
            await runtime.models.embed(["非敏感部署连通性验证"])
            await runtime.models.rerank("验证", [{"_id": "probe", "text": "非敏感部署验证"}])
            checks["gpu_contract"] = "ready"
        except Exception as error:
            checks["gpu_contract"] = {"status": "unavailable", "error_type": type(error).__name__}
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if all(value == "ready" for value in checks.values()) else 1


def main():
    parser = argparse.ArgumentParser(prog="knowledge-service")
    parser.add_argument(
        "command",
        choices=["init-infra", "init-mongo", "migrate", "seed-demo", "doctor", "worker", "eval", "compare"],
    )
    parser.add_argument("--env", choices=["dev", "test", "staging", "prod"])
    parser.add_argument("--role", default="all")
    parser.add_argument("--models", action="store_true", help="明确执行非敏感 GPU 连通性调用")
    parser.add_argument("--config")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output")
    parser.add_argument("--runs", nargs="+")
    args = parser.parse_args()
    if args.env:
        os.environ["APP_ENV"] = args.env
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Presigned transfer URLs must not enter INFO logs.
    runtime = Runtime(get_settings())

    async def run():
        try:
            if args.command == "init-infra":
                await asyncio.to_thread(init_infra, runtime)
            elif args.command == "init-mongo":
                await asyncio.to_thread(init_mongo, runtime)
            elif args.command == "migrate":
                await asyncio.to_thread(migrate, runtime)
            elif args.command == "seed-demo":
                await asyncio.to_thread(seed_demo, runtime)
            elif args.command == "doctor":
                return await doctor(runtime, args.models)
            elif args.command == "worker":
                await Worker(runtime).run()
            else:
                from knowledge_service.evaluation import compare, evaluate

                if args.command == "eval":
                    await evaluate(runtime, args.config, args.split, args.output)
                else:
                    compare(args.runs)
            return 0
        finally:
            await runtime.close()

    try:
        return_code = asyncio.run(run())
    except KeyboardInterrupt:
        return_code = 130
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "code": getattr(error, "code", "DEPENDENCY_ERROR"),
                },
                ensure_ascii=False,
            )
        )
        return_code = 1
    sys.exit(return_code)


if __name__ == "__main__":
    main()
