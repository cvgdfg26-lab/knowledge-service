import os

from pymilvus import MilvusClient

from knowledge_service.core.errors import require


def init_milvus(settings):
    password = os.environ.get("MILVUS_ROOT_PASSWORD", "")
    require(len(password) >= 24, "MILVUS_PASSWORD", "Milvus 管理密码至少24位")
    try:
        client = MilvusClient(uri=settings.milvus_uri, token="root:" + password, timeout=10)
        client.list_users()
    except Exception:
        # Initial credentials are tried only by this explicit administrator bootstrap command.
        client = MilvusClient(uri=settings.milvus_uri, token="root:Milvus", timeout=10)
        client.update_password("root", "Milvus", password, reset_connection=True)
    try:
        roles = set(client.list_roles())
        users = set(client.list_users())
        for name in ("import", "query", "worker"):
            user, role = "kb_" + name, "knowledge_" + name
            service_password = os.environ.get("MILVUS_" + name.upper() + "_PASSWORD", "")
            require(len(service_password) >= 24, "MILVUS_PASSWORD", "Milvus 服务密码至少24位")
            if user not in users:
                client.create_user(user, service_password)
            if role not in roles:
                client.create_role(role)
            privileges = ["Query", "Search", "DescribeCollection", "GetLoadState", "GetFlushState"]
            if name == "worker":
                privileges += ["Insert", "Upsert", "Delete", "Flush", "Load"]
            for privilege in privileges:
                client.grant_privilege_v2(role, privilege, settings.milvus_collection, db_name="default")
            client.grant_role(user, role)
    finally:
        client.close()
