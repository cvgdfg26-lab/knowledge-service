# 多领域知识库与 RAG 工作台

按 [开发手册](docs/开发手册.md)实现的首版工程：导入服务 `8000`、查询服务 `8001`、持久化 Worker，以及五领域统一的知识库、角色、文档版本与对话模型。

CPU 主机运行 FastAPI、LangGraph、MongoDB 副本集、Milvus、etcd、MinIO、Neo4j；GPU 主机只运行 BGE-M3（1024 维稠密 + 学习型稀疏）与 BGE-Reranker-Large。生成通过厂商 API，PDF/DOCX 解析通过 MinerU API。前端为原生 HTML/CSS/JS。

当前交付为可继续开发与联调的工程版本，完整验收边界见 [开发进度](docs/开发进度.md)。没有上线、GPU 实测或评测成绩声明。

准备远程环境时，按 [联调配置清单](docs/联调配置清单.md)填写服务器、模型与 API 信息；密码、Token 和私钥保留在本地配置文件。

当前提供的远程环境已启动，访问 **http://127.0.0.1:18080/**，账号密码见本地 `.env.remote` 的 `SEED_*_PASSWORD`。具体端口、启动命令及验证结果见 [服务启动说明](docs/服务启动说明.md)。本次使用 `infra/compose.remote.yml`，复用已有存储及 GPU 隧道，并为项目提供独立 MongoDB 副本集。

其他已有 CPU/GPU 配置的环境可参考 [现有环境接入说明](docs/现有环境接入说明.md)与 `infra/compose.existing.yml`。CPU 上的 SSH 隧道需要转发 6006、6008、6009；不要套用下文的新环境数据库初始化流程。

## 本地验证

Python 3.12–3.14、uv；Windows PowerShell 在仓库根目录执行：

```powershell
uv --cache-dir .cache/uv sync --project backend --frozen
backend/.venv/Scripts/python -m pytest backend/tests -q
backend/.venv/Scripts/ruff check backend services scripts
node --check frontend/app.js
```

测试使用明确的存储/模型替身，不下载模型、不调用付费 API。真实事务测试需隔离 MongoDB 副本集，设置 `RUN_INTEGRATION=1` 和以 `_test` 结尾的 `MONGODB_DATABASE` 后单独执行 `pytest -m integration`。

## CPU 主机部署

以下是开发环境启动流程，需可运行 Linux 容器的 Docker Engine/Compose。建议至少 8 核、32 GiB 内存与足够的持久磁盘；这是配置起点，容量需实测。

```powershell
python scripts/setup_local.py
docker compose --env-file .env.cpu -f infra/compose.cpu.yml config --quiet
docker compose --env-file .env.cpu -f infra/compose.cpu.yml build
docker compose --env-file .env.cpu -f infra/compose.cpu.yml up -d --wait mongodb etcd minio neo4j milvus
docker compose --env-file .env.cpu -f infra/compose.cpu.yml run --rm kb-admin init-infra
docker compose --env-file .env.cpu -f infra/compose.cpu.yml run --rm kb-admin migrate
docker compose --env-file .env.cpu -f infra/compose.cpu.yml run --rm kb-admin seed-demo
docker compose --env-file .env.cpu -f infra/compose.cpu.yml up -d kb-import kb-query kb-worker gateway
docker compose --env-file .env.cpu -f infra/compose.cpu.yml run --rm kb-admin doctor --models
```

本轮已生成本地 `.env.cpu` 与 `infra/secrets/mongo-key`；不要重复生成或提交这些文件。脚本拒绝覆盖已有配置。填入 GPU 私网 URL、模型 revision、厂商密钥后运行 doctor；缺少 GPU 配置时 doctor 返回失败，存活探针仍可访问。

访问 `http://localhost:8080`。演示账号 `admin / maintainer / reviewer / reader / outsider` 的密码在本地 `.env.cpu` 对应 `SEED_*_PASSWORD`，没有固定公共密码。默认关闭解析/生成外发；管理员在页面分别打开策略后，PDF/DOCX 与厂商生成才可用。MD/TXT 不需要解析 API。

1. 用 `maintainer` 上传 `data/samples/education/course-policy.md`，查询任务状态，查看预览并提交审核。
2. 用 `reviewer` 核对正文与定位，审核通过，点击发布并确认索引任务成功。
3. 用 `reader` 搜索或提问，核对引用、下载来源。未允许生成外发时只返回授权资料。
4. 用 `admin` 调整文档 ACL 或撤销成员，再检查引用与会话记录是否失效。

上传、审核、发布是独立步骤；提交者不能审核自己的内容。GPU、向量或图谱准备失败不会切换当前发布指针。所有写 API 需会话 Cookie、`X-CSRF-Token` 和 `Idempotency-Key`；浏览器负责生成。

CPU 存储端口没有发布到宿主机。`8000/8001` 默认仅绑定环回，开发网关为 `8080`。远程访问可用 SSH 隧道，或由部署人员配置可信域名、TLS 代理、`ALLOWED_ORIGINS`/`TRUSTED_HOSTS`；`staging/prod` 必须 `COOKIE_SECURE=true`。当前 Nginx 配置是 HTTP 开发入口，不能直接当 HTTPS 生产入口。

MinIO 使用固定源码构建，业务 bucket/账户与 Milvus 内部 bucket/账户分离。其社区维护状态需在部署时确认，见 [官方仓库](https://github.com/minio/minio)。MongoDB 服务账号区分导入、查询、Worker；查询账号限制业务集合写权限。Milvus 开启认证并为三类服务授予指定 collection 权限，管理员初始化后更换默认 root 密码。首次初始化不要开放存储端口。

## GPU 主机部署

安装适配 CUDA 12.4 的 NVIDIA 驱动与 NVIDIA Container Toolkit，准备从可信模型仓库获取的完整固定快照。将快照目录只读挂载，记录 revision；不能仅改 revision 字符串而仍使用另一份模型文件。

```bash
cp .env.gpu.example .env.gpu
# 编辑模型路径、revision、服务 Token、GPU ID 与私网绑定地址。
docker compose --env-file .env.gpu -f infra/compose.gpu.yml config --quiet
docker compose --env-file .env.gpu -f infra/compose.gpu.yml up -d --build
```

将相同的服务 Token/revision 和可从 CPU 主机访问的私网 URL 填入 `.env.cpu`。`BGE_M3_SERVICE_URL=http://GPU私网地址:8100`，`RERANKER_SERVICE_URL=http://GPU私网地址:8101`；只允许 CPU 主机访问，跨不可信网络须配置 TLS。GPU 默认绑定环回，需要按实际私网地址修改 `GPU_BIND_IP`。

嵌入拒绝超长输入，始终返回稠密与稀疏两种结果；重排按固定最大 token 长度截断并返回策略信息。单个容器一次执行一批推理，忙时返回 429。两个容器共用同一 GPU 时仍需核对显存；配置中的 GPU ID/显存限额不是已验证容量。GPU 依赖固定直接版本，完整传递依赖与镜像 digest 在目标主机验证后归档。

## API 与工程入口

- 导入 OpenAPI：`http://localhost:8000/docs`；查询 OpenAPI：`http://localhost:8001/docs`。
- `/api/v1/health/live` 检查进程；`/api/v1/health/ready` 检查存储与嵌入配置。后者不代表 GPU 或厂商 API 已实际连通，GPU 调用用 `doctor --models` 验证。
- `backend/src/knowledge_service/services.py`：权限、幂等、审核、CAS 发布与会话配额。
- `worker.py`：租约/接管、解析、索引准备、清理与撤权事件消费。
- `rag.py`：授权召回、重排、BM25 降级、来源与流式回答。
- `services/gpu_service.py`：真实 GPU 推理契约；没有运行时伪向量模式。
- `evals/README.md`：实际检索评测与同数据集对比，无预填成绩。
- `eval/README.md`：Ragas 五项问答评估，程序为 `eval/eval.py`，输入 `eval/qa.csv`、输出 `eval/eval_result.csv`，两者均采用 UTF-8 BOM。

若要从源码连接自行准备的本地服务，将 `.env.cpu` 中 `MONGODB_URI`、MinIO、Milvus、Neo4j 和 GPU 配置改为本地可达地址；Compose 内部 DNS 不能直接用于 Windows 宿主进程。CPU 双服务可分别使用 `uv run --project backend uvicorn knowledge_service.import_app:app --port 8000` 与 `uv run --project backend uvicorn knowledge_service.query_app:app --port 8001` 启动。
