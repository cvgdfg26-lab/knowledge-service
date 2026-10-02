from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_deployment_boundaries_and_private_storage():
    cpu = yaml.safe_load((ROOT / "infra/compose.cpu.yml").read_text(encoding="utf-8"))
    services = cpu["services"]
    assert {
        "kb-import",
        "kb-query",
        "kb-worker",
        "mongodb",
        "milvus",
        "etcd",
        "minio",
        "neo4j",
    } <= services.keys()
    for name in ("mongodb", "milvus", "etcd", "minio", "neo4j"):
        assert "ports" not in services[name]
    for name in ("kb-import", "kb-query", "kb-worker"):
        env = services[name]["environment"]
        assert (
            "MONGO_ROOT_PASSWORD" not in env
            and "MINIO_ROOT_PASSWORD" not in env
            and "MILVUS_ROOT_PASSWORD" not in env
        )
    assert "LLM_API_KEY" not in services["kb-worker"]["environment"]
    assert "MINERU_API_TOKEN" not in services["kb-query"]["environment"]
    assert (
        services["kb-query"]["environment"]["MINIO_ACCESS_KEY"]
        != services["kb-worker"]["environment"]["MINIO_ACCESS_KEY"]
    )
    gpu = yaml.safe_load((ROOT / "infra/compose.gpu.yml").read_text(encoding="utf-8"))
    assert set(gpu["services"]) == {"bge-m3", "bge-reranker"}
    for service in gpu["services"].values():
        assert "gpu" in service["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"]
        assert "MONGODB_URI" not in service["environment"]


def test_frontend_feedback_options_match_api_enum():
    from html.parser import HTMLParser

    from knowledge_service.schemas import FeedbackInput

    class FeedbackOptions(HTMLParser):
        def __init__(self):
            super().__init__()
            self.active, self.options = False, []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "select":
                self.active = attrs.get("id") == "feedback-type"
            if tag == "option" and self.active:
                self.options.append(attrs["value"])

        def handle_endtag(self, tag):
            if tag == "select":
                self.active = False

    parser = FeedbackOptions()
    parser.feed((ROOT / "frontend/index.html").read_text(encoding="utf-8"))
    for category in parser.options:
        FeedbackInput(message_id="test", category=category, detail="test")
    assert len(parser.options) == 3
