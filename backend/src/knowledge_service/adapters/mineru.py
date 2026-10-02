import asyncio
import io
import ipaddress
import json
import socket
import zipfile
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import httpx

from knowledge_service.core.errors import require


def unpack_artifacts(data: bytes, max_bytes: int, max_files: int = 1000):
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        require(len(archive.infolist()) <= max_files, "ARCHIVE_LIMIT", "解析产物文件过多", 422)
        require(
            sum(f.file_size for f in archive.infolist()) <= max_bytes,
            "ARCHIVE_LIMIT",
            "解析产物展开后过大",
            422,
        )
        for info in archive.infolist():
            path = PurePosixPath(info.filename.replace("\\", "/"))
            require(
                not path.is_absolute() and ".." not in path.parts and ":" not in info.filename,
                "ARCHIVE_PATH",
                "解析产物路径不合法",
                422,
            )
            require(
                (info.external_attr >> 16) & 0o170000 != 0o120000,
                "ARCHIVE_PATH",
                "解析产物不允许符号链接",
                422,
            )
            if not info.is_dir():
                require(str(path) not in files, "ARCHIVE_PATH", "解析产物路径重复", 422)
                files[str(path)] = archive.read(info)
    require(files, "PARSE_EMPTY", "解析产物为空", 422)
    return files


class MinerU:
    def __init__(self, settings, transport=None):
        self.s = settings
        self.http = httpx.AsyncClient(timeout=60, follow_redirects=False, transport=transport)

    async def submit(self, filename, version_id):
        require(self.s.mineru_api_token.get_secret_value(), "PARSER_NOT_CONFIGURED", "未配置解析 API", 503)
        response = await self.http.post(
            self.s.mineru_api_base_url.rstrip("/") + "/file-urls/batch",
            headers={"Authorization": "Bearer " + self.s.mineru_api_token.get_secret_value()},
            json={
                "files": [{"name": filename, "data_id": version_id}],
                "model_version": self.s.mineru_model_version,
            },
        )
        response.raise_for_status()
        value = response.json()
        require(value.get("code") == 0, "PARSER_SUBMIT", "解析服务拒绝任务", 503)
        return value["data"]["batch_id"], value["data"]["file_urls"][0]

    async def validate_transfer(self, url):
        value = urlsplit(url)
        require(
            value.scheme == "https"
            and value.hostname in self.s.mineru_transfer_hosts
            and not value.username
            and not value.password
            and value.port in {None, 443},
            "PARSER_URL",
            "解析传输地址不在允许范围",
            422,
        )
        addresses = await asyncio.to_thread(socket.getaddrinfo, value.hostname, 443)
        require(
            all(ipaddress.ip_address(row[4][0]).is_global for row in addresses),
            "PARSER_URL",
            "解析传输地址不可指向内网",
            422,
        )

    async def upload(self, url, data):
        await self.validate_transfer(url)
        response = await self.http.put(url, content=data)  # Never forward the API Bearer token.
        response.raise_for_status()

    async def poll(self, batch_id):
        response = await self.http.get(
            self.s.mineru_api_base_url.rstrip("/") + "/extract-results/batch/" + batch_id,
            headers={"Authorization": "Bearer " + self.s.mineru_api_token.get_secret_value()},
        )
        response.raise_for_status()
        value = response.json()
        require(value.get("code") == 0, "PARSER_POLL", "解析任务查询失败", 503)
        rows = value["data"]["extract_result"]
        require(len(rows) == 1, "PARSER_RESPONSE", "解析任务映射异常", 503)
        return rows[0]

    async def download(self, url):
        await self.validate_transfer(url)
        data = bytearray()
        async with self.http.stream("GET", url) as response:
            response.raise_for_status()
            async for block in response.aiter_bytes():
                data.extend(block)
                require(len(data) <= self.s.max_archive_bytes, "ARCHIVE_LIMIT", "解析产物下载过大", 422)
        files = unpack_artifacts(bytes(data), self.s.max_archive_bytes)
        markdown = next((v.decode("utf-8") for k, v in files.items() if k.endswith(".md")), None)
        blocks = next((json.loads(v) for k, v in files.items() if k.endswith("content_list.json")), None)
        require(
            markdown is not None and isinstance(blocks, list),
            "PARSER_SCHEMA",
            "解析产物缺少正文或定位 JSON，需核对 API Schema",
            422,
        )
        return files, markdown, blocks

    async def close(self):
        await self.http.aclose()
