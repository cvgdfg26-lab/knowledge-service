import re

from knowledge_service.core.errors import require
from knowledge_service.core.security import digest


def split_text(
    text: str, version_id: str, kb_id: str, blocks=None, max_chars=800, overlap=60, source_format="pdf"
):
    """Structural character baseline. The manifest records its unit; no fake token counts."""
    require(max_chars > overlap >= 0, "SPLIT_CONFIG", "切分参数无效", 422)
    source = []
    heading = "正文"
    if blocks is not None:
        for index, block in enumerate(blocks):
            content = (
                block.get("text") or block.get("table_body") or "\n".join(block.get("image_caption", []))
            )
            if not isinstance(content, str) or not content.strip():
                continue
            page = block.get("page_idx")
            if source_format == "pdf":
                require(isinstance(page, int) and page >= 0, "SOURCE_LOCATION", "PDF 块缺少有效页码", 422)
                locator = {"type": "pdf", "page_number": page + 1, "block_index": index}
            else:
                locator = {"type": source_format, "block_index": index, "source": "mineru_content_list"}
            if block.get("bbox") is not None:
                locator["bbox"] = block["bbox"]
                locator["coordinate_source"] = "mineru_content_list"
            source.append((content, heading, locator))
    else:
        lines = text.splitlines()
        start, buffer = 1, []
        for line_number, line in enumerate([*lines, ""], 1):
            match = re.match(r"^#{1,6}\s+(.+)", line)
            if (not line.strip() or match) and buffer:
                source.append(
                    (
                        "\n".join(buffer),
                        heading,
                        {"type": "text", "line_start": start, "line_end": line_number - 1},
                    )
                )
                buffer = []
            if match:
                heading = match[1].strip()
            if line.strip():
                if not buffer:
                    start = line_number
                buffer.append(line)
    result = []
    for content, heading, locator in source:
        # Keep a table intact when possible; repeated header accompanies long Markdown tables.
        table_lines = content.splitlines()
        table_header = ""
        if len(table_lines) > 2 and "|" in table_lines[0] and re.search(r"[-:]{2}", table_lines[1]):
            table_header = "\n".join(table_lines[:2]) + "\n"
        step = max_chars - overlap
        for offset in range(0, len(content), step):
            piece = content[offset : offset + max_chars]
            if offset and table_header:
                piece = table_header + piece
            ordinal = len(result)
            result.append(
                {
                    "_id": f"{version_id}_{ordinal}",
                    "version_id": version_id,
                    "kb_id": kb_id,
                    "ordinal": ordinal,
                    "text": piece,
                    "heading": heading,
                    "locator": locator,
                    "text_hash": digest(piece),
                }
            )
            if offset + max_chars >= len(content):
                break
    require(bool(result), "PARSE_EMPTY", "资料未提取到有效正文", 422)
    return result
