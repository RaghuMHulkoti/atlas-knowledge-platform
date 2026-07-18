"""
text_parser.py

Plain-text and Markdown file parser.

Markdown is treated as text here — the raw Markdown is preserved so headings and
code fences remain available as context. Downstream chunking is Markdown-aware
enough for retrieval without a separate HTML render step.
"""

from app.infrastructure.connectors.files.base import BaseFileParser


class TextParser(BaseFileParser):
    """Extracts text from plain-text and Markdown files."""

    extensions = (".txt", ".md", ".markdown")

    def extract_text(self, data: bytes, filename: str) -> str:
        # errors="replace" mirrors the Git DocumentParser so malformed bytes
        # never abort an upload.
        return data.decode("utf-8", errors="replace")
