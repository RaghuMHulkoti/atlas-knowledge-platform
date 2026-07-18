"""
pdf_parser.py

PDF file parser backed by pypdf.

Extracts the concatenated text of every page. Image-only / scanned PDFs yield
little or no text (no OCR is performed).
"""

import io

from pypdf import PdfReader

from app.core.exceptions import ConnectorException
from app.core.logging import get_logger
from app.infrastructure.connectors.files.base import BaseFileParser

logger = get_logger(__name__)


class PdfParser(BaseFileParser):
    """Extracts text from PDF files."""

    extensions = (".pdf",)

    def extract_text(self, data: bytes, filename: str) -> str:
        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception as exc:
            raise ConnectorException(f"Could not read PDF '{filename}': {exc}") from exc

        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)

        logger.info(
            "PdfParser: extracted %d page(s) of text from '%s'.",
            len(pages),
            filename,
        )
        return "\n\n".join(pages)
