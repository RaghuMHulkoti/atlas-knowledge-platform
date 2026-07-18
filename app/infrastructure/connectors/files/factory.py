"""
factory.py

Resolves the correct file parser for a given file extension.

New file types are added by implementing a BaseFileParser and registering it in
``_PARSERS`` — no other module changes. This is the extension point that keeps
the upload pipeline open for extension, closed for modification.
"""

from pathlib import Path

from app.core.exceptions import ConnectorException
from app.infrastructure.connectors.files.base import BaseFileParser
from app.infrastructure.connectors.files.docx_parser import DocxParser
from app.infrastructure.connectors.files.pdf_parser import PdfParser
from app.infrastructure.connectors.files.text_parser import TextParser


class FileParserFactory:
    """Maps a filename/extension to a concrete BaseFileParser instance."""

    # Instantiate once — parsers are stateless.
    _PARSERS: tuple[BaseFileParser, ...] = (
        TextParser(),
        PdfParser(),
        DocxParser(),
    )

    @classmethod
    def _by_extension(cls) -> dict[str, BaseFileParser]:
        mapping: dict[str, BaseFileParser] = {}
        for parser in cls._PARSERS:
            for ext in parser.extensions:
                mapping[ext] = parser
        return mapping

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Return every extension the upload pipeline can handle."""
        return sorted(cls._by_extension().keys())

    @classmethod
    def for_filename(cls, filename: str) -> BaseFileParser:
        """
        Return the parser that handles *filename*'s extension.

        Raises:
            ConnectorException: If no parser supports the extension.
        """
        ext = Path(filename).suffix.lower()
        parser = cls._by_extension().get(ext)
        if parser is None:
            raise ConnectorException(
                f"Unsupported file type '{ext}'. "
                f"Supported types: {', '.join(cls.supported_extensions())}"
            )
        return parser
