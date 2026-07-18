"""Unit tests for file parsers, the parser factory, and UploadService."""

import io

import pytest

from app.core.exceptions import ConnectorException
from app.domain.knowledge.services.upload_service import UploadService
from app.infrastructure.connectors.files.docx_parser import DocxParser
from app.infrastructure.connectors.files.factory import FileParserFactory
from app.infrastructure.connectors.files.pdf_parser import PdfParser
from app.infrastructure.connectors.files.text_parser import TextParser

# A minimal, valid single-page PDF (with xref table) containing the text below.
_MINIMAL_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 58>>stream
BT /F1 18 Tf 20 100 Td (Atlas PDF ingestion works) Tj ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000052 00000 n
0000000101 00000 n
0000000209 00000 n
0000000317 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
389
%%EOF"""


def test_text_parser_decodes_bytes():
    parser = TextParser()
    assert parser.extract_text(b"# hello", "a.md") == "# hello"


def test_pdf_parser_extracts_text():
    text = PdfParser().extract_text(_MINIMAL_PDF, "sample.pdf")
    assert "Atlas PDF ingestion works" in text


def test_docx_parser_extracts_paragraphs():
    docx = pytest.importorskip("docx")
    buf = io.BytesIO()
    document = docx.Document()
    document.add_paragraph("Atlas deploy runbook.")
    document.add_paragraph("Run make deploy to ship.")
    document.save(buf)

    text = DocxParser().extract_text(buf.getvalue(), "runbook.docx")
    assert "make deploy" in text


def test_factory_resolves_by_extension():
    assert isinstance(FileParserFactory.for_filename("notes.md"), TextParser)
    assert isinstance(FileParserFactory.for_filename("report.docx"), DocxParser)


def test_factory_rejects_unsupported():
    with pytest.raises(ConnectorException):
        FileParserFactory.for_filename("archive.zip")


def test_factory_supported_extensions():
    exts = FileParserFactory.supported_extensions()
    assert ".pdf" in exts and ".docx" in exts and ".md" in exts and ".txt" in exts


def test_upload_service_builds_document():
    service = UploadService()
    doc = service.build_document(
        data=b"restart the worker with systemctl",
        filename="runbook.md",
        collection="atlas",
    )
    assert doc.title == "runbook"
    assert doc.repository == "atlas"
    assert doc.language == "markdown"
    assert doc.checksum
    assert doc.metadata["ingestion_source"] == "upload"


def test_upload_service_is_idempotent():
    service = UploadService()
    a = service.build_document(b"same bytes", "f.txt", "atlas")
    b = service.build_document(b"same bytes", "f.txt", "atlas")
    assert a.id == b.id


def test_upload_service_rejects_empty_text():
    with pytest.raises(ConnectorException):
        UploadService().build_document(b"   ", "empty.txt", "atlas")


def test_upload_service_rejects_unsupported_type():
    with pytest.raises(ConnectorException):
        UploadService().build_document(b"data", "image.png", "atlas")
