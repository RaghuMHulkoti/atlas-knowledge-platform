"""
test_document_converter.py

Unit tests for DocumentConverter.
"""

from langchain_core.documents import Document

from app.ai.converters.document_converter import DocumentConverter
from app.domain.knowledge.models import KnowledgeDocument


def _make_doc(**kwargs) -> KnowledgeDocument:
    defaults = dict(
        id="abc123",
        source="https://github.com/org/repo",
        repository="repo",
        path="src/main.py",
        title="main",
        content="def hello(): pass",
        language="python",
        metadata={"size_bytes": 100},
    )
    defaults.update(kwargs)
    return KnowledgeDocument(**defaults)


class TestDocumentConverter:
    def setup_method(self):
        self.converter = DocumentConverter()

    def test_convert_returns_langchain_document(self):
        doc = _make_doc()
        result = self.converter.convert(doc)
        assert isinstance(result, Document)

    def test_page_content_is_content(self):
        doc = _make_doc(content="hello world")
        result = self.converter.convert(doc)
        assert result.page_content == "hello world"

    def test_metadata_preserves_source(self):
        doc = _make_doc(source="https://github.com/org/repo")
        result = self.converter.convert(doc)
        assert result.metadata["source"] == "https://github.com/org/repo"

    def test_metadata_preserves_path(self):
        doc = _make_doc(path="docs/README.md")
        result = self.converter.convert(doc)
        assert result.metadata["path"] == "docs/README.md"

    def test_metadata_preserves_language(self):
        doc = _make_doc(language="python")
        result = self.converter.convert(doc)
        assert result.metadata["language"] == "python"

    def test_metadata_preserves_repository(self):
        doc = _make_doc(repository="atlas")
        result = self.converter.convert(doc)
        assert result.metadata["repository"] == "atlas"

    def test_domain_metadata_is_merged(self):
        doc = _make_doc(metadata={"size_bytes": 512, "file_extension": ".py"})
        result = self.converter.convert(doc)
        assert result.metadata["size_bytes"] == 512
        assert result.metadata["file_extension"] == ".py"

    def test_none_values_are_excluded(self):
        doc = _make_doc(branch=None, commit_hash=None)
        result = self.converter.convert(doc)
        assert "branch" not in result.metadata
        assert "commit_hash" not in result.metadata

    def test_convert_many_returns_correct_count(self):
        docs = [_make_doc(id=f"id-{i}", content=f"content {i}") for i in range(5)]
        results = self.converter.convert_many(docs)
        assert len(results) == 5

    def test_convert_many_preserves_order(self):
        docs = [_make_doc(id=f"id-{i}", content=f"content {i}") for i in range(3)]
        results = self.converter.convert_many(docs)
        for i, result in enumerate(results):
            assert result.page_content == f"content {i}"
