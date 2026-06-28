"""
test_recursive_splitter.py

Unit tests for RecursiveSplitter.
"""

from langchain_core.documents import Document

from app.ai.splitters.recursive_splitter import RecursiveSplitter


def _make_lc_doc(content: str, **meta) -> Document:
    return Document(page_content=content, metadata={"source": "test", **meta})


class TestRecursiveSplitter:
    def setup_method(self):
        self.splitter = RecursiveSplitter()

    def test_short_content_returns_single_chunk(self):
        doc = _make_lc_doc("Short content.")
        result = self.splitter.split([doc])
        assert len(result) == 1

    def test_long_content_produces_multiple_chunks(self):
        # Build content larger than CHUNK_SIZE (default 1000)
        long_text = "word " * 500  # ~2500 chars
        doc = _make_lc_doc(long_text)
        result = self.splitter.split([doc])
        assert len(result) > 1

    def test_metadata_preserved_on_all_chunks(self):
        long_text = "word " * 500
        doc = _make_lc_doc(long_text, path="src/main.py", language="python")
        chunks = self.splitter.split([doc])
        for chunk in chunks:
            assert chunk.metadata["path"] == "src/main.py"
            assert chunk.metadata["language"] == "python"

    def test_empty_list_returns_empty(self):
        result = self.splitter.split([])
        assert result == []

    def test_multiple_documents_are_all_split(self):
        docs = [_make_lc_doc("word " * 300) for _ in range(3)]
        result = self.splitter.split(docs)
        assert len(result) >= 3

    def test_chunk_content_is_non_empty(self):
        doc = _make_lc_doc("word " * 500)
        chunks = self.splitter.split([doc])
        for chunk in chunks:
            assert chunk.page_content.strip()
