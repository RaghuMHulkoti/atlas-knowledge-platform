"""
test_embedding_service.py

Unit tests for EmbeddingService.

All tests mock BaseEmbeddingProvider — no real API calls are made.
"""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from app.ai.embeddings.embedding_service import EmbeddingService


def _make_chunks(n: int) -> list[Document]:
    return [Document(page_content=f"chunk {i}", metadata={}) for i in range(n)]


class TestEmbeddingService:
    def setup_method(self):
        self.mock_provider = MagicMock()
        self.service = EmbeddingService(provider=self.mock_provider)

    def test_embed_calls_provider_with_texts(self):
        chunks = _make_chunks(3)
        self.mock_provider.embed_documents.return_value = [[0.1] * 5] * 3

        self.service.embed(chunks)

        self.mock_provider.embed_documents.assert_called_once_with(
            ["chunk 0", "chunk 1", "chunk 2"]
        )

    def test_embed_returns_correct_count(self):
        n = 5
        chunks = _make_chunks(n)
        self.mock_provider.embed_documents.return_value = [[0.0] * 10] * n

        result = self.service.embed(chunks)

        assert len(result) == n

    def test_embed_empty_returns_empty(self):
        result = self.service.embed([])
        self.mock_provider.embed_documents.assert_not_called()
        assert result == []

    def test_embed_vectors_match_provider_output(self):
        chunks = _make_chunks(2)
        expected = [[1.0, 2.0], [3.0, 4.0]]
        self.mock_provider.embed_documents.return_value = expected

        result = self.service.embed(chunks)

        assert result == expected
