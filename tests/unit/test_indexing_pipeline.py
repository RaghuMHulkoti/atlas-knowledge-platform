"""
test_indexing_pipeline.py

Unit tests for IndexingPipeline.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.ai.pipelines.indexing_pipeline import IndexingPipeline
from app.core.exceptions import IndexingException
from app.domain.knowledge.models import KnowledgeDocument


def _make_doc(id_str: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        id=id_str,
        source="http://test",
        repository="test",
        path="test",
        title="test",
        content="test content",
        language="txt",
        metadata={},
    )


class TestIndexingPipeline:
    def setup_method(self):
        self.mock_converter = MagicMock()
        self.mock_splitter = MagicMock()
        self.mock_embedder = MagicMock()
        self.mock_store = MagicMock()

        self.pipeline = IndexingPipeline(
            converter=self.mock_converter,
            splitter=self.mock_splitter,
            embedding_service=self.mock_embedder,
            vector_store=self.mock_store,
            collection_name="test_col",
        )

    @pytest.mark.anyio
    async def test_run_orchestrates_stages_correctly(self):
        # 1. Input
        docs = [_make_doc("d1")]
        # 2. Convert
        lc_docs = [Document(page_content="test", metadata={"id": "d1"})]
        self.mock_converter.convert_many.return_value = lc_docs
        # 3. Split
        chunks = [
            Document(page_content="t1", metadata={"id": "d1"}),
            Document(page_content="t2", metadata={"id": "d1"}),
        ]
        self.mock_splitter.split.return_value = chunks
        # 4. Embed
        self.mock_embedder.embed.return_value = [[0.1], [0.2]]

        # Execute
        count = await self.pipeline.run(docs)

        # Verify orchestrator flow
        assert count == 2
        self.mock_converter.convert_many.assert_called_once_with(docs)
        self.mock_splitter.split.assert_called_once_with(lc_docs)
        self.mock_embedder.embed.assert_called_once_with(chunks)

        # Verify vector store upsert
        self.mock_store.upsert.assert_called_once()
        kwargs = self.mock_store.upsert.call_args.kwargs
        assert kwargs["collection_name"] == "test_col"
        assert len(kwargs["ids"]) == 2
        assert kwargs["documents"] == ["t1", "t2"]
        assert kwargs["embeddings"] == [[0.1], [0.2]]
        assert len(kwargs["metadatas"]) == 2

    @pytest.mark.anyio
    async def test_run_empty_list_returns_zero(self):
        count = await self.pipeline.run([])
        assert count == 0
        self.mock_converter.convert_many.assert_not_called()
        self.mock_store.upsert.assert_not_called()

    @pytest.mark.anyio
    async def test_run_raises_indexing_exception(self):
        self.mock_converter.convert_many.side_effect = ValueError("boom")

        with pytest.raises(IndexingException, match="IndexingPipeline failed: boom"):
            await self.pipeline.run([_make_doc("d1")])

    def test_prepare_upsert_payload_generates_unique_ids(self):
        chunks = [
            Document(page_content="p1", metadata={"id": "parent1"}),
            Document(page_content="p2", metadata={"id": "parent1"}),
        ]
        ids, texts, metas = IndexingPipeline._prepare_upsert_payload(chunks)

        assert len(ids) == 2
        assert ids[0] != ids[1]
        assert ids[0].startswith("parent1_")
        assert ids[1].startswith("parent1_")

    def test_prepare_upsert_payload_ids_are_deterministic(self):
        """Re-chunking the same document yields the same chunk ids (idempotent)."""
        chunks = [
            Document(page_content="p1", metadata={"id": "parent1"}),
            Document(page_content="p2", metadata={"id": "parent1"}),
            Document(page_content="q1", metadata={"id": "parent2"}),
        ]
        ids_a, _, _ = IndexingPipeline._prepare_upsert_payload(chunks)
        ids_b, _, _ = IndexingPipeline._prepare_upsert_payload(chunks)

        assert ids_a == ids_b == ["parent1_0", "parent1_1", "parent2_0"]

    def test_prepare_upsert_payload_cleans_metadata(self):
        chunks = [
            Document(
                page_content="p1",
                metadata={"valid": "str", "bad": None, "list": [1, 2]},
            )
        ]
        _, _, metas = IndexingPipeline._prepare_upsert_payload(chunks)

        assert "valid" in metas[0]
        assert "bad" not in metas[0]
        assert "list" not in metas[0]
