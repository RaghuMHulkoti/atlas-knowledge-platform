"""Unit tests for EmbeddingFactory provider selection (offline)."""

import pytest

from app.ai.embeddings import embedding_factory as ef
from app.core.exceptions import IndexingException


def test_unknown_provider_raises():
    with pytest.raises(IndexingException, match="Unknown EMBEDDING_PROVIDER"):
        ef.EmbeddingFactory.create("does-not-exist")


def test_google_without_key_raises(monkeypatch):
    monkeypatch.setattr(ef.settings, "GOOGLE_API_KEY", None)
    with pytest.raises(IndexingException, match="requires GOOGLE_API_KEY"):
        ef.EmbeddingFactory.create("google")


def test_google_provider_is_selected(monkeypatch):
    """create('google') builds a GoogleEmbeddingProvider without hitting network."""
    created = {}

    class _FakeGoogle:
        def __init__(self):
            created["yes"] = True

    # A key must be present for the google branch to proceed.
    monkeypatch.setattr(ef.settings, "GOOGLE_API_KEY", "test-key")

    # Patch the lazily-imported symbol at its source module.
    import app.ai.embeddings.provider as gp

    monkeypatch.setattr(gp, "GoogleEmbeddingProvider", _FakeGoogle)

    provider = ef.EmbeddingFactory.create("google")
    assert created.get("yes") is True
    assert isinstance(provider, _FakeGoogle)
