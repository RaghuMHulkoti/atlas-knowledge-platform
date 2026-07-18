"""
local_provider.py

On-device embedding provider — free, unlimited, no API key.

Uses ChromaDB's bundled ONNX model (all-MiniLM-L6-v2, 384-dim) via onnxruntime,
which ships with chromadb (no torch). The model (~80 MB) is fetched once to a
local cache on first use.

CPU threads are pinned to ``EMBEDDING_NUM_THREADS``. This matters enormously in
containers: chromadb builds the onnxruntime session with default options, so
onnxruntime otherwise spawns one thread per HOST core — which it cannot align
with the pod's CPU limit — and those threads thrash, making embedding ~10-20x
slower. We patch the session creation to pin the intra-op thread count.
"""

from chromadb.utils import embedding_functions

from app.ai.embeddings.base import BaseEmbeddingProvider
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _build_pinned_embedding_function(threads: int):
    """
    Build ChromaDB's default embedding function with its onnxruntime session
    pinned to *threads* CPU threads.

    onnxruntime's thread count is fixed when the InferenceSession is created, and
    chromadb does not expose it. We temporarily wrap ``InferenceSession`` to set
    ``intra_op_num_threads`` on the session options, then force the (lazy)
    session to build now so the pin takes effect. Falls back to the unpinned
    default if anything goes wrong.
    """
    ef = embedding_functions.DefaultEmbeddingFunction()

    try:
        import onnxruntime as ort

        original = ort.InferenceSession

        def pinned(*args, **kwargs):
            options = kwargs.get("sess_options")
            if options is not None:
                options.intra_op_num_threads = threads
                options.inter_op_num_threads = 1
            return original(*args, **kwargs)

        ort.InferenceSession = pinned
        try:
            # A dummy embed triggers the cached-property session build under the
            # patch, so the thread pin is baked into the cached session.
            ef(["warmup"])
        finally:
            ort.InferenceSession = original
    except Exception:
        logger.warning(
            "Could not pin onnxruntime threads; using default thread pool.",
            exc_info=True,
        )

    return ef


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by an on-device ONNX MiniLM model.

    The same model embeds documents and queries, so write-time and query-time
    vectors are always dimensionally consistent (384-dim).
    """

    def __init__(self) -> None:
        threads = max(1, settings.EMBEDDING_NUM_THREADS)
        self._embedding_function = _build_pinned_embedding_function(threads)
        logger.info(
            "LocalEmbeddingProvider initialised (all-MiniLM-L6-v2, 384-dim, "
            "%d thread(s)).",
            threads,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document texts on-device.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of dense float vectors in the same order as input.
        """
        if not texts:
            return []

        # ChromaDB embedding functions return numpy arrays; coerce to plain
        # Python lists so the vectors are JSON-serialisable end to end.
        vectors = self._embedding_function(texts)
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string on-device.

        Args:
            text: Query string.

        Returns:
            A single dense float vector.
        """
        return self.embed_documents([text])[0]
