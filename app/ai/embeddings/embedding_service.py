from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings


class GoogleEmbeddingsService:
    """
    Service wrapper for Google Generative AI Embeddings.
    """

    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
        )

    def get_embeddings(self) -> Embeddings:
        """Return the LangChain Embeddings interface."""
        return self.embeddings
