from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseRetriever(ABC):
    """
    Abstract interface for retrievers.
    """

    @abstractmethod
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """Asynchronously retrieve relevant documents for a query."""
        pass
