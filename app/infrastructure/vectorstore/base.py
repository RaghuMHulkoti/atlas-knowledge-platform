from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """
    Contract for every vector store implementation.
    """

    @abstractmethod
    def get_or_create_collection(self, name: str) -> Any:
        pass

    @abstractmethod
    def get_collection(self, name: str) -> Any:
        pass

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        pass

    @abstractmethod
    def list_collections(self) -> list[Any]:
        pass

    @abstractmethod
    def heartbeat(self) -> bool:
        pass
