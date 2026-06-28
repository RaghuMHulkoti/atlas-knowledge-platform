"""
base.py

Abstract contract for all source connectors in the Atlas platform.

A connector is responsible for acquiring source content from a remote origin
(Git repository, S3 bucket, Confluence space, etc.) and making it available
as files on the local filesystem. It has no knowledge of parsing or documents.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseConnector(ABC):
    """
    Contract that every source connector must implement.

    Responsibilities:
    - Acquire source content from a remote origin.
    - Keep the local clone up to date.
    - Clean up local storage when requested.

    Non-responsibilities:
    - Parsing files into domain objects.
    - Walking the file tree.
    - Any knowledge of KnowledgeDocument.
    """

    @property
    @abstractmethod
    def repository_path(self) -> Path:
        """
        Base directory under which all clones are stored.

        Example: /project_root/storage/repositories
        """
        raise NotImplementedError

    @abstractmethod
    def clone(self, url: str) -> Path:
        """
        Clone the remote source to a local directory.

        Returns the absolute path to the cloned directory.
        Must be idempotent: if the clone already exists, update it instead.
        """
        raise NotImplementedError

    @abstractmethod
    def pull(self, repo_path: Path) -> None:
        """
        Update an existing local clone from its remote origin.

        Args:
            repo_path: Absolute path to the local clone directory.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, repo_path: Path) -> None:
        """
        Remove a local clone from the filesystem.

        Args:
            repo_path: Absolute path to the local clone directory.
        """
        raise NotImplementedError
