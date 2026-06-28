"""
connector.py

Git source connector for the Atlas Knowledge Platform.

Responsible for cloning and updating Git repositories to the local filesystem.
All file discovery and parsing is delegated to GitLoader and DocumentParser.
"""

import shutil
from pathlib import Path
from urllib.parse import urlparse

import git

from app.core.config import settings
from app.core.exceptions import ConnectorException
from app.core.logging import get_logger
from app.infrastructure.connectors.base import BaseConnector

logger = get_logger(__name__)


class GitConnector(BaseConnector):
    """
    Clones and updates Git repositories using GitPython.

    Responsibilities:
    - Clone a remote Git repository to local storage.
    - Pull updates for an already-cloned repository.
    - Delete a local clone from the filesystem.

    Non-responsibilities:
    - Walking files (GitLoader).
    - Parsing files into KnowledgeDocument (DocumentParser).
    """

    @property
    def repository_path(self) -> Path:
        """Base directory for all cloned repositories, from settings."""
        return settings.repository_path

    def _derive_repo_name(self, url: str) -> str:
        """
        Derives a filesystem-safe directory name from a repository URL.

        Example:
            https://github.com/org/atlas.git  →  atlas
            git@github.com:org/atlas.git      →  atlas
        """
        path = urlparse(url).path or url
        name = path.rstrip("/").split("/")[-1]
        return name.removesuffix(".git") or "repository"

    def _local_path(self, url: str) -> Path:
        """Resolves the absolute local path for a given repository URL."""
        return self.repository_path / self._derive_repo_name(url)

    def clone(self, url: str) -> Path:
        """
        Clone the repository at *url* or pull it if it already exists.

        Returns:
            Absolute path to the local clone directory.

        Raises:
            ConnectorException: If the clone or pull operation fails.
        """
        local_path = self._local_path(url)

        if local_path.exists():
            logger.info(
                "Repository already exists locally, pulling latest: %s",
                local_path,
            )
            self.pull(local_path)
            return local_path

        logger.info("Cloning repository: %s → %s", url, local_path)

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            git.Repo.clone_from(url, str(local_path))
        except git.GitCommandError as exc:
            raise ConnectorException(
                f"Failed to clone repository '{url}': {exc}"
            ) from exc

        logger.info(
            "Repository cloned successfully: %s (%s)",
            self._derive_repo_name(url),
            local_path,
        )

        return local_path

    def pull(self, repo_path: Path) -> None:
        """
        Pull the latest changes for an existing local clone.

        Args:
            repo_path: Absolute path to the local Git clone.

        Raises:
            ConnectorException: If the pull operation fails.
        """
        logger.info("Pulling latest changes: %s", repo_path)

        try:
            repo = git.Repo(str(repo_path))
            origin = repo.remotes.origin
            origin.pull()
        except git.InvalidGitRepositoryError as exc:
            raise ConnectorException(
                f"Not a valid Git repository: '{repo_path}'"
            ) from exc
        except git.GitCommandError as exc:
            raise ConnectorException(
                f"Failed to pull repository at '{repo_path}': {exc}"
            ) from exc

        logger.info("Repository updated: %s", repo_path)

    def delete(self, repo_path: Path) -> None:
        """
        Remove a local clone from the filesystem.

        Args:
            repo_path: Absolute path to the local clone directory.

        Raises:
            ConnectorException: If the directory cannot be removed.
        """
        if not repo_path.exists():
            logger.warning(
                "Attempted to delete non-existent repository path: %s",
                repo_path,
            )
            return

        logger.info("Deleting local repository: %s", repo_path)

        try:
            shutil.rmtree(repo_path)
        except OSError as exc:
            raise ConnectorException(
                f"Failed to delete repository at '{repo_path}': {exc}"
            ) from exc

        logger.info("Repository deleted: %s", repo_path)
