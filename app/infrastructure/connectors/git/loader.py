"""
loader.py

File discovery for cloned Git repositories.

Recursively walks a local repository directory, applies ignore rules, and
returns only files with supported extensions. No file content is read here —
that is the responsibility of DocumentParser.
"""

from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)

# Directories to exclude from traversal regardless of nesting depth.
_IGNORED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".idea",
        ".vscode",
        "node_modules",
        "target",
        "build",
        "dist",
        "__pycache__",
        "venv",
        ".venv",
    }
)

# File extensions that the ingestion pipeline can process.
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".java",
        ".kt",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".sql",
        ".properties",
        ".gradle",
    }
)


class GitLoader:
    """
    Discovers all ingestible files inside a locally cloned repository.

    Responsibilities:
    - Recursively walk the repository directory.
    - Skip ignored directories.
    - Filter to supported file extensions only.
    - Return a list of absolute Paths.

    Non-responsibilities:
    - Reading or parsing file content.
    - Constructing KnowledgeDocument objects.
    """

    def load(self, repo_path: Path) -> list[Path]:
        """
        Walk *repo_path* and return all files eligible for ingestion.

        Args:
            repo_path: Absolute path to the root of the local clone.

        Returns:
            Sorted list of absolute Paths to ingestible files.
        """
        discovered: list[Path] = []

        for item in self._walk(repo_path):
            discovered.append(item)

        logger.info(
            "GitLoader discovered %d file(s) in '%s'",
            len(discovered),
            repo_path,
        )

        return discovered

    def _walk(self, directory: Path):
        """
        Recursively yield files, skipping ignored directories and
        unsupported extensions.
        """
        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            logger.warning("Permission denied, skipping directory: %s", directory)
            return

        for entry in entries:
            if entry.is_dir():
                if entry.name in _IGNORED_DIRS:
                    logger.debug("Skipping ignored directory: %s", entry)
                    continue
                yield from self._walk(entry)

            elif entry.is_file():
                if entry.suffix.lower() in _SUPPORTED_EXTENSIONS:
                    yield entry
                else:
                    logger.debug("Skipping unsupported file type: %s", entry.name)
