"""
Application-wide constants.

Avoid magic strings throughout the codebase.
"""

from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class KnowledgeSource(str, Enum):
    GIT = "git"
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TEXT = "text"


class SupportedFileType(str, Enum):
    PDF = ".pdf"
    DOCX = ".docx"
    MD = ".md"
    TXT = ".txt"


class CollectionName:
    DEFAULT = "atlas"


class API:
    V1 = "/api/v1"


class HealthStatus:
    UP = "UP"
    DOWN = "DOWN"
