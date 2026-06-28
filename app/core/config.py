from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration.

    Configuration values are loaded in the following order:

    1. Environment Variables
    2. .env file
    3. Default values
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    APP_NAME: str = "Atlas"

    APP_VERSION: str = "0.1.0"

    APP_DESCRIPTION: str = "Enterprise Engineering Knowledge Platform"

    ENVIRONMENT: str = Field(default="development")

    DEBUG: bool = Field(default=True)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    API_V1_PREFIX: str = "/api/v1"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    LLM_PROVIDER: str = "openrouter"

    OPENROUTER_API_KEY: SecretStr

    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    LLM_PRIMARY_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"

    LLM_FALLBACK_MODELS: str = "openai/gpt-oss-120b:free"

    @property
    def fallback_models(self) -> list[str]:
        """
        Returns the configured fallback models.

        Example:
            openai/gpt-oss-120b:free,
            deepseek/deepseek-r1:free

        becomes

            [
                "openai/gpt-oss-120b:free",
                "deepseek/deepseek-r1:free",
            ]
        """

        if not self.LLM_FALLBACK_MODELS:
            return []

        return [
            model.strip()
            for model in self.LLM_FALLBACK_MODELS.split(",")
            if model.strip()
        ]

    # ------------------------------------------------------------------
    # Chroma Cloud
    # ------------------------------------------------------------------

    CHROMA_API_KEY: SecretStr

    CHROMA_TENANT: str

    CHROMA_DATABASE: str

    CHROMA_PATH: str = "./storage/chroma"

    DEFAULT_COLLECTION: str = "atlas"

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    UPLOAD_DIRECTORY: str = "./storage/uploads"

    REPOSITORY_DIRECTORY: str = "./storage/repositories"

    MAX_UPLOAD_SIZE_MB: int = 100

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    CHUNK_SIZE: int = 1000

    CHUNK_OVERLAP: int = 200

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # ------------------------------------------------------------------
    # Project Paths
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def upload_path(self) -> Path:
        return self.project_root / "storage" / "uploads"

    @property
    def repository_path(self) -> Path:
        return self.project_root / "storage" / "repositories"

    @property
    def chroma_path(self) -> Path:
        return self.project_root / "storage" / "chroma"


@lru_cache
def get_settings() -> Settings:
    """
    Returns a singleton Settings instance.

    The configuration is loaded only once during the
    application's lifetime.
    """
    return Settings()


settings = get_settings()
