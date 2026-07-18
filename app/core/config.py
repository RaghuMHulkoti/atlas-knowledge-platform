import os
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
    # Git authentication (optional — required only for PRIVATE repositories)
    # ------------------------------------------------------------------

    # Personal access token used to clone private HTTPS repositories.
    # For GitHub, create a fine-grained or classic PAT with 'repo' (read) scope.
    GIT_TOKEN: SecretStr | None = None

    # Optional username to pair with the token. Not needed for GitHub PATs
    # (the token alone authenticates), but some hosts require it.
    GIT_USERNAME: str | None = None

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    CHUNK_SIZE: int = 1000

    CHUNK_OVERLAP: int = 200

    # ------------------------------------------------------------------
    # Embeddings — on-device ONNX MiniLM (384-dim). Free, no key, no quota.
    # ------------------------------------------------------------------

    EMBEDDING_PROVIDER: str = "local"

    # Chunks are embedded in batches of this size. Smaller batches sharply lower
    # peak memory (embedding a whole repo at once can spike memory enough to be
    # OOM-killed in a memory-limited container). Raise only if you have headroom.
    EMBEDDING_BATCH_SIZE: int = 16

    # Number of CPU threads the embedding model may use (onnxruntime + BLAS).
    # CRITICAL in containers: by default onnxruntime spawns one thread per HOST
    # core (it cannot see the pod's CPU limit), and those threads thrash against
    # a throttled CPU allocation — making embedding ~10-20x slower. Pin this to
    # the pod's CPU-limit cores (e.g. 2 for `limits.cpu: "2"`).
    EMBEDDING_NUM_THREADS: int = 4

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    # Default number of chunks returned by semantic retrieval.
    RETRIEVAL_TOP_K: int = 5

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


def _pin_math_library_threads() -> None:
    """
    Cap the CPU threads NumPy/BLAS spawn, to the configured embedding thread
    count. Like onnxruntime, OpenBLAS/MKL otherwise default to one thread per
    HOST core and thrash under a container CPU limit. Must run before NumPy is
    imported; config is imported early enough for that. ``setdefault`` lets an
    explicit environment value win.
    """
    threads = str(max(1, settings.EMBEDDING_NUM_THREADS))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, threads)


_pin_math_library_threads()
