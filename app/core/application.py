from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.logging import configure_logging, get_logger


from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import (
    AtlasException,
    atlas_exception_handler,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Logging configured successfully.")

    print("🚀 Atlas is starting...")

    yield

    print("🛑 Atlas is shutting down...")


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    app.include_router(
        api_router,
        prefix=settings.API_V1_PREFIX,
    )

    app.add_exception_handler(
        AtlasException,
        atlas_exception_handler,
    )

    return app
