from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import AtlasException, atlas_exception_handler
from app.core.logging import configure_logging, get_logger
from app.middleware.exception import unhandled_exception_handler
from app.middleware.logging import AccessLogMiddleware
from app.middleware.request_id import RequestIDMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Atlas starting (env=%s).", settings.ENVIRONMENT)
    yield
    logger.info("Atlas shutting down.")


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # Middleware is applied bottom-up: RequestIDMiddleware runs first so the
    # request id is available to the access logger.
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    app.add_exception_handler(AtlasException, atlas_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    return app
