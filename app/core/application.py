import asyncio
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


def _warm_up() -> None:
    """
    Load heavy singletons before the app serves traffic.

    Most importantly this downloads/loads the on-device embedding model, so the
    first ingest request does not pay the one-time ~80 MB download cost (which
    otherwise makes that request slow enough to hit a gateway 502). Runs at
    startup; failures are logged but do not prevent the app from starting so the
    health endpoint stays reachable.
    """
    from app.core.dependencies import get_embedding_provider, get_vector_store

    try:
        logger.info("Warm-up: loading embedding model (one-time download)...")
        provider = get_embedding_provider()
        # A real embed forces the model to download and initialise now.
        provider.embed_query("warmup")
        logger.info("Warm-up: embedding model ready.")
    except Exception:
        logger.exception("Warm-up: embedding model failed to load.")

    try:
        get_vector_store()
        logger.info("Warm-up: vector store client ready.")
    except Exception:
        logger.exception("Warm-up: vector store initialisation failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Atlas starting (env=%s).", settings.ENVIRONMENT)
    # Run the blocking warm-up off the event loop so startup logging still flows.
    await asyncio.to_thread(_warm_up)
    logger.info("Atlas warm-up complete — ready to serve.")
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
