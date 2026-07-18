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

# Cap the startup LLM reachability check so a rate-limited free model's long
# Retry-After can never stall startup.
_LLM_WARMUP_TIMEOUT_SECONDS = 10.0


async def _warm_up() -> None:
    """
    Establish every external connection before the app serves traffic.

    Warms three things so the first request pays none of the cost and any
    misconfiguration surfaces at startup rather than mid-request:

      1. Embedding model — loads the on-device ONNX model (one-time ~80 MB
         download), avoiding a slow first ingest.
      2. ChromaDB — opens the client, checks the heartbeat, and ensures the
         default collection exists.
      3. LLM (OpenRouter) — initialises the model clients and verifies the
         provider is reachable.

    Every step is best-effort: failures are logged but never prevent startup, so
    the health endpoints stay reachable and the operator can see what is wrong.
    """
    from app.core.dependencies import (
        get_embedding_provider,
        get_llm,
        get_vector_store,
    )

    # 1. Embedding model (blocking CPU + one-time download) — off the loop.
    try:
        logger.info("Warm-up: loading embedding model...")
        provider = get_embedding_provider()
        await asyncio.to_thread(provider.embed_query, "warmup")
        logger.info("Warm-up: embedding model ready.")
    except Exception:
        logger.exception("Warm-up: embedding model failed to load.")

    # 2. ChromaDB — connect, heartbeat, ensure the default collection exists.
    try:
        logger.info("Warm-up: connecting to ChromaDB...")
        store = get_vector_store()
        alive = await asyncio.to_thread(store.heartbeat)
        if alive:
            await asyncio.to_thread(
                store.get_or_create_collection, settings.DEFAULT_COLLECTION
            )
            logger.info(
                "Warm-up: ChromaDB connected; collection '%s' ready.",
                settings.DEFAULT_COLLECTION,
            )
        else:
            logger.warning("Warm-up: ChromaDB heartbeat failed — check credentials.")
    except Exception:
        logger.exception("Warm-up: ChromaDB connection failed.")

    # 3. LLM (OpenRouter) — initialise clients and verify reachability.
    #    Bounded by a timeout: a rate-limited free model can return a long
    #    Retry-After (tens of seconds), and we must not let that stall startup.
    try:
        logger.info("Warm-up: connecting to LLM (%s)...", settings.LLM_PRIMARY_MODEL)
        llm = get_llm()
        reachable = await asyncio.wait_for(
            llm.health_check(), timeout=_LLM_WARMUP_TIMEOUT_SECONDS
        )
        if reachable:
            logger.info("Warm-up: LLM reachable (%s).", settings.LLM_PRIMARY_MODEL)
        else:
            logger.warning(
                "Warm-up: LLM health check failed (rate limit or bad key). "
                "Clients are initialised; the app is still starting."
            )
    except asyncio.TimeoutError:
        logger.warning(
            "Warm-up: LLM health check timed out after %.0fs (likely rate "
            "limited). Clients are initialised; the app is still starting.",
            _LLM_WARMUP_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("Warm-up: LLM connection failed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Atlas starting (env=%s).", settings.ENVIRONMENT)
    await _warm_up()
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
