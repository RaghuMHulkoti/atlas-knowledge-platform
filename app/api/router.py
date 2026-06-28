from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.collections import router as collections_router
from app.api.v1.knowledge import router as knowledge_router

api_router = APIRouter()

api_router.include_router(
    health_router,
    prefix="/health",
    tags=["Health"],
)

api_router.include_router(
    collections_router,
    prefix="/collections",
    tags=["Collections"],
)

api_router.include_router(
    knowledge_router,
    prefix="/knowledge",
    tags=["Knowledge"],
)
