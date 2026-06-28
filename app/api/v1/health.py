from fastapi import APIRouter

from app.core.config import settings
from app.core.dependencies import get_llm, get_vector_store
from app.core.constants import HealthStatus

router = APIRouter()


@router.get("/")
async def health():

    llm = get_llm()
    vector_store = get_vector_store()

    chroma_status = HealthStatus.UP if vector_store.heartbeat() else HealthStatus.DOWN

    llm_status = HealthStatus.UP if await llm.health_check() else HealthStatus.DOWN

    overall = (
        HealthStatus.UP
        if (chroma_status == HealthStatus.UP and llm_status == HealthStatus.UP)
        else HealthStatus.DOWN
    )

    return {
        "status": overall,
        "application": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "services": {
            "chromadb": chroma_status,
            "llm": llm_status,
        },
    }
