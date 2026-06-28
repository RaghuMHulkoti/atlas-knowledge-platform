from app.core.config import settings


class ModelManager:
    @classmethod
    def get_models(cls) -> list[str]:
        """
        Returns the primary model followed by any configured fallback models.
        """
        return [settings.LLM_PRIMARY_MODEL] + settings.fallback_models
