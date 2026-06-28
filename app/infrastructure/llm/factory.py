from langchain_openai import ChatOpenAI

from app.core.config import settings

OPENROUTER_MODELS = {
    "llama": "meta-llama/llama-3.3-70b-instruct:free",
    "gpt-oss": "openai/gpt-oss-120b:free",
}


class LLMFactory:

    @classmethod
    def create(cls, model_name: str) -> ChatOpenAI:
        model = OPENROUTER_MODELS.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")

        return ChatOpenAI(
            model=model,
            api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
            base_url=settings.OPENROUTER_BASE_URL,
            temperature=0,
        )
