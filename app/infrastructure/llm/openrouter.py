from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.core.exceptions import LLMException
from app.core.logging import get_logger
from app.infrastructure.llm.base import BaseLLM
from app.infrastructure.llm.model_manager import ModelManager

logger = get_logger(__name__)


class OpenRouterLLM(BaseLLM):
    """
    OpenRouter implementation of the BaseLLM contract.

    Responsibilities:
    - Manage ChatOpenAI instances
    - Execute prompts
    - Handle model failover
    - Count tokens

    This class should not contain prompt engineering,
    retrieval logic or business logic.
    """

    def __init__(self) -> None:
        self._models: dict[str, ChatOpenAI] = {}

    def _get_model(self, model_name: str) -> ChatOpenAI:
        """
        Lazily creates and caches ChatOpenAI instances.
        """

        if model_name not in self._models:

            logger.info(
                "Initializing OpenRouter model: %s",
                model_name,
            )

            self._models[model_name] = ChatOpenAI(
                model=model_name,
                api_key=settings.OPENROUTER_API_KEY.get_secret_value(),
                base_url=settings.OPENROUTER_BASE_URL,
                temperature=0,
            )

        return self._models[model_name]

    async def _invoke(
        self,
        messages: Sequence[BaseMessage],
        **kwargs,
    ) -> BaseMessage:
        """
        Tries the primary model followed by all configured
        fallback models.
        """

        last_exception: Exception | None = None

        for model_name in ModelManager.get_models():

            try:

                logger.info(
                    "Using model: %s",
                    model_name,
                )

                llm = self._get_model(model_name)

                response = await llm.ainvoke(
                    messages,
                    **kwargs,
                )

                return response

            except Exception as ex:

                logger.warning(
                    "Model %s failed. Trying next fallback.",
                    model_name,
                )

                last_exception = ex

        raise LLMException(
            "All configured OpenRouter models failed."
        ) from last_exception

    async def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """
        Generates a response from plain text.
        """

        response = await self._invoke(
            [HumanMessage(content=prompt)],
            **kwargs,
        )

        return str(response.content)

    async def generate_messages(
        self,
        messages: Sequence[BaseMessage],
        **kwargs,
    ) -> str:
        """
        Generates a response from a conversation.
        """

        response = await self._invoke(
            messages,
            **kwargs,
        )

        return str(response.content)

    def count_tokens(
        self,
        text: str,
    ) -> int:
        """
        Counts the number of tokens for the configured
        primary model.
        """

        llm = self._get_model(
            settings.LLM_PRIMARY_MODEL,
        )

        return llm.get_num_tokens(text)

    def provider_name(self) -> str:
        return "openrouter"

    def model_name(self) -> str:
        return settings.LLM_PRIMARY_MODEL

    async def health_check(self) -> bool:
        """Performs a lightweight request to verify that the configured provider is reachable."""

        try:
            await self.generate(
                "Reply with exactly: OK",
                max_tokens=2,
            )
            return True

        except Exception as ex:
            logger.exception(
                "OpenRouter health check failed.",
                exc_info=ex,
            )
            return False
