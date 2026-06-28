from abc import ABC, abstractmethod
from collections.abc import Sequence

from langchain_core.messages import BaseMessage


class BaseLLM(ABC):
    """
    Contract for all LLM providers.

    The rest of Atlas depends on this interface instead of
    provider-specific SDKs.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """
        Generate a response from a plain text prompt.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_messages(
        self,
        messages: Sequence[BaseMessage],
        **kwargs,
    ) -> str:
        """
        Generate a response from a conversation.
        """
        raise NotImplementedError

    @abstractmethod
    def count_tokens(
        self,
        text: str,
    ) -> int:
        """
        Return the number of tokens for the given text.
        """
        raise NotImplementedError

    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the provider name.

        Example:
        - openrouter
        - openai
        - gemini
        - ollama
        """
        raise NotImplementedError

    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Checks whether the provider is available.
        """
        raise NotImplementedError
