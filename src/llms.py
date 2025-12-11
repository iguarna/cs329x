"""Abstract base class and implementations for LLM clients."""

from abc import ABC, abstractmethod
from typing import Callable, Optional


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_output_tokens: int = 800,
    ) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt to set context
            max_output_tokens: Maximum number of tokens to generate

        Returns:
            The generated text completion
        """
        raise NotImplementedError()


class OpenAiClient(BaseLLMClient):
    """OpenAI API client implementation."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the OpenAI LLM client.

        Args:
            model: OpenAI model name (e.g., "gpt-4o-mini", "gpt-4o")
            temperature: Sampling temperature
            api_key: Optional API key (uses OPENAI_API_KEY env var if not provided)
        """
        super().__init__()

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package is not installed. Install with: pip install openai"
            )

        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_output_tokens: int = 800,
    ) -> str:
        """Generate a completion using OpenAI API.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt to set context
            max_output_tokens: Maximum number of tokens to generate

        Returns:
            The generated text completion
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt.strip()})

        if 'gpt-4' in self._model:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=max_output_tokens,
            )
        elif 'gpt-5' in self._model:
            # GPT-5 does't support temperature parameter
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_completion_tokens=max_output_tokens,
            )
        else:
            raise ValueError(f'Unsupported model {self._model}.')
        
        return response.choices[0].message.content.strip()


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without API calls."""

    def __init__(
        self,
        responder: Optional[Callable[..., str]] = None,
    ):
        """
        Initialize the Mock LLM client.

        Args:
            responder: Custom responder function that takes (prompt, system_prompt, max_output_tokens) and returns str
        """
        super().__init__()
        self._responder = responder or self._default_responder

    def _default_responder(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_output_tokens: int = 800,
    ) -> str:
        """Default mock responder that returns a simple placeholder."""
        del prompt, system_prompt, max_output_tokens
        return "This is a mock response for testing."

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_output_tokens: int = 800,
    ) -> str:
        """Generate a mock completion using the custom responder."""
        return self._responder(
            prompt=prompt,
            system_prompt=system_prompt,
            max_output_tokens=max_output_tokens,
        )