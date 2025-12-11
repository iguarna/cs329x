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
    
class HuggingFaceClient(BaseLLMClient):
    """HuggingFace Transformers client implementation."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.8,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        token: Optional[str] = None,
    ):
        """
        Initialize the HuggingFace client.

        Args:
            model: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-chat-hf")
            temperature: Sampling temperature
            device: Device to load model on ("cuda", "cpu", or None for auto)
            torch_dtype: Torch dtype for model weights ("auto", "float16", "bfloat16")
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
            token: Optional HuggingFace API token for accessing gated models
        """
        super().__init__(model, temperature)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise RuntimeError(
                "transformers and torch are not installed. Install with: "
                "pip install transformers torch accelerate"
            )

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Configure dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype_val = dtype_map.get(torch_dtype, "auto")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional quantization
        model_kwargs = {
            "torch_dtype": torch_dtype_val,
            "device_map": "auto" if device == "cuda" else None,
            "token": token,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self.model_obj = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)

        if not load_in_8bit and not load_in_4bit and device == "cpu":
            self.model_obj = self.model_obj.to(device)

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_output_tokens: int = 800,
    ) -> str:
        """Generate a completion using HuggingFace model."""
        import torch

        # Format prompt with system message if provided
        if system_prompt:
            # Try to use chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": prompt.strip()},
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback to simple concatenation
                formatted_prompt = f"{system_prompt.strip()}\n\n{prompt.strip()}"
        else:
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt.strip()}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt.strip()

        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model_obj.generate(
                **inputs,
                max_new_tokens=max_output_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output (skip the input tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()