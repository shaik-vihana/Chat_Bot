"""
Robust LLM Provider Abstraction
Supports multiple backends: Ollama, OpenAI, Groq, Anthropic
With automatic fallback, retries, and streaming support
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Generator, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    GROQ = "groq"
    ANTHROPIC = "anthropic"


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: ProviderType = ProviderType.OLLAMA
    model_name: str = "mistral"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    streaming: bool = True

    # Provider-specific settings
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._is_available = None

    @abstractmethod
    def invoke(self, prompt: str, callbacks: Optional[List] = None) -> str:
        """Invoke the LLM with a prompt"""
        pass

    @abstractmethod
    def stream(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        """Stream response tokens"""
        pass

    @abstractmethod
    def check_availability(self) -> bool:
        """Check if the provider is available"""
        pass

    @property
    def is_available(self) -> bool:
        if self._is_available is None:
            self._is_available = self.check_availability()
        return self._is_available

    def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry"""
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)

        raise last_exception


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM Provider"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from langchain_ollama import ChatOllama
                self._client = ChatOllama(
                    model=self.config.model_name,
                    base_url=self.config.base_url,
                    temperature=self.config.temperature,
                    num_predict=self.config.max_tokens,
                    streaming=self.config.streaming,
                )
            except ImportError:
                raise ImportError("langchain-ollama is required for Ollama provider")
        return self._client

    def check_availability(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return self.config.model_name in model_names or any(
                    self.config.model_name in name for name in model_names
                )
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
        return False

    def invoke(self, prompt: str, callbacks: Optional[List] = None) -> str:
        def _invoke():
            client = self._get_client()
            response = client.invoke(prompt, config={"callbacks": callbacks} if callbacks else None)
            return response.content if hasattr(response, "content") else str(response)

        return self._retry_with_backoff(_invoke)

    def stream(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        client = self._get_client()
        for chunk in client.stream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if callback:
                callback(token)
            yield token


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not provided")
                self._client = OpenAI(api_key=api_key, timeout=self.config.timeout)
            except ImportError:
                raise ImportError("openai package is required for OpenAI provider")
        return self._client

    def check_availability(self) -> bool:
        try:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            return bool(api_key)
        except Exception:
            return False

    def invoke(self, prompt: str, callbacks: Optional[List] = None) -> str:
        def _invoke():
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            result = response.choices[0].message.content
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_llm_end"):
                        cb.on_llm_end(response)
            return result

        return self._retry_with_backoff(_invoke)

    def stream(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        client = self._get_client()
        stream = client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                if callback:
                    callback(token)
                yield token


class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider - Ultra-fast inference"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("Groq API key not provided")
                self._client = Groq(api_key=api_key)
            except ImportError:
                raise ImportError("groq package is required for Groq provider")
        return self._client

    def check_availability(self) -> bool:
        try:
            api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
            return bool(api_key)
        except Exception:
            return False

    def invoke(self, prompt: str, callbacks: Optional[List] = None) -> str:
        def _invoke():
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            result = response.choices[0].message.content
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_llm_end"):
                        cb.on_llm_end(response)
            return result

        return self._retry_with_backoff(_invoke)

    def stream(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        client = self._get_client()
        stream = client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                if callback:
                    callback(token)
                yield token


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM Provider"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("Anthropic API key not provided")
                self._client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                raise ImportError("anthropic package is required for Anthropic provider")
        return self._client

    def check_availability(self) -> bool:
        try:
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            return bool(api_key)
        except Exception:
            return False

    def invoke(self, prompt: str, callbacks: Optional[List] = None) -> str:
        def _invoke():
            client = self._get_client()
            response = client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_llm_end"):
                        cb.on_llm_end(response)
            return result

        return self._retry_with_backoff(_invoke)

    def stream(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        client = self._get_client()
        with client.messages.stream(
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                if callback:
                    callback(text)
                yield text


class LLMProviderFactory:
    """Factory for creating LLM providers"""

    _providers = {
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.GROQ: GroqProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
    }

    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMProvider:
        provider_class = cls._providers.get(config.provider)
        if not provider_class:
            raise ValueError(f"Unknown provider: {config.provider}")
        return provider_class(config)

    @classmethod
    def create_with_fallback(cls, configs: List[LLMConfig]) -> BaseLLMProvider:
        """Create provider with automatic fallback to next available"""
        for config in configs:
            try:
                provider = cls.create(config)
                if provider.is_available:
                    logger.info(f"Using {config.provider.value} provider with model {config.model_name}")
                    return provider
            except Exception as e:
                logger.warning(f"Provider {config.provider.value} failed: {e}")

        raise RuntimeError("No available LLM providers found")


class RobustLLM:
    """
    Robust LLM wrapper with automatic fallback, caching, and enhanced streaming
    """

    def __init__(self, configs: Optional[List[LLMConfig]] = None):
        if configs is None:
            configs = [
                LLMConfig(provider=ProviderType.OLLAMA, model_name="mistral"),
            ]

        self.configs = configs
        self._provider = None
        self._current_config = None

    @property
    def provider(self) -> BaseLLMProvider:
        if self._provider is None:
            self._provider = LLMProviderFactory.create_with_fallback(self.configs)
            self._current_config = self._provider.config
        return self._provider

    @property
    def current_model(self) -> str:
        return self._current_config.model_name if self._current_config else "unknown"

    @property
    def current_provider(self) -> str:
        return self._current_config.provider.value if self._current_config else "unknown"

    def invoke(self, prompt: str, callbacks: Optional[List] = None) -> str:
        """Invoke LLM with automatic retry and fallback"""
        try:
            return self.provider.invoke(prompt, callbacks)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            # Try to find alternative provider
            for config in self.configs:
                if config != self._current_config:
                    try:
                        alt_provider = LLMProviderFactory.create(config)
                        if alt_provider.is_available:
                            self._provider = alt_provider
                            self._current_config = config
                            return self.provider.invoke(prompt, callbacks)
                    except Exception:
                        continue
            raise

    def stream(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> Generator[str, None, None]:
        """Stream response with callback support"""
        yield from self.provider.stream(prompt, callback)

    def invoke_with_langchain_callbacks(self, prompt: str, callbacks: List) -> str:
        """Special invoke method for LangChain streaming callbacks"""
        if self._current_config and self._current_config.provider == ProviderType.OLLAMA:
            return self.provider.invoke(prompt, callbacks)

        # For other providers, simulate callback behavior
        full_response = ""
        for token in self.stream(prompt):
            full_response += token
            for cb in callbacks:
                if hasattr(cb, "on_llm_new_token"):
                    cb.on_llm_new_token(token)

        for cb in callbacks:
            if hasattr(cb, "on_llm_end"):
                cb.on_llm_end(None)

        return full_response


# Default model configurations for different providers
DEFAULT_MODELS = {
    ProviderType.OLLAMA: ["mistral", "llama3.2", "qwen2.5", "deepseek-r1:8b"],
    ProviderType.OPENAI: ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
    ProviderType.GROQ: ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"],
    ProviderType.ANTHROPIC: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
}


def get_available_models(provider: ProviderType, base_url: str = "http://localhost:11434") -> List[str]:
    """Get available models for a provider"""
    if provider == ProviderType.OLLAMA:
        try:
            import requests
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "").split(":")[0] for m in models]
        except Exception:
            pass
    return DEFAULT_MODELS.get(provider, [])
