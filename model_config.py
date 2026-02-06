"""
Model Configuration for PDF QA System
Supports multiple LLM providers with automatic fallback
"""

import os
from typing import List, Dict, Any

# Provider types
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"
PROVIDER_GROQ = "groq"
PROVIDER_ANTHROPIC = "anthropic"

# Default model configuration - Primary provider
MODEL_CONFIG = {
    "provider": PROVIDER_OLLAMA,
    "model_name": "mistral",
    "base_url": "http://localhost:11434",
    "api_key": None,
    "temperature": 0.7,
    "max_tokens": 4096,
    "streaming": True,
}

# Fallback configurations (used if primary fails)
FALLBACK_CONFIGS: List[Dict[str, Any]] = [
    {
        "provider": PROVIDER_GROQ,
        "model_name": "llama-3.3-70b-versatile",
        "api_key": os.getenv("GROQ_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    {
        "provider": PROVIDER_OPENAI,
        "model_name": "gpt-4o-mini",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 4096,
    },
]

# Available models per provider
AVAILABLE_MODELS = {
    PROVIDER_OLLAMA: [
        "mistral",
        "llama3.2",
        "llama3.2:1b",
        "qwen2.5",
        "qwen2.5:3b",
        "deepseek-r1:8b",
        "deepseek-r1:1.5b",
        "phi3",
        "gemma2",
        "codellama",
    ],
    PROVIDER_OPENAI: [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ],
    PROVIDER_GROQ: [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    PROVIDER_ANTHROPIC: [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
}

# Provider display names
PROVIDER_NAMES = {
    PROVIDER_OLLAMA: "Ollama (Local)",
    PROVIDER_OPENAI: "OpenAI",
    PROVIDER_GROQ: "Groq (Fast)",
    PROVIDER_ANTHROPIC: "Anthropic",
}

# Provider descriptions
PROVIDER_DESCRIPTIONS = {
    PROVIDER_OLLAMA: "Run models locally using Ollama. Free but requires local setup.",
    PROVIDER_OPENAI: "OpenAI's GPT models. High quality but requires API key.",
    PROVIDER_GROQ: "Ultra-fast inference with Groq. Free tier available.",
    PROVIDER_ANTHROPIC: "Anthropic's Claude models. High quality but requires API key.",
}


def get_provider_config(provider: str, model_name: str = None, api_key: str = None) -> Dict[str, Any]:
    """Get configuration for a specific provider"""
    config = {
        "provider": provider,
        "model_name": model_name or AVAILABLE_MODELS.get(provider, [""])[0],
        "temperature": 0.7,
        "max_tokens": 4096,
        "streaming": True,
    }

    if provider == PROVIDER_OLLAMA:
        config["base_url"] = MODEL_CONFIG.get("base_url", "http://localhost:11434")
    elif provider in [PROVIDER_OPENAI, PROVIDER_GROQ, PROVIDER_ANTHROPIC]:
        env_var = f"{provider.upper()}_API_KEY"
        config["api_key"] = api_key or os.getenv(env_var)

    return config


def validate_api_key(provider: str, api_key: str = None) -> bool:
    """Validate if API key is available for a provider"""
    if provider == PROVIDER_OLLAMA:
        return True

    key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
    return bool(key)
