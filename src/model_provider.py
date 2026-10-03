from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`."""
    val = (value or "").strip().lower().replace("-", "").replace("_", "")
    if val in ("openai", "open_ai"):
        return "openai"
    if val in ("custom", "openaicompatible", "openai_compatible"):
        return "custom"
    if val in ("gemini", "google", "googlegenai"):
        return "gemini"
    if val in ("anthropic", "anthorpic", "claude"):
        return "anthropic"
    if val in ("ollama",):
        return "ollama"
    if val in ("openrouter", "open_router"):
        return "openrouter"
    return val


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider."""
    prov = normalize_provider(config.provider)
    if prov == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    elif prov == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    elif prov == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )
    elif prov == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )
    elif prov == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )
    elif prov == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
            return ChatOpenRouter(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url="https://openrouter.ai/api/v1",
            )
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
