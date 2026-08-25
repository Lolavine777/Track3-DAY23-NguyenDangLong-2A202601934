"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def get_llm(model: str | None = None, temperature: float = 0.0) -> BaseChatModel:

    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY -> ChatGoogleGenerativeAI
    2. OPENAI_API_KEY -> ChatOpenAI
    3. ANTHROPIC_API_KEY -> ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        g_model = model or os.getenv("LLM_MODEL") or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=g_model,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        o_model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        return ChatOpenAI(
            model=o_model,
            temperature=temperature,
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        a_model = model or os.getenv("LLM_MODEL") or "claude-sonnet-4-20250514"
        return ChatAnthropic(
            model=a_model,
            temperature=temperature,
        )


    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
