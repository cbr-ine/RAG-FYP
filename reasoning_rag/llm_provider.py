"""LLM provider helpers for OpenAI clients."""
import os
from typing import Dict, Optional, Tuple

from openai import OpenAI
from env_utils import load_project_env


def get_llm_client() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Build an OpenAI client from environment variables."""
    load_project_env(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if openai_api_key:
        base_url = os.getenv("OPENAI_BASE_URL")
        kwargs = {"api_key": openai_api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs), "openai"

    return None, None


def get_model_name(role: str, provider: Optional[str]) -> Optional[str]:
    """Pick a model name for the requested role and provider."""
    if provider == "openai":
        if role == "decomposition":
            return os.getenv("OPENAI_DECOMPOSITION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        if role == "generation":
            return os.getenv("OPENAI_GENERATION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    return None


def get_token_limit_kwargs(model_name: Optional[str], output_tokens: int) -> Dict[str, int]:
    """Return the correct token-limit parameter for the target OpenAI model."""
    if not model_name:
        return {}
    if model_name.startswith("gpt-5"):
        return {"max_completion_tokens": output_tokens}
    return {"max_tokens": output_tokens}


def get_llm_status(role: str) -> Dict[str, Optional[str]]:
    """Return provider/model status for debugging and experiment metadata."""
    _, provider = get_llm_client()
    model_name = get_model_name(role, provider)
    return {
        "provider": provider,
        "model_name": model_name,
        "enabled": bool(provider and model_name)
    }
