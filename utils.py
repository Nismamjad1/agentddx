"""Shared utilities for AgentDDx — OpenRouter LLM API wrapper.

All LLM calls go through llm_call() with automatic retry for:
  - Rate limits (429)
  - Transient server errors (500, 502, 503, 504)
  - Connection errors / timeouts
"""
import os, re, time
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path, override=True)

from openai import OpenAI

_client = None

# Use standard model, NOT :nitro — quantized variants amplify MCQ position bias
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.3-70b-instruct"
)


def has_api_key() -> bool:
    """Check if an LLM API key is configured (used by agents to gate LLM calls)."""
    return bool(os.environ.get("OPENROUTER_API_KEY", ""))


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        _client = OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1"
        )
    return _client


# Errors that are safe to retry (transient)
_RETRYABLE_PATTERNS = [
    "429", "rate_limit", "Rate limit",         # rate limits
    "500", "502", "503", "504",                 # server errors
    "Connection", "connection",                  # connection issues
    "Timeout", "timeout", "timed out",          # timeouts
    "overloaded", "capacity",                   # provider overload
    "SSLError", "RemoteDisconnected",           # network glitches
]


def _is_retryable(error_str: str) -> bool:
    """Check if an error is transient and safe to retry."""
    return any(p in error_str for p in _RETRYABLE_PATTERNS)


def llm_call(messages, max_tokens=500, temperature=0.1, max_retries=8):
    """Make an LLM API call with robust retry logic.

    Retries on rate limits AND transient errors (500, 502, timeouts, etc).
    Uses exponential backoff: 15s, 30s, 60s, 60s, 120s, ...
    """
    client = get_client()
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = r.choices[0].message.content
            if content is None:
                content = ""
            return content
        except Exception as e:
            err_str = str(e)
            if _is_retryable(err_str):
                # Exponential backoff: 15, 30, 60, 60, 120, 120, ...
                wait = min(15 * (2 ** attempt), 300)
                print(f"  Retryable error — waiting {wait}s (attempt {attempt+1}/{max_retries}): {err_str[:80]}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"OpenRouter failed after {max_retries} retries")
