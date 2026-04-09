"""
Prompt fetching utility — retrieves active AI prompts from the NestJS backend.

Uses a simple in-memory TTL cache (60s) to avoid hitting the backend on every
LLM call. Falls back to a hardcoded default if the backend is unavailable.
"""

import time
from typing import Optional

import httpx
import structlog

from .config import settings

logger = structlog.get_logger(__name__)

_CACHE_TTL = 60  # seconds
_cache: dict[str, tuple[str, float]] = {}  # type → (content, expires_at)

_FALLBACK_PROMPTS: dict[str, str] = {
    "TRIAGE": (
        "You are the triage agent for Helena, a security operations AI assistant. "
        "Classify the user's intent and return ONLY a JSON object: "
        '{"intent": "<rfc|incident|knowledge|escalation|unknown>"}'
    ),
    "FALLBACK": "You are a helpful security operations assistant.",
}


async def fetch_active_prompt(prompt_type: str, fallback: Optional[str] = None) -> str:
    """Return the active prompt content for *prompt_type*.

    Resolution order:
    1. In-memory cache (TTL: 60s)
    2. Backend GET /api/v3/ai-prompts/{type}/active
    3. Hardcoded fallback in _FALLBACK_PROMPTS
    4. *fallback* argument
    5. Empty string
    """
    now = time.monotonic()

    cached_content, expires_at = _cache.get(prompt_type, ("", 0.0))
    if cached_content and now < expires_at:
        return cached_content

    if settings.nestjs_base_url:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(
                    f"{settings.nestjs_base_url}/api/v3/ai-prompts/{prompt_type}/active",
                )
                resp.raise_for_status()
                content: str = resp.json()["content"]
                _cache[prompt_type] = (content, now + _CACHE_TTL)
                logger.debug("prompt_fetched", type=prompt_type, source="backend")
                return content
        except Exception as exc:
            logger.warning("prompt_fetch_failed", type=prompt_type, error=str(exc))

    default = fallback or _FALLBACK_PROMPTS.get(prompt_type, "")
    logger.debug("prompt_fallback", type=prompt_type)
    return default


def invalidate_prompt_cache(prompt_type: Optional[str] = None) -> None:
    """Invalidate cached prompt(s). Pass None to clear all."""
    if prompt_type:
        _cache.pop(prompt_type, None)
    else:
        _cache.clear()
