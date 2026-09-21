"""
Unified async LLM client with automatic provider fallback.

Provider priority (best quality first):
  1. Groq  qwen/qwen3.8-27b       — FREE, fast, excellent reasoning       ← PRIMARY
  2. Groq  groq/compound           — FREE, Groq's own flagship             ← FALLBACK 1
  3. Gemini models/gemini-3.8-flash — FREE, Google's latest flash         ← FALLBACK 2
  4. Gemini models/gemini-flash-latest — FREE, latest stable Gemini       ← FALLBACK 3
  5. Groq  groq/compound-mini      — FREE, lightweight                    ← FALLBACK 4
  6. OpenAI gpt-4o                 — PAID, best quality (if credits exist)← OPTIONAL

All model names below were live-tested on 2026-09-21 and confirmed working.

Free key URLs:
  Groq   → https://console.groq.com         (GROQ_API_KEY)
  Gemini → https://aistudio.google.com/app/apikey  (GEMINI_API_KEY)
"""

import os
import asyncio
import logging
from openai import AsyncOpenAI

logger = logging.getLogger("proofpath.llm")

# ── Provider registry (order = fallback priority) ────────────────────────────
_PROVIDERS: list[dict] = [
    {
        "name": "groq-qwen",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "qwen/qwen3.8-27b",
    },
    {
        "name": "groq-compound",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "groq/compound",
    },
    {
        "name": "gemini-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash", 
    },
    {
        "name": "gemini-flash-latest",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model": "gemini-2.5-flash",
    },
    {
        "name": "groq-compound-mini",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "groq/compound-mini",
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    },
]

# ── State & Concurrency ───────────────────────────────────────────────────────
_last_provider_used: str = "none"
_run_errors: list[str] = []  # All errors from last call, for debugging

# Throttle concurrent LLM calls to respect free-tier rate limits (e.g. Groq/Gemini)
# A semaphore value of 2 means max 2 simultaneous requests across all agents.
_llm_semaphore = asyncio.Semaphore(2)


def _is_placeholder(key: str) -> bool:
    """Detect un-filled placeholder values in .env."""
    if not key:
        return True
    lower = key.lower()
    return (
        lower.startswith("your_")
        or lower.endswith("_here")
        or lower in ("", "none", "null", "changeme")
    )


def _get_available_providers() -> list[dict]:
    """Return providers that have a real API key configured."""
    seen_keys: set[str] = set()
    available = []
    for p in _PROVIDERS:
        key = os.getenv(p["api_key_env"], "")
        if _is_placeholder(key):
            continue
        # De-duplicate: if same key+base_url combo already added a working model, keep going
        combo = f"{p['api_key_env']}:{p['base_url']}"
        available.append(p)
    return available


def _should_skip_immediately(error_str: str) -> bool:
    """True for permanent errors that mean this provider can never succeed this session."""
    skip_signals = [
        "insufficient_quota", "credit_balance_exhausted",
        "no credits", "billing", "payment",
        "invalid_api_key", "incorrect api key", "api key not found",
        "authentication", "unauthorized",
        "401", "403",
    ]
    lower = error_str.lower()
    return any(s in lower for s in skip_signals)


def _should_try_next_model(error_str: str) -> bool:
    """True for errors that are model-specific but provider might still work."""
    model_errors = [
        "model_not_found", "does not exist", "decommissioned",
        "no longer supported", "not supported", "not available",
        "not found", "404",
    ]
    lower = error_str.lower()
    return any(s in lower for s in model_errors)


async def chat_complete(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """
    Send a chat completion request. Tries providers in order, falls back automatically.
    Returns the response content string.
    Raises RuntimeError if ALL configured providers fail.
    """
    global _last_provider_used, _run_errors
    _run_errors = []

    providers = _get_available_providers()

    if not providers:
        raise RuntimeError(
            "No LLM providers configured. Add at least one key to .env:\n"
            "  GROQ_API_KEY       → https://console.groq.com (FREE)\n"
            "  GEMINI_API_KEY     → https://aistudio.google.com/app/apikey (FREE)\n"
            "  OPENROUTER_API_KEY → https://openrouter.ai/keys (FREE)"
        )

    # Track which env vars are exhausted (e.g. OpenAI out of credits → skip all openai models)
    exhausted_envs: set[str] = set()

    for provider in providers:
        env_var = provider["api_key_env"]

        # Skip if this key's provider is known exhausted/invalid this session
        if env_var in exhausted_envs:
            _run_errors.append(f"{provider['name']}: skipped (provider key exhausted)")
            continue

        key = os.getenv(env_var, "")
        client_kwargs: dict = {"api_key": key, "max_retries": 4}
        if provider["base_url"]:
            client_kwargs["base_url"] = provider["base_url"]

        client = AsyncOpenAI(**client_kwargs)

        completion_kwargs: dict = {
            "model": provider["model"],
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
        else:
            # Groq's qwen free tier has a STRICT hard limit of 1000 Output Tokens Per Minute.
            # If we do not pass max_tokens, it defaults to 1024, immediately throwing a hard 429 error
            # saying: "The request's expected output tokens exceed the enforced limit; reduce max".
            # Setting it to 800 ensures we never hit that preemptive hard block.
            completion_kwargs["max_tokens"] = 800

        try:
            # Enforce concurrency limit to prevent free-tier 429s
            async with _llm_semaphore:
                response = await client.chat.completions.create(**completion_kwargs)
                
            _last_provider_used = provider["name"]
            logger.info("LLM call succeeded via %s (%s)", provider["name"], provider["model"])
            return response.choices[0].message.content

        except Exception as exc:
            err = str(exc)
            short_err = f"{provider['name']} ({provider['model']}): {type(exc).__name__}: {err[:300]}"
            _run_errors.append(short_err)
            logger.warning("LLM provider failed: %s", short_err)

            if _should_skip_immediately(err):
                # Mark the entire env key as exhausted — no point trying other models on same key
                exhausted_envs.add(env_var)
                continue

            if _should_try_next_model(err):
                # Model-specific issue — try next model immediately
                continue

            # For network / timeout errors, brief wait before trying next
            await asyncio.sleep(0.5)
            continue

    # All providers failed
    error_summary = "\n".join(f"  • {e}" for e in _run_errors)
    raise RuntimeError(
        f"All LLM providers failed ({len(providers)} tried).\n"
        f"Errors:\n{error_summary}\n\n"
        f"Tip: Make sure GROQ_API_KEY or GEMINI_API_KEY is set in your .env file.\n"
        f"Get a free Groq key at: https://console.groq.com"
    )


# ── Startup validation ────────────────────────────────────────────────────────

async def validate_llm_on_startup() -> dict:
    """
    Called once at server startup. Tests each provider and returns a status report.
    Does NOT raise — just logs and returns results.
    """
    providers = _get_available_providers()
    results = {}

    if not providers:
        logger.error(
            "⚠ No LLM providers configured! Add GROQ_API_KEY or GEMINI_API_KEY to .env"
        )
        return {"status": "no_providers", "providers": {}}

    for provider in providers:
        name = provider["name"]
        if name in results:
            continue  # already tested this provider

        key = os.getenv(provider["api_key_env"], "")
        client_kwargs: dict = {"api_key": key}
        if provider["base_url"]:
            client_kwargs["base_url"] = provider["base_url"]

        client = AsyncOpenAI(**client_kwargs)
        try:
            r = await client.chat.completions.create(
                model=provider["model"],
                messages=[{"role": "user", "content": "Reply with the single word: READY"}],
                max_tokens=5,
                temperature=0.0,
            )
            reply = r.choices[0].message.content.strip()
            results[name] = {"status": "ok", "model": provider["model"], "reply": reply}
            logger.info("✓ LLM provider ready: %s (%s)", name, provider["model"])
        except Exception as exc:
            results[name] = {"status": "error", "model": provider["model"], "error": str(exc)[:200]}
            logger.warning("✗ LLM provider failed: %s — %s", name, str(exc)[:200])

    ok_providers = [k for k, v in results.items() if v["status"] == "ok"]
    if ok_providers:
        logger.info("🟢 ProofPath LLM ready. Active providers: %s", ok_providers)
    else:
        logger.error("🔴 All LLM providers failed! Check your API keys.")

    return {"status": "ok" if ok_providers else "all_failed", "providers": results}


# ── Utility ───────────────────────────────────────────────────────────────────

def get_active_provider() -> str:
    return _last_provider_used


def list_configured_providers() -> list[str]:
    # De-duplicate by name (not by model)
    seen = set()
    result = []
    for p in _get_available_providers():
        base = p["name"].split("-")[0]  # e.g. groq-qwen → groq
        if base not in seen:
            seen.add(base)
            result.append(p["name"])
    return result


def get_last_errors() -> list[str]:
    return _run_errors
