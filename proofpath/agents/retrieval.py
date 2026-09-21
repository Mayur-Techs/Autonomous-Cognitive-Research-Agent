import os
import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from tavily import AsyncTavilyClient
from models import Source, ProviderHealth

_tavily_health = ProviderHealth(name="tavily")

async def search_for_subquestion(query: str, max_results: int = 3) -> list[Source]:
    """Retrieve sources for a single sub-question using Tavily with retry + health tracking."""
    if not _tavily_health.is_available():
        # Provider is OPEN — skip silently, orchestrator will see 0 sources
        return []

    tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    for attempt in range(3):
        try:
            response = await tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_raw_content=True
            )
            _tavily_health.record_success()
            break
        except Exception as exc:
            _tavily_health.record_failure()
            if attempt == 2 or not _tavily_health.is_available():
                # All retries exhausted or circuit opened
                return []
            # Exponential backoff: 1s, 2s
            await asyncio.sleep(2 ** attempt)
    else:
        return []

    sources = []
    for idx, result in enumerate(response.get("results", [])):
        raw_text = result.get("raw_content") or result.get("content", "")
        if not raw_text:
            continue

        source_id = str(abs(hash(result["url"] + str(idx))))
        sources.append(Source(
            id=source_id,
            url=result["url"],
            title=result.get("title") or "Untitled",
            source_type="web",
            accessed_at=datetime.now(timezone.utc),
            retrieval_query=query,
            content_hash=str(hash(raw_text[:500])),
            raw_text=raw_text,
        ))
    return sources


async def retrieve_concurrently(queries: list[str]) -> list[list[Source]]:
    """Run retrieval concurrently for a tier of sub-questions."""
    tasks = [search_for_subquestion(q) for q in queries]
    return await asyncio.gather(*tasks)


def get_provider_health() -> dict:
    return _tavily_health.model_dump()
