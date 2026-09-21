import json
import re
import uuid
import asyncio
from agents.llm_client import chat_complete
from models import Claim, SubQuestion, Source, EvidenceSpan

ANALYSIS_PROMPT = """Sub-question: "{sub_question}"
Source text: "{raw_text}"
Extract up to 3 atomic claims from this source that help answer the sub-question.
For each claim:
  - Write one precise atomic fact (one idea only).
  - Quote the EXACT sentence or phrase from the source text that supports it —
    copy it character-for-character, do not paraphrase the quote.
Return strict JSON inside a ```json block: [{{"text": "...", "quote": "..."}}]"""


def _find_span(document_text: str, quote: str) -> tuple[int, int] | None:
    """Locate exact character offsets of a quote inside the document."""
    norm = lambda s: " ".join(s.split())
    norm_doc = norm(document_text)
    norm_quote = norm(quote)
    idx = norm_doc.find(norm_quote)
    if idx == -1:
        return None
    return (idx, idx + len(norm_quote))


async def extract_claims(
    sub_q: SubQuestion, source: Source
) -> tuple[list[Claim], list[EvidenceSpan]]:
    truncated_text = source.raw_text[:10000] if source.raw_text else ""

    try:
        content = await chat_complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data extraction assistant. Return ONLY a valid JSON array in a ```json block.",
                },
                {
                    "role": "user",
                    "content": ANALYSIS_PROMPT.format(
                        sub_question=sub_q.text, raw_text=truncated_text
                    ),
                },
            ],
            temperature=0.0,
        )
    except Exception:
        return [], []

    blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    json_str = blocks[-1] if blocks else None
    if not json_str:
        arrays = re.findall(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        json_str = arrays[-1] if arrays else None
    if not json_str:
        return [], []

    try:
        claims_data = json.loads(json_str)
    except json.JSONDecodeError:
        return [], []

    claims: list[Claim] = []
    spans: list[EvidenceSpan] = []

    for item in claims_data:
        quote = item.get("quote", "")
        span_id = f"span_{uuid.uuid4().hex[:8]}"

        offsets = _find_span(source.raw_text, quote) if quote else None
        if offsets:
            span = EvidenceSpan(
                id=span_id,
                source_id=source.id,
                start_char=offsets[0],
                end_char=offsets[1],
                quoted_text=quote,
            )
            spans.append(span)
        else:
            span_id = None

        claim = Claim(
            id=f"claim_{uuid.uuid4().hex[:8]}",
            sub_question_id=sub_q.id,
            text=item.get("text", ""),
            source_id=source.id,
            quote=quote,
            evidence_span_id=span_id,
        )
        claims.append(claim)

    return claims, spans


async def analyze_sources_concurrently(
    sub_q: SubQuestion, sources: list[Source]
) -> tuple[list[Claim], list[EvidenceSpan]]:
    """Run analysis concurrently for all sources for a single sub-question."""
    tasks = [extract_claims(sub_q, s) for s in sources]
    results = await asyncio.gather(*tasks)
    all_claims = [c for claims, _ in results for c in claims]
    all_spans = [s for _, spans in results for s in spans]
    return all_claims, all_spans
