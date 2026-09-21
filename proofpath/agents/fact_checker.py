import json
import re
import uuid
import asyncio
from agents.llm_client import chat_complete
from models import Claim, Source, Conflict

# ── Deterministic quote-match ─────────────────────────────────────────────────
def quote_matches(source_text: str, quote: str) -> bool:
    norm = lambda s: " ".join(s.split()).lower()
    return bool(quote.strip()) and norm(quote) in norm(source_text)


# ── LLM entailment judge ─────────────────────────────────────────────────────
ENTAILMENT_PROMPT = (
    'Passage: "{quote}"\n'
    'Claim: "{claim}"\n'
    "Does the passage support, contradict, or neither support nor contradict the claim?\n"
    "Answer with exactly one word: supported, contradicted, or neither."
)


async def fact_check(claim: Claim, source: Source) -> Claim:
    # 1. Deterministic gate — no LLM call if quote is absent
    if not quote_matches(source.raw_text, claim.quote):
        claim.verification_status = "rejected"
        return claim

    # 2. LLM entailment (uses fallback chain automatically)
    try:
        verdict = await chat_complete(
            messages=[
                {
                    "role": "user",
                    "content": ENTAILMENT_PROMPT.format(
                        quote=claim.quote[:2000], claim=claim.text
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=10,
        )
        verdict = verdict.strip().lower().split()[0] if verdict.strip() else "neither"
    except Exception:
        verdict = "neither"

    claim.verification_status = {
        "supported": "supported",
        "contradicted": "contradicted",
    }.get(verdict, "unresolved")

    return claim


def compute_confidence(claim_status_counts: dict) -> str:
    if claim_status_counts.get("supported", 0) >= 2:
        return "high"
    if claim_status_counts.get("supported", 0) == 1:
        return "medium"
    return "low"


async def fact_check_concurrently(
    claims: list[Claim], sources_dict: dict[str, Source]
) -> list[Claim]:
    tasks = []
    for c in claims:
        src = sources_dict.get(c.source_id)
        if src:
            tasks.append(fact_check(c, src))
        else:
            c.verification_status = "rejected"
    if tasks:
        checked = await asyncio.gather(*tasks)
        return list(checked)
    return claims


# ── Contradiction detection ───────────────────────────────────────────────────
CONTRADICTION_PROMPT = """Here are claims answering the same research sub-question:
{claims_list}
Return a JSON list of any pairs that contradict each other, with a one-sentence reason. If none, return [].
Strict format inside a ```json block: [{{"claim_a": "...", "claim_b": "...", "reason": "..."}}]"""


async def detect_contradictions(
    claims: list[Claim], sources_dict: dict[str, Source]
) -> list[Conflict]:
    if len(claims) < 2:
        return []

    claims_list_str = ""
    for idx, c in enumerate(claims):
        src = sources_dict.get(c.source_id)
        title = src.title if src else "Unknown"
        claims_list_str += f"{idx + 1}. {c.text} — source: {title}\n"

    try:
        content = await chat_complete(
            messages=[
                {
                    "role": "system",
                    "content": "You are a logical contradiction detector. Return strict JSON in a ```json block.",
                },
                {
                    "role": "user",
                    "content": CONTRADICTION_PROMPT.format(claims_list=claims_list_str),
                },
            ],
            temperature=0.0,
        )
    except Exception:
        return []

    blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    json_str = blocks[-1] if blocks else None
    if not json_str:
        arrays = re.findall(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        json_str = arrays[-1] if arrays else None
    if not json_str:
        return []

    try:
        pairs = json.loads(json_str)
    except Exception:
        return []

    conflicts = []
    for pair in pairs:
        claim_a_text = pair.get("claim_a", "")
        claim_b_text = pair.get("claim_b", "")
        conflicts.append(
            Conflict(
                id=f"conf_{uuid.uuid4().hex[:8]}",
                claim_a=claim_a_text,
                claim_b=claim_b_text,
                reason=pair.get("reason", "Unresolved"),
            )
        )
        for c in claims:
            if c.text == claim_a_text or c.text == claim_b_text:
                c.verification_status = "contradicted"

    return conflicts
