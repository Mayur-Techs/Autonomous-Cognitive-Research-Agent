import os
import json
import re
import uuid
import asyncio
from openai import AsyncOpenAI
from models import Claim, Source, Conflict

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Stage 4 / Section 5: Exact deterministic quote-match + LLM entailment
def quote_matches(source_text: str, quote: str) -> bool:
    norm = lambda s: " ".join(s.split()).lower()
    return norm(quote) in norm(source_text)

class LLMJudge:
    """Wrapper to make the provided 'await llm.judge' code work seamlessly"""
    def __init__(self, async_client):
        self.client = async_client
        
    async def judge(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

async def fact_check(claim: Claim, source: Source, llm: LLMJudge) -> Claim:
    # 1. Deterministic check
    if not quote_matches(source.raw_text, claim.quote):
        claim.verification_status = "rejected" # never even reaches the LLM
        return claim
        
    # 2. LLM Entailment
    verdict = await llm.judge(
        f'Passage: "{claim.quote}"\nClaim: "{claim.text}"\n'
        f'Does the passage support, contradict, or neither support nor contradict the claim? '
        f'Answer with exactly one word: supported, contradicted, or neither.'
    )
    
    claim.verification_status = {
        "supported": "supported", 
        "contradicted": "contradicted"
    }.get(verdict.strip().lower(), "unresolved")
    
    return claim

def compute_confidence(claim_status_counts: dict) -> str:
    if claim_status_counts.get("supported", 0) >= 2: return "high"
    if claim_status_counts.get("supported", 0) == 1: return "medium"
    return "low"

async def fact_check_concurrently(claims: list[Claim], sources_dict: dict[str, Source]) -> list[Claim]:
    llm = LLMJudge(client)
    tasks = [fact_check(c, sources_dict[c.source_id], llm) for c in claims]
    return await asyncio.gather(*tasks)

# Stage 4 / Section 6: Contradiction Detection
CONTRADICTION_PROMPT = """Here are claims answering the same research sub-question:
{claims_list}
Return a JSON list of any pairs that contradict each other, with a one-sentence reason. If none, return [].
Strict format: [{{"claim_a": "...", "claim_b": "...", "reason": "..."}}]"""

async def detect_contradictions(claims: list[Claim], sources_dict: dict[str, Source]) -> list[Conflict]:
    # Group claims by subquestion to evaluate contradiction per group
    # Note: We assume this function is called per subquestion group as per the spec:
    # "one contradiction-detection LLM call per subquestion group"
    if len(claims) < 2:
        return []
        
    claims_list_str = ""
    for idx, c in enumerate(claims):
        source = sources_dict.get(c.source_id)
        title = source.title if source else "Unknown"
        claims_list_str += f"{idx+1}. {c.text} — source: {title}\n"
        
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a logical contradiction detector. Return strict JSON array."},
            {"role": "user", "content": CONTRADICTION_PROMPT.format(claims_list=claims_list_str)}
        ],
        temperature=0.0
    )
    
    content = response.choices[0].message.content
    blocks = re.findall(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
    if blocks:
        json_str = blocks[-1]
    else:
        arrays = re.findall(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
        if arrays:
            json_str = arrays[-1]
        else:
            return []
            
    try:
        pairs = json.loads(json_str)
    except:
        return []
        
    conflicts = []
    for pair in pairs:
        conflicts.append(Conflict(
            id=f"conf_{uuid.uuid4().hex[:8]}",
            claim_a=pair.get("claim_a", ""),
            claim_b=pair.get("claim_b", ""),
            reason=pair.get("reason", "")
        ))
        
        # Mark the specific claims as contradicted if they were previously supported or unresolved
        for c in claims:
            if c.text == pair.get("claim_a") or c.text == pair.get("claim_b"):
                c.verification_status = "contradicted"

    return conflicts
