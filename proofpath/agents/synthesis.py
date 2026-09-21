import os
from openai import AsyncOpenAI
from models import ResearchState, SubQuestion

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYNTHESIS_SYSTEM_PROMPT = """You are a rigorous research synthesizer. You will be provided with a research question, a sub-question, and a list of claims gathered from sources. 
Each claim has a specific `verification_status`. You MUST follow these generation rules for every claim based on its status:

| status | allowed language |
| --- | --- |
| supported | stated as a finding |
| contradicted | must state the disagreement explicitly, cite both sides |
| unresolved | must be hedged, not stated as fact |
| insufficient_evidence | subquestion gets an explicit "insufficient evidence" sentence, never silently dropped |
| rejected | subquestion gets an explicit "insufficient evidence" sentence, never silently dropped |

Write a final synthesis paragraph for this sub-question following these rules exactly. Cite sources by their titles where appropriate. If all claims are rejected or there are no claims, output a clear statement of insufficient evidence."""

async def synthesize_subquestion(sub_q: SubQuestion, claims_for_subq: list, sources_dict: dict) -> str:
    claims_text = ""
    for idx, c in enumerate(claims_for_subq):
        source_title = sources_dict[c.source_id].title if c.source_id in sources_dict else "Unknown Source"
        claims_text += f"Claim {idx+1}: {c.text}\nStatus: {c.verification_status}\nSource: {source_title}\n\n"
        
    if not claims_text:
        claims_text = "No claims found. (Status: insufficient_evidence)"

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"Sub-question: {sub_q.text}\n\nClaims:\n{claims_text}"}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

async def synthesize_final_report(state: ResearchState) -> str:
    import asyncio
    
    tasks = []
    for sub_q in state.sub_questions:
        sub_claims = [c for c in state.claims if c.sub_question_id == sub_q.id]
        # state.sources is a dict we'll attach dynamically during the run
        tasks.append(synthesize_subquestion(sub_q, sub_claims, getattr(state, "sources", {})))
        
    results = await asyncio.gather(*tasks)
    
    report = f"# Final Report: {state.question}\n\n"
    for sub_q, synthesis in zip(state.sub_questions, results):
        report += f"## {sub_q.text}\n{synthesis}\n\n"
        
    return report
