import asyncio
from agents.llm_client import chat_complete
from models import ResearchState, SubQuestion

SYNTHESIS_SYSTEM_PROMPT = """You are a rigorous research synthesizer.
You will be given a sub-question and a list of claims with their verification_status.

Follow these rules EXACTLY for every claim:
| status               | required language                                                  |
|----------------------|--------------------------------------------------------------------|
| supported            | state as a finding, cite source title                              |
| contradicted         | explicitly state the disagreement, cite both sides                 |
| unresolved           | hedge clearly, do NOT state as fact                                |
| insufficient_evidence| write an explicit "insufficient evidence" sentence                 |
| rejected             | write an explicit "insufficient evidence" sentence                 |

Do NOT invent claims or add information not in the provided list.
If there are no claims, output a clear "insufficient evidence" statement."""


async def synthesize_subquestion(
    sub_q: SubQuestion, claims_for_subq: list, sources_dict: dict
) -> str:
    claims_text = ""
    for idx, c in enumerate(claims_for_subq):
        src = sources_dict.get(c.source_id)
        title = src.title if src else "Unknown Source"
        claims_text += (
            f"Claim {idx + 1}: {c.text}\n"
            f"Status: {c.verification_status}\n"
            f"Source: {title}\n\n"
        )

    if not claims_text:
        claims_text = "No claims found. (Status: insufficient_evidence)"

    try:
        return await chat_complete(
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Sub-question: {sub_q.text}\n\nClaims:\n{claims_text}",
                },
            ],
            temperature=0.2,
            max_tokens=400,
        )
    except Exception as e:
        return f"[Synthesis failed for this sub-question: {e}]"


async def synthesize_final_report(state: ResearchState) -> str:
    tasks = []
    sources_dict = getattr(state, "sources", {})
    for sub_q in state.sub_questions:
        sub_claims = [c for c in state.claims if c.sub_question_id == sub_q.id]
        tasks.append(synthesize_subquestion(sub_q, sub_claims, sources_dict))

    results = await asyncio.gather(*tasks)

    report = f"# Research Report\n\n**Question:** {state.question}\n\n"

    if state.conflicts:
        report += f"> ⚡ **{len(state.conflicts)} contradiction(s) detected** — see Contradictions panel.\n\n"

    insuff = [sq for sq in state.sub_questions if sq.status == "insufficient_evidence"]
    if insuff:
        report += f"> ⚠️ **{len(insuff)} sub-question(s) had insufficient evidence.**\n\n"

    report += "---\n\n"

    for sub_q, synthesis in zip(state.sub_questions, results):
        tier_label = f"[{sub_q.tier}]"
        status_icon = "✅" if sub_q.status == "answered" else "⚠️"
        report += f"## {status_icon} {sub_q.text} {tier_label}\n\n{synthesis}\n\n---\n\n"

    report += f"\n**Stop reason:** {state.stop_reason or 'complete'}\n"
    report += f"**Coverage:** {state.metrics.coverage_pct}% ({state.metrics.answered}/{state.metrics.total_subquestions} answered)\n"

    return report
