import json
import re
from agents.llm_client import chat_complete
from models import SubQuestion

PLANNER_PROMPT = """Research question: "{question}"
Break this into 4-6 sub-questions needed to answer it rigorously.
Tag each as "foundational" (must be answered first, defines terms/scope) or
"dependent" (needs a foundational answer as context).
Return strict JSON: [{{"id": "sq_1", "text": "...", "tier": "foundational|dependent"}}]
Then critique your own list: are any sub-questions too vague to search for directly?
Revise those and return the final JSON array only inside a ```json block."""


async def generate_plan(question: str) -> list[SubQuestion]:
    content = await chat_complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous research planner. Follow the prompt exactly. "
                    "Output the final JSON array inside a ```json code block."
                ),
            },
            {"role": "user", "content": PLANNER_PROMPT.format(question=question)},
        ],
        temperature=0.7,
    )

    # Extract last JSON array from response
    blocks = re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    json_str = blocks[-1] if blocks else None
    if not json_str:
        arrays = re.findall(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        json_str = arrays[-1] if arrays else None
    if not json_str:
        raise ValueError(f"Could not extract JSON from planner output:\n{content}")

    sub_qs_data = json.loads(json_str)
    return [SubQuestion(**sq) for sq in sub_qs_data]
