import os
import json
import re
from openai import AsyncOpenAI
from models import SubQuestion

# Ensure OPENAI_API_KEY is available
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PLANNER_PROMPT = """Research question: "{question}"
Break this into 4-6 sub-questions needed to answer it rigorously.
Tag each as "foundational" (must be answered first, defines terms/scope) or
"dependent" (needs a foundational answer as context).
Return strict JSON: [{"id": "...", "text": "...", "tier": "foundational|dependent"}]
Then critique your own list: are any sub-questions too vague to search for directly?
Revise those and return the final JSON only."""

async def generate_plan(question: str) -> list[SubQuestion]:
    response = await client.chat.completions.create(
        model="gpt-4o",  # or whichever explicit model you prefer
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous research planner. Follow the prompt instructions exactly. "
                    "Make sure to output the final JSON array within a markdown json codeblock."
                )
            },
            {"role": "user", "content": PLANNER_PROMPT.format(question=question)}
        ],
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    
    # We expect the model to critique and then return the revised JSON in a code block
    blocks = re.findall(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
    if blocks:
        json_str = blocks[-1]
    else:
        # Fallback if no code blocks are used
        arrays = re.findall(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
        if arrays:
            json_str = arrays[-1]
        else:
            raise ValueError(f"Could not extract JSON from planner output:\n{content}")
            
    sub_qs_data = json.loads(json_str)
    return [SubQuestion(**sq) for sq in sub_qs_data]

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    # Setup for quick local testing (run from project root)
    load_dotenv()
    
    async def test():
        q = "Does daily meditation permanently change brain structure in adults?"
        print(f"Testing planner with question: {q}\n")
        sub_qs = await generate_plan(q)
        for sq in sub_qs:
            print(f"[{sq.tier.upper()}] {sq.id}: {sq.text}")

    asyncio.run(test())
