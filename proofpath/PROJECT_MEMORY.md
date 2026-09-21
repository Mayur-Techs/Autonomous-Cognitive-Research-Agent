# 🧠 Project Memory: ProofPath

**Date Created:** September 2026
**Purpose of this document:** A comprehensive "brain dump" so you can return to this project in a month or a year and instantly understand how it works, why it was built this way, and how to fix it if it breaks.

---

## 1. What is ProofPath?
ProofPath is an **evidence-first autonomous research agent**. Instead of simply asking an LLM to "write a report" (which causes hallucinations and unverified claims), ProofPath forces the AI to extract precise quotes, verify them algorithmically, and compile a report *only* using validated claims.

**Core Philosophy:** "The LLM proposes. The evidence compiler decides what is allowed into the final report."

**Strict Rules Followed:**
- **No Heavy Frameworks:** No LangChain, No LangGraph. Pure Python `asyncio` to maintain absolute control over the execution flow.
- **Quote Offsets:** Evidence is tracked using exact character offsets (`start_char`, `end_char`) to guarantee the LLM didn't invent the quote.
- **Separation of Concerns:** `verification_status` (supported/contradicted) is completely separate from `confidence` (high/medium/low).

---

## 2. Core Architecture & Data Flow

The system runs on a **Two-Tier Orchestrator** inside `orchestrator.py`:
1. **Tier 1 (Foundational):** Core definition questions.
2. **Tier 2 (Dependent):** Questions that rely on Tier 1 context.

### The Pipeline Flow (Per Sub-Question):
1. **`planner.py`**: Breaks the user's question into 4-6 tiered sub-questions.
2. **`retrieval.py`**: Calls Tavily API to fetch web sources.
3. **`analysis.py`**: LLM reads the source and extracts atomic `Claim` objects and `EvidenceSpan` objects (exact string matches).
4. **`fact_checker.py`**:
   - *Step 1 (Deterministic):* Checks if the exact quote actually exists in the raw text. (If not, claim is instantly REJECTED).
   - *Step 2 (LLM Judge):* Evaluates if the quote actually entails the claim.
   - *Step 3 (Contradiction Engine):* Groups all verified claims and checks if any contradict each other.
5. **`synthesis.py`**: Writes the final report, explicitly hedging or citing contradictions based on the verification statuses.

---

## 3. The Tech Stack & APIs

- **Backend:** FastAPI, Python `asyncio`, Pydantic (for strict schema validation).
- **Frontend:** Vanilla HTML/JS (`static/index.html`). Uses Server-Sent Events (SSE) for the live trace UI. No React/Node build steps required.
- **Retrieval API:** Tavily (`TAVILY_API_KEY`).
- **LLM Routing:** Custom fallback client (`agents/llm_client.py`).

### The LLM Fallback System (Free Tier Optimized)
Because free LLM tiers have aggressive rate limits, `llm_client.py` uses a custom routing system. It relies on the official `openai` Python SDK (since most providers use OpenAI-compatible endpoints) and tries providers in this order:
1. **Groq** (`GROQ_API_KEY`): `qwen-2.5-32b` or `compound`. Blazing fast, free.
2. **Gemini** (`GEMINI_API_KEY`): Google's free tier.
3. **OpenRouter** (`OPENROUTER_API_KEY`): Points to `meta-llama/llama-3.1-8b-instruct:free`.

---

## 4. Directory Map

```text
proofpath/
├── main.py               # The FastAPI server, REST endpoints, SSE stream
├── orchestrator.py       # The brain. Manages the 2-tier queue and budget
├── models.py             # Pydantic schemas (SubQuestion, Claim, EvidenceSpan, ResearchState)
├── agents/
│   ├── llm_client.py     # Unified LLM caller with auto-fallback and concurrency limits
│   ├── planner.py        # Breaks down the question
│   ├── retrieval.py      # Searches the web (Tavily)
│   ├── analysis.py       # Extracts claims and exact quotes
│   ├── fact_checker.py   # Verifies quotes and detects contradictions
│   └── synthesis.py      # Compiles the final report
├── static/
│   └── index.html        # The beautiful dark-mode UI
└── tests/
    └── test_proofpath.py # 24 offline smoke tests
```

---

## 5. Troubleshooting (If It Fails, Read This)

### Issue: Groq throws `429 RateLimitError` constantly
**Why:** Groq's free tier has a hard limit of 7000 Input Tokens/Min and 1000 Output Tokens/Min.
**The Fix we applied:** 
- In `llm_client.py`, `_llm_semaphore = asyncio.Semaphore(2)` limits the app to 2 concurrent LLM requests. Do not increase this unless using a paid API.
- We explicitly pass `max_tokens=800` (or 400 for synthesis). If you remove this, the SDK defaults to requesting 1024 tokens, and Groq will instantly reject it because 1024 > 1000 limit.

### Issue: The UI is stuck on "Starting Research..."
**Why:** The Server-Sent Events (SSE) connection dropped or an API key is totally invalid.
**To Fix:** 
1. Check the VS Code terminal. `main.py` logs exactly which LLM provider failed.
2. Ensure you have valid keys in your `.env` file. (Check `.env.example` for links to get them).

### Issue: "ValueError: Could not extract JSON from planner output"
**Why:** The LLM ignored the system prompt and didn't output a ` ```json ` block.
**To Fix:** The code already has regex fallback logic to aggressively hunt for arrays `[...]`, but if a new model is completely broken, you may need to tweak the prompt in `planner.py`.

---

## 6. Future Roadmap (What to do next)

If you decide to pick this project back up and expand it, here is what was planned:

1. **Phase 2: Academic Adapters**
   - Swap out Tavily for PubMed, ArXiv, or OpenAlex APIs so this can be used for deep medical/scientific research.
2. **Phase 3: Work Clustering (Deduplication)**
   - Right now, sources are unique based on URL. Add logic to deduplicate papers by DOI or PMID so the agent doesn't read the same paper from two different links.
3. **Phase 4: Human-in-the-Loop**
   - Pause the orchestrator after `planner.py` finishes. Send the sub-questions to the UI, let the user edit them, and click "Approve & Continue".
4. **Phase 5: Exporting**
   - Add a button to the UI that generates a formatted PDF of the final report with markdown-to-pdf conversion.

---

## 7. Quick Start Memory

**To run the server:**
```bash
uvicorn main:app --reload
# Then open http://127.0.0.1:8000
```

**To run the tests:**
```bash
pytest tests/ -v
```
