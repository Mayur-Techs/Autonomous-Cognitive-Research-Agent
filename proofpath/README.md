# ProofPath

> **"The LLM proposes. The evidence compiler decides what is allowed into the final report."**

ProofPath is an evidence-first autonomous research system designed to make research outputs auditable. Instead of treating an LLM-generated report as the primary artifact, the system builds an intermediate evidence graph connecting research questions, source documents, evidence spans, claims, verification results and conflicts. The final report is compiled from validated claims, while unresolved or insufficient evidence is surfaced explicitly.

---

## Problem

AI research systems are good at producing fluent reports but can produce weakly supported, overconfident or inconsistently cited claims. The difficult problem is not generating prose — it is validating whether a claim is actually supported by the retrieved evidence.

## What This Is

A small, engineering-focused **evidence intelligence system** where:

1. LLMs propose research plans, candidate claims, and interpretations.
2. Deterministic code validates data contracts and evidence provenance via **exact character-offset `EvidenceSpan` objects**.
3. Independent verification signals test whether evidence actually supports a claim.
4. Contradictions are first-class objects — never hidden.
5. The final report is compiled only from validated claim objects.
6. The entire run is inspectable through a live trace and REST API.

---

## Architecture

```
USER QUESTION
      │
      ▼
  PLANNER  (self-critiques its own plan)
      │
      ▼
DEPENDENCY GRAPH
  tier=foundational ──► concurrent retrieval ──► EvidenceSpan extraction ──► claim ledger
  tier=dependent   ──► (waits for tier 1)  ──► same pipeline
                                                       │
                           ┌───────────────────────────┤
                           ▼                           ▼
                    LLM ENTAILMENT               QUOTE-MATCH
                    (single judge call)          (deterministic)
                           │
                           ▼
                      ADJUDICATOR  →  verification_status  (separate from confidence)
                           │
                           ▼
                  CONTRADICTION ENGINE
                           │
                           ▼
                   COVERAGE METRICS  →  stop_reason
                           │
                           ▼
                   REPORT COMPILER  (only allows supported/contradicted claims through)
                           │
                           ▼
                    FINAL REPORT  +  SSE live trace  +  REST API
```

## Why Hand-Rolled Orchestration

For this bounded prototype, a custom orchestrator provides direct control over state transitions, scheduling, tier dependencies, budget tracking, and trace events. A framework such as LangGraph would become more attractive if the graph grows significantly or requires richer human-in-the-loop branching.

---

## Agent Workflow

| Agent | Input | Output |
|---|---|---|
| `planner.py` | question | tiered sub-questions (self-critiqued) |
| `retrieval.py` | sub-question query | `Source` objects + ProviderHealth tracking |
| `analysis.py` | source + sub-question | `Claim` + `EvidenceSpan` with char offsets |
| `fact_checker.py` | claim + source | `verification_status` (quote-match then LLM) |
| `fact_checker.detect_contradictions` | claims per sub-question | `Conflict` objects |
| `synthesis.py` | validated claims + statuses | report prose obeying status rules |

---

## Evidence Model

### EvidenceSpan

The core auditability object. Stores exact character offsets so the quote can be mechanically verified against the source document:

```python
document.text[start_char:end_char] == quoted_text  # must be True or claim is REJECTED
```

### Claim Lifecycle

```
extracted  →  quote-match check  →  REJECTED (if quote not found)
                                 →  LLM entailment judge  →  supported / contradicted / unresolved
                                                           →  confidence assigned (separate field)
```

### Verification Status vs Confidence

These are deliberately separate fields on `Claim`. Conflating them is a known bug in naive implementations that allows a single verifier to produce artificially high confidence.

---

## Contradiction Handling

Conflicts are detected per sub-question group. Each `Conflict` stores both claim texts and a reason. Contradicted claims are **never smoothed away** — they appear in the conflict panel and are reported as disagreement in the synthesis.

---

## Stopping Logic

The orchestrator stops when any of these conditions is met:

| Reason | Description |
|---|---|
| `coverage_threshold_reached` | ≥80% of sub-questions answered |
| `research_budget_exhausted` | max searches / LLM calls / sources reached |
| `iteration_limit_reached` | max iterations reached |
| `planner_failed` | planner returned invalid output |
| `research_complete` | all tiers finished |

The stop reason is recorded in `ResearchState.stop_reason` and returned in the API response.

---

## Failure Handling

- **Provider failures**: `ProviderHealth` circuit-breaker with 3-failure threshold. Failed provider returns empty sources; run continues with available results.
- **Retry**: up to 3 attempts with exponential backoff (1s, 2s).
- **Malformed LLM JSON**: regex fallback extraction; returns empty list on total failure instead of crashing.
- **Quote not found**: claim is immediately `rejected` — never reaches LLM judge.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/research` | Start a new research run |
| `GET` | `/api/research/{run_id}/events` | SSE live trace stream |
| `GET` | `/api/research/{run_id}` | Run state + sub-question statuses |
| `GET` | `/api/research/{run_id}/claims` | All claims with verification status |
| `GET` | `/api/research/{run_id}/sources` | All retrieved sources (metadata) |
| `GET` | `/api/research/{run_id}/report` | Final compiled report |
| `GET` | `/api/research/{run_id}/conflicts` | All detected conflicts |
| `GET` | `/api/health` | Provider health + active run count |

---

## UI Panels

| Panel | What it shows |
|---|---|
| **Live Trace** | Timestamped agent events in real time via SSE |
| **Research Plan** | Sub-questions with tier (foundational/dependent) and live status |
| **Claim Explorer** | Every claim with status badge, confidence, and source quote |
| **Contradictions** | Conflict pairs with reason |
| **Evidence Gaps** | Sub-questions where verified evidence was insufficient |
| **Metrics Bar** | Sub-questions, sources, claims, verified count, conflicts, coverage %, LLM calls, searches |
| **Final Report** | Compiled report — only validated claims are asserted |

---

## Installation

```bash
git clone <repo-url>
cd proofpath

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

## Environment Variables

```
GROQ_API_KEY=...       # Required — Fastest, FREE models
GEMINI_API_KEY=...     # Required — Google's FREE models
OPENROUTER_API_KEY=... # Required — Free Llama models
TAVILY_API_KEY=...     # Required — web retrieval (free tier available)
```

**Note:** You only strictly need *one* of the LLM keys (Groq, Gemini, or OpenRouter). The system will automatically fall back if one runs out of quota.

## Running

```bash
uvicorn main:app --reload
# Open http://localhost:8000
```

## Running Tests

```bash
pytest tests/ -v
```

All tests run without API keys.

---

## AI-Assisted Development

AI tools were used for scaffolding, boilerplate generation, documentation assistance, and implementation suggestions.

Core architecture and reasoning decisions were manually reviewed. The following were explicitly implemented and reviewed:

- orchestration logic and tier scheduling
- evidence-span validation (character-offset round-trip check)
- claim verification pipeline (deterministic quote-match → LLM entailment)
- contradiction detection and conflict storage
- `verification_status` / `confidence` separation
- budget tracking and stopping criteria
- provider health / circuit-breaker model
- final report compiler status rules

Generated code was tested before inclusion.

---

## Limitations

1. Web retrieval is constrained by the Tavily API (free tier: limited searches/month).
2. Source extraction can fail on dynamically rendered pages.
3. LLM and quote-match verification are probabilistic signals, not truth oracles.
4. Canonical-work deduplication is not implemented in the MVP (URL-based IDs only).
5. Contradiction detection can miss nuanced methodological disagreements.
6. Confidence categories (`high/medium/low`) are engineering heuristics, not statistical measures.
7. The system does not establish scientific truth — it improves provenance and validation discipline.

---

## Future Roadmap

**Phase 2**
- Canonical work clustering by DOI/PMID
- Citation graph visualization
- Human approval gate (`POST /api/research/{run_id}/approve-plan`)
- PubMed / OpenAlex retrieval adapters
- Research replay from saved run artifacts

**Phase 3**
- Domain packs (cognitive science, market research, technical due diligence)
- PDF / HTML export
- Organization-level evidence memory
- NLI model as second verification signal

---

## Project Structure

```
proofpath/
├── main.py               # FastAPI app + all REST routes
├── orchestrator.py       # Two-tier scheduler with budget + coverage metrics
├── models.py             # All Pydantic schemas (SubQuestion, Claim, EvidenceSpan, …)
├── agents/
│   ├── planner.py        # Sub-question decomposition + self-critique
│   ├── retrieval.py      # Tavily async search + ProviderHealth circuit-breaker
│   ├── analysis.py       # Claim extraction + EvidenceSpan character offsets
│   ├── fact_checker.py   # Quote-match + LLM entailment + contradiction detection
│   └── synthesis.py      # Status-rule-driven report generation
├── static/
│   └── index.html        # Single-page UI (plain HTML/JS, no build step)
├── tests/
│   └── test_proofpath.py # Smoke tests (no API keys needed)
├── requirements.txt
├── .env.example
└── .gitignore
```
