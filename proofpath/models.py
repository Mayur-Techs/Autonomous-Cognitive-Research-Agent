from __future__ import annotations
from pydantic import BaseModel, Field, computed_field
from typing import Literal
from datetime import datetime, timezone
import hashlib


# ─────────────────────────────────────────────
# Budget & Health
# ─────────────────────────────────────────────

class ResearchBudget(BaseModel):
    max_iterations: int = 3
    max_searches: int = 30
    max_sources: int = 50
    max_llm_calls: int = 40
    timeout_seconds: int = 300

    # Counters (mutable during run)
    searches_used: int = 0
    llm_calls_used: int = 0
    sources_found: int = 0
    iterations_used: int = 0

    def exhausted(self) -> bool:
        return (
            self.searches_used >= self.max_searches
            or self.llm_calls_used >= self.max_llm_calls
            or self.sources_found >= self.max_sources
            or self.iterations_used >= self.max_iterations
        )

    def stop_reason(self) -> str | None:
        if self.searches_used >= self.max_searches:
            return "research_budget_exhausted: max_searches reached"
        if self.llm_calls_used >= self.max_llm_calls:
            return "research_budget_exhausted: max_llm_calls reached"
        if self.sources_found >= self.max_sources:
            return "research_budget_exhausted: max_sources reached"
        if self.iterations_used >= self.max_iterations:
            return "iteration_limit_reached"
        return None


class ProviderHealth(BaseModel):
    name: str
    failure_count: int = 0
    last_failure: datetime | None = None
    state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = "CLOSED"
    FAILURE_THRESHOLD: int = 3

    def record_failure(self):
        self.failure_count += 1
        self.last_failure = datetime.now(timezone.utc)
        if self.failure_count >= self.FAILURE_THRESHOLD:
            self.state = "OPEN"

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def is_available(self) -> bool:
        return self.state != "OPEN"


# ─────────────────────────────────────────────
# Core domain objects
# ─────────────────────────────────────────────

class SubQuestion(BaseModel):
    id: str
    text: str
    tier: Literal["foundational", "dependent"]
    depends_on: list[str] = []
    priority: Literal["high", "medium", "low"] = "high"
    status: Literal[
        "pending", "blocked", "ready", "researching",
        "answered", "insufficient_evidence", "failed"
    ] = "pending"


class Source(BaseModel):
    id: str
    url: str
    title: str
    source_type: str = "web"
    authors: list[str] = []
    publication_date: datetime | None = None
    accessed_at: datetime
    retrieval_query: str = ""
    content_hash: str = ""
    raw_text: str  # kept here for verification; NOT copied into ResearchState hot path


class EvidenceSpan(BaseModel):
    """Exact character-offset evidence provenance. Core auditability object."""
    id: str
    source_id: str
    start_char: int
    end_char: int
    quoted_text: str

    def validate_against(self, document_text: str) -> bool:
        """Deterministic check: offsets must round-trip exactly."""
        extracted = document_text[self.start_char:self.end_char]
        norm = lambda s: " ".join(s.split()).lower()
        return norm(extracted) == norm(self.quoted_text)


class Claim(BaseModel):
    id: str
    sub_question_id: str
    text: str                     # one atomic fact
    source_id: str
    quote: str                    # must quote-match source.raw_text (normalized)
    evidence_span_id: str | None = None   # links to EvidenceSpan once created
    verification_status: Literal[
        "supported", "contradicted", "unresolved",
        "insufficient_evidence", "rejected"
    ] = "unresolved"
    confidence: Literal["high", "medium", "low"] | None = None  # set AFTER status, never merged


class Conflict(BaseModel):
    id: str
    claim_a: str
    claim_b: str
    reason: str
    status: Literal["detected", "resolved", "unresolved"] = "detected"


class CoverageMetrics(BaseModel):
    total_subquestions: int = 0
    answered: int = 0
    insufficient_evidence: int = 0
    supported_claims: int = 0
    contradicted_claims: int = 0
    unresolved_claims: int = 0
    rejected_claims: int = 0
    total_sources: int = 0
    total_conflicts: int = 0
    llm_calls: int = 0
    searches: int = 0

    @property
    def coverage_pct(self) -> float:
        if self.total_subquestions == 0:
            return 0.0
        return round((self.answered / self.total_subquestions) * 100, 1)


class ResearchState(BaseModel):
    run_id: str
    question: str
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    sub_questions: list[SubQuestion] = []
    source_ids: list[str] = []      # IDs only — raw text lives in orchestrator.sources dict
    evidence_span_ids: list[str] = []
    claims: list[Claim] = []
    conflicts: list[Conflict] = []
    metrics: CoverageMetrics = Field(default_factory=CoverageMetrics)
    stop_reason: str | None = None
    final_report: str | None = None
    trace: list[dict] = []
