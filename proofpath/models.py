from pydantic import BaseModel
from typing import Literal
from datetime import datetime

class SubQuestion(BaseModel):
    id: str
    text: str
    tier: Literal["foundational", "dependent"]

class Source(BaseModel):
    id: str
    url: str
    title: str
    raw_text: str
    fetched_at: datetime

class Claim(BaseModel):
    id: str
    sub_question_id: str
    text: str # one atomic fact
    source_id: str
    quote: str # must quote-match source.raw_text (normalized)
    verification_status: Literal[
        "supported", "contradicted", "unresolved", "insufficient_evidence", "rejected"
    ] = "unresolved"
    confidence: Literal["high", "medium", "low"] | None = None # set AFTER status, never mixed with it

class Conflict(BaseModel):
    id: str
    claim_a: str
    claim_b: str
    reason: str

class ResearchState(BaseModel):
    run_id: str
    question: str
    sub_questions: list[SubQuestion] = []
    source_ids: list[str] = [] # not full Source objects — keep raw text out of hot state
    claims: list[Claim] = []
    conflicts: list[Conflict] = []
    trace: list[dict] = []
