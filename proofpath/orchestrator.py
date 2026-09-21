import asyncio
import time
from datetime import datetime, timezone
from models import ResearchState, CoverageMetrics, ResearchBudget, EvidenceSpan
from agents.planner import generate_plan
from agents.retrieval import retrieve_concurrently
from agents.analysis import analyze_sources_concurrently
from agents.fact_checker import fact_check_concurrently, detect_contradictions, compute_confidence
from agents.synthesis import synthesize_final_report


class Orchestrator:
    def __init__(self):
        # Full Source objects keyed by id — kept OUT of ResearchState
        self.sources: dict = {}
        # EvidenceSpan objects keyed by id
        self.evidence_spans: dict = {}
        # SSE queues keyed by run_id
        self.run_queues: dict = {}
        # Completed states keyed by run_id
        self.completed_states: dict = {}

    async def run_research(self, question: str, run_id: str) -> ResearchState:
        state = ResearchState(run_id=run_id, question=question)
        start_time = time.time()

        queue: asyncio.Queue = asyncio.Queue()
        self.run_queues[run_id] = queue

        async def emit(msg: str, extra: dict | None = None):
            elapsed = round(time.time() - start_time, 1)
            event = {
                "event": msg,
                "elapsed_s": elapsed,
                "ts": datetime.now(timezone.utc).isoformat()
            }
            if extra:
                event.update(extra)
            state.trace.append(event)
            await queue.put(event)

        # ── 1. Planner ──────────────────────────────────────────────────────
        await emit("Planner started")
        try:
            sub_qs = await generate_plan(question)
            state.budget.llm_calls_used += 1
        except Exception as exc:
            await emit(f"Planner failed: {exc}")
            state.stop_reason = "planner_failed"
            await queue.put({"event": "done", "report": "Research could not be planned.", "state": state.model_dump()})
            self.completed_states[run_id] = state
            return state

        state.sub_questions = sub_qs
        for sq in state.sub_questions:
            sq.status = "ready"

        await emit(
            f"Planner finished: {len(sub_qs)} sub-questions",
            {"subquestions": [sq.model_dump() for sq in sub_qs]}
        )

        tier1 = [sq for sq in sub_qs if sq.tier == "foundational"]
        tier2 = [sq for sq in sub_qs if sq.tier == "dependent"]

        # ── 2. Process each tier ────────────────────────────────────────────
        async def process_tier(tier_name: str, qs: list):
            if not qs:
                return
            if state.budget.exhausted():
                state.stop_reason = state.budget.stop_reason()
                await emit(f"Budget exhausted before {tier_name}")
                return

            for sq in qs:
                sq.status = "researching"

            await emit(f"{tier_name}: Retrieval started ({len(qs)} questions)")
            queries = [q.text for q in qs]
            sources_per_q = await retrieve_concurrently(queries)
            state.budget.searches_used += len(queries)

            for sq, sources in zip(qs, sources_per_q):
                for s in sources:
                    self.sources[s.id] = s
                    if s.id not in state.source_ids:
                        state.source_ids.append(s.id)
                        state.budget.sources_found += 1

            await emit(f"{tier_name}: {state.budget.sources_found} unique sources so far")

            await emit(f"{tier_name}: Extracting claims and evidence spans")
            analysis_tasks = [
                analyze_sources_concurrently(sq, sources)
                for sq, sources in zip(qs, sources_per_q)
            ]
            results_per_q = await asyncio.gather(*analysis_tasks)
            state.budget.llm_calls_used += len(qs)

            all_tier_claims = []
            for (claims, spans) in results_per_q:
                all_tier_claims.extend(claims)
                for span in spans:
                    self.evidence_spans[span.id] = span
                    state.evidence_span_ids.append(span.id)

            await emit(f"{tier_name}: Fact-checking {len(all_tier_claims)} claims")
            if all_tier_claims:
                fact_checked = await fact_check_concurrently(all_tier_claims, self.sources)
                state.budget.llm_calls_used += len(all_tier_claims)
            else:
                fact_checked = []

            for c in fact_checked:
                c.confidence = compute_confidence({c.verification_status: 1})
                state.claims.append(c)

            await emit(f"{tier_name}: Detecting contradictions")
            contradiction_tasks = [
                detect_contradictions(
                    [c for c in fact_checked if c.sub_question_id == sq.id],
                    self.sources
                )
                for sq in qs
            ]
            conflicts_per_q = await asyncio.gather(*contradiction_tasks)
            state.budget.llm_calls_used += len(qs)

            for conf_list in conflicts_per_q:
                state.conflicts.extend(conf_list)

            # Update subquestion status
            for sq in qs:
                sq_claims = [c for c in fact_checked if c.sub_question_id == sq.id]
                if not sq_claims:
                    sq.status = "insufficient_evidence"
                elif any(c.verification_status == "supported" for c in sq_claims):
                    sq.status = "answered"
                else:
                    sq.status = "insufficient_evidence"

        await process_tier("Tier 1 (foundational)", tier1)

        if not state.budget.exhausted():
            for sq in tier2:
                sq.status = "ready"
            await process_tier("Tier 2 (dependent)", tier2)
        else:
            for sq in tier2:
                sq.status = "blocked"
            await emit("Tier 2 blocked — budget exhausted")

        # ── 3. Coverage metrics ─────────────────────────────────────────────
        state.metrics = self._compute_metrics(state)
        await emit(
            f"Coverage: {state.metrics.coverage_pct}% "
            f"({state.metrics.answered}/{state.metrics.total_subquestions} answered, "
            f"{state.metrics.insufficient_evidence} insufficient evidence)",
            {"metrics": state.metrics.model_dump()}
        )

        # ── 4. Stop reason ──────────────────────────────────────────────────
        if not state.stop_reason:
            if state.budget.exhausted():
                state.stop_reason = state.budget.stop_reason()
            elif state.metrics.coverage_pct >= 80:
                state.stop_reason = "coverage_threshold_reached"
            else:
                state.stop_reason = "research_complete"

        await emit(f"Stop reason: {state.stop_reason}")

        # ── 5. Synthesis ─────────────────────────────────────────────────────
        await emit("Synthesis started")
        state.__dict__["sources"] = self.sources
        final_report = await synthesize_final_report(state)
        state.budget.llm_calls_used += len(state.sub_questions)
        state.final_report = final_report

        # ── 6. Done ──────────────────────────────────────────────────────────
        await emit("Research complete")
        self.completed_states[run_id] = state
        await queue.put({
            "event": "done",
            "report": final_report,
            "stop_reason": state.stop_reason,
            "metrics": state.metrics.model_dump(),
        })

        return state

    def _compute_metrics(self, state: ResearchState) -> CoverageMetrics:
        answered = sum(1 for sq in state.sub_questions if sq.status == "answered")
        insuff = sum(1 for sq in state.sub_questions if sq.status == "insufficient_evidence")
        return CoverageMetrics(
            total_subquestions=len(state.sub_questions),
            answered=answered,
            insufficient_evidence=insuff,
            supported_claims=sum(1 for c in state.claims if c.verification_status == "supported"),
            contradicted_claims=sum(1 for c in state.claims if c.verification_status == "contradicted"),
            unresolved_claims=sum(1 for c in state.claims if c.verification_status == "unresolved"),
            rejected_claims=sum(1 for c in state.claims if c.verification_status == "rejected"),
            total_sources=len(state.source_ids),
            total_conflicts=len(state.conflicts),
            llm_calls=state.budget.llm_calls_used,
            searches=state.budget.searches_used,
        )
