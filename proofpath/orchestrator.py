import asyncio
from models import ResearchState

class Orchestrator:
    def __init__(self):
        # Store full Source objects in a dict keyed by id outside the state
        # to keep raw_text out of the hot state as per section 4
        self.sources = {}

    async def run_research(self, question: str) -> ResearchState:
        # Initialize state
        state = ResearchState(run_id="run_1", question=question)
        
        # 1. Planner (produces tiered sub-questions)
        self._emit_trace(state, "Planner started")
        
        # 2. Tier 1 Retrieval + Analysis + Fact-Checking
        self._emit_trace(state, "Tier 1 (foundational) started")
        
        # 3. Tier 2 Retrieval + Analysis + Fact-Checking (depends on Tier 1)
        self._emit_trace(state, "Tier 2 (dependent) started")
        
        # 4. Contradiction Detection
        self._emit_trace(state, "Contradiction pass started")
        
        # 5. Synthesis
        self._emit_trace(state, "Synthesis started")
        
        return state

    def _emit_trace(self, state: ResearchState, message: str):
        # Append to trace for SSE emission later
        trace_event = {"event": message, "timestamp": "placeholder"}
        state.trace.append(trace_event)
