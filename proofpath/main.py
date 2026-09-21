import json
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from orchestrator import Orchestrator
from agents.retrieval import get_provider_health

app = FastAPI(title="ProofPath v2", description="Evidence-First Autonomous Research System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

orchestrator = Orchestrator()


# ── Request models ──────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    question: str
    depth: str = "deep"
    recency: str = "10y"
    evidence_standard: str = "balanced"


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/research")
async def start_research(request: ResearchRequest):
    """Start a new research run. Returns run_id immediately."""
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    asyncio.create_task(orchestrator.run_research(request.question, run_id))
    return {"status": "started", "run_id": run_id}


@app.get("/api/research/{run_id}/events")
async def get_events(run_id: str):
    """Server-Sent Events stream for live trace updates."""
    # Queue may not exist yet if the task just started — wait briefly
    for _ in range(10):
        queue = orchestrator.run_queues.get(run_id)
        if queue:
            break
        await asyncio.sleep(0.2)
    else:
        raise HTTPException(status_code=404, detail="Run not found or not started yet")

    async def event_generator():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("event") == "done":
                await asyncio.sleep(0.3)
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/research/{run_id}")
async def get_run_state(run_id: str):
    """Return the completed state for a run (after it finishes)."""
    state = orchestrator.completed_states.get(run_id)
    if not state:
        return {"status": "in_progress", "run_id": run_id}
    return {
        "run_id": run_id,
        "question": state.question,
        "stop_reason": state.stop_reason,
        "metrics": state.metrics.model_dump(),
        "subquestions": [sq.model_dump() for sq in state.sub_questions],
    }


@app.get("/api/research/{run_id}/claims")
async def get_claims(run_id: str):
    """Return all claims for a run with their verification status."""
    state = orchestrator.completed_states.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found or still in progress")
    return {
        "run_id": run_id,
        "total": len(state.claims),
        "claims": [c.model_dump() for c in state.claims],
    }


@app.get("/api/research/{run_id}/sources")
async def get_sources(run_id: str):
    """Return all retrieved sources for a run (metadata only, no raw text)."""
    state = orchestrator.completed_states.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found or still in progress")
    sources_meta = []
    for sid in state.source_ids:
        src = orchestrator.sources.get(sid)
        if src:
            sources_meta.append({
                "id": src.id,
                "title": src.title,
                "url": src.url,
                "source_type": src.source_type,
                "accessed_at": src.accessed_at.isoformat(),
            })
    return {"run_id": run_id, "total": len(sources_meta), "sources": sources_meta}


@app.get("/api/research/{run_id}/report")
async def get_report(run_id: str):
    """Return the final compiled report."""
    state = orchestrator.completed_states.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found or still in progress")
    return {
        "run_id": run_id,
        "report": state.final_report,
        "stop_reason": state.stop_reason,
        "metrics": state.metrics.model_dump(),
    }


@app.get("/api/research/{run_id}/conflicts")
async def get_conflicts(run_id: str):
    """Return all detected conflicts for a run."""
    state = orchestrator.completed_states.get(run_id)
    if not state:
        raise HTTPException(status_code=404, detail="Run not found or still in progress")
    return {
        "run_id": run_id,
        "total": len(state.conflicts),
        "conflicts": [c.model_dump() for c in state.conflicts],
    }


@app.get("/api/health")
async def health():
    """Provider health and system status."""
    return {
        "status": "ok",
        "providers": {"tavily": get_provider_health()},
        "active_runs": len(orchestrator.run_queues),
        "completed_runs": len(orchestrator.completed_states),
    }
