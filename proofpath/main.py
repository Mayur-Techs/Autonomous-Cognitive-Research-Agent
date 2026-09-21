from fastapi import FastAPI
from pydantic import BaseModel
from orchestrator import Orchestrator

app = FastAPI(title="ProofPath v2")
orchestrator = Orchestrator()

class ResearchRequest(BaseModel):
    question: str

@app.post("/research")
async def start_research(request: ResearchRequest):
    # Stub route: wire up the orchestrator
    state = await orchestrator.run_research(request.question)
    return {"status": "started", "run_id": state.run_id, "trace": state.trace}

@app.get("/research/{run_id}/events")
async def get_events(run_id: str):
    # Stub route: Server-Sent Events (SSE) trace emission
    return {"message": "SSE endpoint stub"}
