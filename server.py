"""Minimal API server to trigger research runs and return results.

Run: uvicorn server:app --reload --port 8001
"""
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Ensure src/ is on sys.path for module imports when running as script
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.append(str(SRC))

# Load environment variables (API keys, etc.)
load_dotenv(dotenv_path=ROOT / ".env", override=False)

from open_deep_research.deep_researcher import deep_researcher

app = FastAPI(title="DeepResearch API", version="0.1.0")

# Allow local static page (http.server 8000) to call API on 8001
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str


class ResearchStatus(BaseModel):
    status: str
    error: str | None = None


# In-memory run registry (simple local dev use)
runs: Dict[str, Dict[str, str]] = {}


async def run_research(run_id: str, query: str):
    try:
        runs[run_id]["status"] = "running"
        # Invoke the graph with a single user message
        result = await deep_researcher.ainvoke({"messages": [HumanMessage(content=query)]})
        runs[run_id]["status"] = "done"
        runs[run_id]["final_report"] = str(result.get("final_report", ""))
    except Exception as e:  # pragma: no cover - runtime guard
        runs[run_id]["status"] = "error"
        runs[run_id]["error"] = str(e)


@app.post("/api/research", response_model=ResearchStatus)
async def start_research(req: ResearchRequest, tasks: BackgroundTasks):
    run_id = str(uuid.uuid4())
    runs[run_id] = {"status": "pending", "query": req.query}
    tasks.add_task(run_research, run_id, req.query)
    return ResearchStatus(status=run_id)


@app.get("/api/research/{run_id}/status", response_model=ResearchStatus)
async def get_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    data = runs[run_id]
    return ResearchStatus(status=data.get("status", "unknown"), error=data.get("error"))


@app.get("/api/research/{run_id}/result")
async def get_result(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    data = runs[run_id]
    if data.get("status") != "done":
        raise HTTPException(status_code=400, detail="Run not complete")

    # Load persisted files if present
    runs_dir = os.path.join(os.getcwd(), "runs")
    acemap_path = os.path.join(runs_dir, "acemap_results.json")
    final_report_path = os.path.join(runs_dir, "final_report.md")

    acemap_data = []
    try:
        with open(acemap_path, "r", encoding="utf-8") as f:
            acemap_data = json.load(f)
    except Exception:
        pass

    final_report = data.get("final_report", "")
    if not final_report and os.path.exists(final_report_path):
        try:
            with open(final_report_path, "r", encoding="utf-8") as f:
                final_report = f.read()
        except Exception:
            final_report = ""

    return {
        "final_report": final_report,
        "acemap_results": acemap_data,
    }


@app.get("/")
async def root():
    return {"message": "DeepResearch API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
