"""
AIOps Copilot - REST API
Wraps the LangGraph agent as a FastAPI service.

Endpoints:
    POST /investigate        - Start a new investigation
    GET  /status/{thread_id} - Check investigation status
    POST /approve/{thread_id} - Approve or deny a pending remediation
"""
import langfuse as langfuse_sdk
import uuid
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from graph import graph, get_langfuse_handler
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AIOps Copilot API",
    description="Multi-agent LLM system for autonomous Kubernetes incident investigation and remediation",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# IN-MEMORY STATE (replace with Postgres in production)
# ─────────────────────────────────────────────

investigations = {}  # thread_id -> investigation state


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class InvestigateRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None  # provide to resume an existing investigation


class InvestigateResponse(BaseModel):
    thread_id: str
    status: str  # "running" | "awaiting_approval" | "completed" | "failed"
    message: str


class ApproveRequest(BaseModel):
    decision: str  # "approve" or "deny"


class StatusResponse(BaseModel):
    thread_id: str
    status: str
    question: str
    final_answer: Optional[str] = None
    pending_action: Optional[str] = None  # what action is awaiting approval
    agents_called: list[str] = []
    error: Optional[str] = None


# ─────────────────────────────────────────────
# BACKGROUND INVESTIGATION RUNNER
# ─────────────────────────────────────────────

def run_investigation(thread_id: str, question: str, resume: bool = False, decision: str = None):
    """Runs the agent graph in the background and updates investigation state."""
    try:
        investigations[thread_id]["status"] = "running"
        langfuse_handler = get_langfuse_handler(session_id=thread_id)
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler],
        }

        if resume and decision:
            result = graph.invoke(Command(resume=decision), config=config)
        elif resume:
            result = graph.invoke(None, config=config)
        else:
            result = graph.invoke(
                {"messages": [HumanMessage(content=question)],
                 "next_agent": "", "iteration_count": 0, "agents_called": []},
                config=config,
            )

        if "__interrupt__" in result:
            interrupt_info = result["__interrupt__"][0].value
            investigations[thread_id]["status"] = "awaiting_approval"
            investigations[thread_id]["pending_action"] = interrupt_info.get("message", "Approval required")
        else:
            messages = result["messages"]
            ai_messages = [m for m in messages if isinstance(m, AIMessage) and m.content]
            final_answer = ai_messages[-1].content if ai_messages else "No answer produced."
            investigations[thread_id]["status"] = "completed"
            investigations[thread_id]["final_answer"] = final_answer
            investigations[thread_id]["agents_called"] = result.get("agents_called", [])

        

    except Exception as e:
    # Only mark as failed if investigation didn't already complete
        if investigations[thread_id]["status"] == "running":
            investigations[thread_id]["status"] = "failed"
            investigations[thread_id]["error"] = str(e)
        else:
            # Investigation completed but cleanup failed — not a real failure
            investigations[thread_id]["error"] = f"Cleanup warning: {str(e)}"


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "service": "AIOps Copilot API"}


@app.post("/investigate", response_model=InvestigateResponse)
async def investigate(request: InvestigateRequest, background_tasks: BackgroundTasks):
    """Start a new investigation or resume an existing one."""
    thread_id = request.thread_id or str(uuid.uuid4())
    resume = request.thread_id is not None

    investigations[thread_id] = {
        "status": "running",
        "question": request.question,
        "final_answer": None,
        "pending_action": None,
        "agents_called": [],
        "error": None,
    }

    background_tasks.add_task(
        run_investigation,
        thread_id=thread_id,
        question=request.question,
        resume=resume,
    )

    return InvestigateResponse(
        thread_id=thread_id,
        status="running",
        message=f"Investigation started. Poll /status/{thread_id} for updates.",
    )


@app.get("/status/{thread_id}", response_model=StatusResponse)
def status(thread_id: str):
    """Check the status of an investigation."""
    if thread_id not in investigations:
        raise HTTPException(status_code=404, detail=f"Investigation {thread_id} not found.")

    inv = investigations[thread_id]
    return StatusResponse(
        thread_id=thread_id,
        status=inv["status"],
        question=inv["question"],
        final_answer=inv.get("final_answer"),
        pending_action=inv.get("pending_action"),
        agents_called=inv.get("agents_called", []),
        error=inv.get("error"),
    )


@app.post("/approve/{thread_id}", response_model=InvestigateResponse)
async def approve(thread_id: str, request: ApproveRequest, background_tasks: BackgroundTasks):
    """Approve or deny a pending remediation action."""
    if thread_id not in investigations:
        raise HTTPException(status_code=404, detail=f"Investigation {thread_id} not found.")

    inv = investigations[thread_id]
    if inv["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Investigation is not awaiting approval. Current status: {inv['status']}"
        )

    if request.decision not in ("approve", "deny"):
        raise HTTPException(status_code=400, detail="Decision must be 'approve' or 'deny'.")

    investigations[thread_id]["status"] = "running"
    investigations[thread_id]["pending_action"] = None

    background_tasks.add_task(
        run_investigation,
        thread_id=thread_id,
        question=inv["question"],
        resume=True,
        decision=request.decision,
    )

    return InvestigateResponse(
        thread_id=thread_id,
        status="running",
        message=f"Decision '{request.decision}' submitted. Poll /status/{thread_id} for updates.",
    )


@app.get("/investigations")
def list_investigations():
    """List all investigations and their current status."""
    return {
        "total": len(investigations),
        "investigations": [
            {
                "thread_id": tid,
                "status": inv["status"],
                "question": inv["question"][:80],
            }
            for tid, inv in investigations.items()
        ]
    }