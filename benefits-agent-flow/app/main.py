"""FastAPI surface: POST /turn, POST /confirm, GET /trace/{conversation_id}, and the chat UI."""
import os
import pathlib
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

from app import audit
from app.graph_build import build_graph
from app.registry import versions
from app.state import TurnState
from app.tracing import capture_run, run_config

app = FastAPI(title="benefits-agent-flow")
GRAPH = build_graph(os.environ.get("CHECKPOINT_PATH", "var/checkpoints.sqlite"))

RUN_URLS: dict[str, str] = {}

STATIC = pathlib.Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


class TurnRequest(BaseModel):
    conversationId: str
    tenantId: str
    participantRef: str
    utterance: str
    uiContext: dict = {}


class ConfirmRequest(BaseModel):
    conversationId: str
    proposalId: str
    nonce: str


def _config(conversation_id: str, tenant_id: str) -> dict:
    return run_config(conversation_id, tenant_id, versions())


def pending(config: dict) -> dict | None:
    """A thread parked at the confirmation gate: the next utterance must not resume it."""
    snapshot = GRAPH.get_state(config)
    if not snapshot.next:
        return None
    proposal = (snapshot.values or {}).get("proposal")
    return proposal.model_dump() if hasattr(proposal, "model_dump") else proposal


@app.post("/turn")
def turn(req: TurnRequest) -> dict:
    config = _config(req.conversationId, req.tenantId)
    parked = pending(config)
    if parked is not None:
        raise HTTPException(status_code=409, detail={
            "kind": "pending_confirmation",
            "message": "A proposal is awaiting confirmation on this conversation. "
                       "Confirm or let it expire before starting another turn.",
            "proposal_id": parked["proposal_id"],
            "expires_at": parked["expires_at"],
        })

    state = TurnState(
        conversation_id=req.conversationId,
        turn_id=uuid.uuid4().hex[:12],
        tenant_id=req.tenantId,
        participant_ref=req.participantRef,
        utterance=req.utterance,
        ui_context=req.uiContext,
        clarify_rounds=int(req.uiContext.get("clarify_rounds", 0)),
    )
    with capture_run() as run:
        result = GRAPH.invoke(state, config)
    if run.url:
        RUN_URLS[req.conversationId] = run.url
    return result.get("response") or {"kind": "answer", "text": "", "citations": []}


@app.post("/confirm")
def confirm(req: ConfirmRequest) -> dict:
    config = _config(req.conversationId, "")
    snapshot = GRAPH.get_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="no proposal is awaiting confirmation")
    try:
        with capture_run() as run:
            result = GRAPH.invoke(
                Command(resume={"proposal_id": req.proposalId, "nonce": req.nonce}), config
            )
        if run.url:
            RUN_URLS[req.conversationId] = run.url
    except ValueError as exc:  # nonce mismatch, wrong proposal, or expiry, raised in-graph
        raise HTTPException(status_code=409, detail=str(exc))
    response = result.get("response") or {}
    if response.get("kind") != "receipt":
        raise HTTPException(status_code=409, detail=response)
    return response


@app.get("/trace/{conversation_id}")
def trace(conversation_id: str) -> dict:
    return {
        "conversation_id": conversation_id,
        "events": audit.read(conversation_id),
        "langsmith_run_url": RUN_URLS.get(conversation_id),
        "versions": versions(),
    }
