"""FastAPI surface: POST /turn, POST /confirm, GET /trace/{conversation_id}."""
import os
import uuid

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel

from app import audit
from app.graph_build import build_graph
from app.registry import versions
from app.state import TurnState
from app.tracing import run_config, run_url

app = FastAPI(title="benefits-agent-flow")
GRAPH = build_graph(os.environ.get("CHECKPOINT_PATH", "var/checkpoints.sqlite"))


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
    result = GRAPH.invoke(state, config)
    return result.get("response") or {"kind": "answer", "text": "", "citations": []}


@app.post("/confirm")
def confirm(req: ConfirmRequest) -> dict:
    config = _config(req.conversationId, "")
    snapshot = GRAPH.get_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="no proposal is awaiting confirmation")
    try:
        result = GRAPH.invoke(
            Command(resume={"proposal_id": req.proposalId, "nonce": req.nonce}), config
        )
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
        "langsmith_run_url": run_url(),
        "versions": versions(),
    }
