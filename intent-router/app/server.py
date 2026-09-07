"""Demo assistant API over the routing engine.

The UI is a consumer of the planner, never a second planner: it posts an utterance plus the
context knobs, and renders whatever graph the table selected. Confirmations are nonce-bound,
and the "execution" behind them is stubbed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.responder import respond  # noqa: E402
from classifier import StubClassifier  # noqa: E402
from graphs.registry import load_registry  # noqa: E402
from router.catalog import load_catalog  # noqa: E402
from router.engine import IntentRouter  # noqa: E402
from router.models import RequestContext, RoutingError  # noqa: E402
from router.table import load_table  # noqa: E402
from router.trace import TraceCollector  # noqa: E402

CATALOG = load_catalog(APP_ROOT / "catalog" / "v1.yaml")
TABLE = load_table(APP_ROOT / "table" / "v1.yaml")
REGISTRY = load_registry(APP_ROOT / "graphs" / "registry.yaml")
TRACE = TraceCollector()
ROUTER = IntentRouter(
    catalog=CATALOG,
    table=TABLE,
    registry=REGISTRY,
    classifier=StubClassifier(catalog=CATALOG),
    trace_sink=TRACE,
)

CAPABILITIES = {
    "enabled": {"ret.contribution.change": {"enabled": True, "rung": 2}},
    "disabled": {"ret.contribution.change": {"enabled": False}},
}

app = FastAPI(title="Intent routing demo assistant")


class ContextPayload(BaseModel):
    capability: Literal["enabled", "disabled"] = "enabled"
    auth_level: Literal["standard", "stepped_up"] = "standard"
    tenant_frozen: bool = False
    viewing_plan: str = "PPO-High"
    participant_ref: str = "participant-1"
    conversation_state: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    utterance: str
    context: ContextPayload = Field(default_factory=ContextPayload)


class ConfirmRequest(BaseModel):
    proposal_id: str
    confirmation_nonce: str


class ChatResponse(BaseModel):
    assistant: str
    options: List[Dict[str, str]]
    proposal: Optional[Dict[str, Any]]
    tool_calls: List[str]
    decision: Dict[str, Any]
    conversation_state: Dict[str, Any]


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    context = RequestContext(
        tenant_id="tenant-acme",
        participant_ref=request.context.participant_ref,
        auth_level=request.context.auth_level,
        tenant_capabilities=CAPABILITIES[request.context.capability],
        tenant_frozen=request.context.tenant_frozen,
        ui_context={"viewing_plan": request.context.viewing_plan},
        conversation_state=request.context.conversation_state,
    )
    try:
        decision = ROUTER.route(request.utterance, context)
    except RoutingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    turn = await respond(decision, REGISTRY)
    return ChatResponse(
        assistant=turn.text,
        options=turn.options,
        proposal=turn.proposal,
        tool_calls=turn.tool_calls,
        decision={
            "graph": decision.graph,
            "entry_node": decision.entry_node,
            "intent": decision.intent,
            "band": decision.band,
            "source": decision.source,
            "confidence": decision.confidence,
            "fired_row": decision.fired_row,
            "budgets": dict(decision.budgets),
            "ladder_path": list(decision.ladder_path),
            "slots": dict(decision.slots),
            "missing_slots": list(decision.missing_slots),
            "cache_key": decision.cache_key,
            "versions": dict(decision.versions),
            "note": decision.note,
            "log_for_catalog_review": decision.log_for_catalog_review,
        },
        conversation_state=dict(decision.conversation_state),
    )


@app.post("/api/confirm")
async def confirm(request: ConfirmRequest) -> Dict[str, str]:
    """Stubbed downstream: a proposal only moves when its own nonce comes back."""
    if not request.confirmation_nonce:
        raise HTTPException(status_code=400, detail="confirmation nonce required")
    return {
        "status": "accepted",
        "detail": f"Proposal {request.proposal_id} confirmed. Execution is out of scope in this demo.",
    }


@app.get("/api/table")
async def table() -> Dict[str, Any]:
    return {
        "versions": {"catalog": CATALOG.version, "table": TABLE.version},
        "rows": [
            {
                "index": row.index,
                "intent": row.intent,
                "band": row.band,
                "capability": row.capability,
                "risk": row.risk,
                "auth": row.auth,
                "graph": row.graph,
                "budgets": dict(row.budgets),
                "note": row.note,
            }
            for row in TABLE.rows
        ],
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")
