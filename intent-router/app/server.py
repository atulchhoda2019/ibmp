"""Demo assistant API over the routing engine.

The UI is a consumer of the planner, never a second planner: it posts an utterance plus the
context knobs, and renders whatever graph the table selected. Confirmations are nonce-bound,
and the "execution" behind them is stubbed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.corridor import ConfirmationError, ConfirmationStore  # noqa: E402
from app.responder import execute_pending, respond  # noqa: E402
from classifier import StubClassifier  # noqa: E402
from classifier.model import ModelClassifier  # noqa: E402
from graphs.registry import load_registry  # noqa: E402
from modelplane.gateway import Scorecard, backend_from_env, build_gateway  # noqa: E402
from router.catalog import load_catalog  # noqa: E402
from router.engine import IntentRouter  # noqa: E402
from router.models import RequestContext, RoutingError  # noqa: E402
from router.table import load_table  # noqa: E402
from router.trace import TraceCollector  # noqa: E402

CATALOG = load_catalog(APP_ROOT / "catalog" / "v1.yaml")
TABLE = load_table(APP_ROOT / "table" / "v1.yaml")
REGISTRY = load_registry(APP_ROOT / "graphs" / "registry.yaml")
TRACE = TraceCollector()

#: fake backend unless MODEL_BACKEND=openai; the composer never invents a fact either way
GATEWAY = build_gateway(backend=backend_from_env())
ROUTER = IntentRouter(
    catalog=CATALOG,
    table=TABLE,
    registry=REGISTRY,
    classifier=(
        ModelClassifier(gateway=GATEWAY, catalog=CATALOG)
        if os.environ.get("CLASSIFIER", "stub").lower() == "model"
        else StubClassifier(catalog=CATALOG)
    ),
    trace_sink=TRACE,
)
SCORECARD = Scorecard()

TENANT_ID = "tenant-acme"
#: the confirmation corridor lives server-side: the client never holds a nonce it can replay
STORE = ConfirmationStore()

app = FastAPI(title="Intent routing demo assistant")


def _capabilities(capability: str, rung: int) -> Dict[str, Any]:
    if capability == "disabled":
        return {"ret.contribution.change": {"enabled": False}}
    return {"ret.contribution.change": {"enabled": True, "rung": rung}}


class ContextPayload(BaseModel):
    capability: Literal["enabled", "disabled"] = "enabled"
    #: autonomy rung the tenant bought: 1 draft, 2 propose+confirm, 3 execute+notify, 4 autonomous
    rung: int = Field(default=2, ge=1, le=4)
    auth_level: Literal["standard", "stepped_up"] = "standard"
    tenant_frozen: bool = False
    #: verified grants; the UI toggles them, a real gateway reads them from the identity plane
    can_view: bool = True
    can_change: bool = True
    evidence_age_days: int = Field(default=2, ge=0, le=400)
    viewing_plan: str = "PPO-High"
    participant_ref: str = "participant-1"
    conversation_state: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    utterance: str
    context: ContextPayload = Field(default_factory=ContextPayload)


class ConfirmRequest(BaseModel):
    proposal_id: str
    confirmation_nonce: str
    participant_ref: str = "participant-1"
    rung: int = Field(default=2, ge=1, le=4)
    conversation_state: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    assistant: str
    options: List[Dict[str, str]]
    proposal: Optional[Dict[str, Any]]
    receipt: Optional[Dict[str, Any]]
    tool_calls: List[str]
    decision: Dict[str, Any]
    conversation_state: Dict[str, Any]
    model: Optional[Dict[str, Any]] = None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    context = RequestContext(
        tenant_id=TENANT_ID,
        participant_ref=request.context.participant_ref,
        auth_level=request.context.auth_level,
        tenant_capabilities=_capabilities(request.context.capability, request.context.rung),
        tenant_frozen=request.context.tenant_frozen,
        entitlements={"VIEW": request.context.can_view, "CHANGE": request.context.can_change},
        ui_context={"viewing_plan": request.context.viewing_plan},
        conversation_state=request.context.conversation_state,
    )
    try:
        decision = ROUTER.route(request.utterance, context)
    except RoutingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    turn = await respond(
        decision,
        REGISTRY,
        context,
        STORE,
        evidence_age_days=request.context.evidence_age_days,
        gateway=GATEWAY,
    )
    SCORECARD.answers += 1
    SCORECARD.confidence_sum += decision.confidence
    if turn.model:
        SCORECARD.validator_catch += sum(
            1 for attempt in turn.model.get("attempts", ()) if not attempt["ok"]
        )
        SCORECARD.escalation += int(bool(turn.model.get("escalated")))
        SCORECARD.handoff += int(bool(turn.model.get("escalated_to_human")))
        SCORECARD.cost_usd = round(SCORECARD.cost_usd + turn.model.get("cost_usd", 0.0), 6)
    if decision.graph == "G-FALLBACK":
        SCORECARD.handoff += 1
    return ChatResponse(
        assistant=turn.text,
        options=turn.options,
        proposal=turn.proposal,
        receipt=turn.receipt,
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
            "capability": decision.capability,
            "rung": decision.rung,
            "permission": decision.permission,
            "posture": decision.posture,
            "band_edges": decision.band_edges,
            "evidence_policy": decision.evidence_policy,
        },
        conversation_state=dict(turn.conversation_state),
        model=turn.model,
    )


@app.post("/api/confirm")
async def confirm(request: ConfirmRequest) -> Dict[str, Any]:
    """The corridor: only a live, nonce-bound proposal executes, and a retry replays its receipt."""
    if not request.confirmation_nonce:
        raise HTTPException(status_code=400, detail="confirmation nonce required")
    replay = STORE.replay(request.proposal_id)
    if replay is not None:
        return {"status": "duplicate", "receipt": _receipt_payload(replay)}
    try:
        pending = STORE.claim(
            request.proposal_id,
            request.confirmation_nonce,
            tenant_id=TENANT_ID,
            participant_ref=request.participant_ref,
        )
    except ConfirmationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None

    graph = REGISTRY.instantiate("G-CONTRIB-COMMIT")
    REGISTRY.assert_manifest_legal(
        "G-CONTRIB-COMMIT", "TRANSACT", capability_enabled=True, rung=request.rung, confirmed=True
    )
    turn = await execute_pending(pending, graph, STORE, dict(request.conversation_state))
    return {
        "status": "executed",
        "assistant": turn.text,
        "receipt": turn.receipt,
        "tool_calls": turn.tool_calls,
        "conversation_state": turn.conversation_state,
    }


def _receipt_payload(receipt) -> Dict[str, Any]:
    return {
        "command_key": receipt.command_key,
        "action": receipt.action,
        "params": dict(receipt.params),
        "effective_date": receipt.effective_date,
        "executed_at": receipt.executed_at,
        "reversal": receipt.reversal,
        "duplicate": receipt.duplicate,
    }


@app.get("/api/models")
async def models() -> Dict[str, Any]:
    """What is being served, on what eval, in which ring — plus the live scorecard."""
    registry = GATEWAY.registry
    return {
        "versions": {"registry": registry.version, "routing": GATEWAY.table.version},
        "frontier": registry.frontier,
        "serving": {
            task: {"config": pointer.config, "rollback_to": pointer.rollback_to}
            for task, pointer in registry.serving.items()
        },
        "served": [
            {
                "name": config.name,
                "provider": config.provider,
                "base": config.base,
                "adapter": config.adapter,
                "scope": config.scope,
                "task": config.task,
                "ring": config.ring,
                "eval_id": config.eval.eval_id,
                "score": config.eval.score,
                "judge_family": config.eval.judge_family,
                "measured_gap": config.eval.measured_gap,
            }
            for config in registry.served.values()
        ],
        "rows": [
            {
                "index": row.index,
                "task": row.task,
                "domain": row.domain,
                "posture": row.posture,
                "tenant": row.tenant,
                "config": row.config,
            }
            for row in GATEWAY.table.rows
        ],
        "scorecard": SCORECARD.snapshot(),
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
                "evidence": dict(row.evidence) if row.evidence else None,
            }
            for row in TABLE.rows
        ],
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")
