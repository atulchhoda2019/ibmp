"""Compose a draft from the envelope only. The gateway is a mock; it creates no facts."""
from app.audit import node_event
from app.mocks import model_gateway
from app.state import TurnState


def run(state: TurnState) -> dict:
    attempts = state.retries.get("reason", 0)
    envelope = [item.model_dump() for item in state.envelope]
    try:
        result = model_gateway.compose(state.plan.graph_id, envelope, state.intent.name)
    except model_gateway.ModelTimeout:
        node_event(state, "reason", timeout=True, attempt=attempts)
        return {
            "draft": None,
            "retries": {**state.retries, "reason": attempts + 1},
            "validation": {"passed": False, "reasons": ["model_timeout"]},
        }
    node_event(state, "reason", attempt=attempts, chars=len(result["text"]), model=result["model"])
    return {"draft": result["text"], "retries": {**state.retries, "reason": attempts + 1}}
