"""Governance gate 1: capability bundle, rung ceiling, posture.

The effective rung is min(tenant rung, table rung_max). Rung 1 drafts, rung 2 previews and
waits for a nonce, rung 3 and 4 run the corridor without pausing but keep revalidation,
idempotency, read-after-write verification and the receipt.
"""
from app.audit import node_event
from app.mocks import odl
from app.registry import bundles
from app.state import TurnState


def bundle_for(tenant_id: str) -> dict:
    tenants = bundles()["tenants"]
    if tenant_id not in tenants:
        raise KeyError(f"no capability bundle for tenant {tenant_id}")
    return tenants[tenant_id]


def effective_rung(tenant_id: str, intent: str, rung_max: int) -> int:
    tenant_rung = bundle_for(tenant_id)["rungs"].get(intent, 0)
    return min(tenant_rung, rung_max)


def run(state: TurnState) -> dict:
    plan = state.plan
    intent = state.intent
    bundle = bundle_for(state.tenant_id)

    if odl.participant(state.participant_ref)["tenant_id"] != state.tenant_id:
        node_event(state, "gate_plan", allowed=False, reason="tenant_mismatch")
        return {"response": {
            "kind": "handoff",
            "summary": "I cannot act on this record from this workspace.",
            "top_intents": [intent.name],
            "reason": "tenant_mismatch",
        }}

    if plan.requires_capability and plan.requires_capability not in bundle["capabilities"]:
        node_event(state, "gate_plan", allowed=False, reason="capability_not_purchased")
        return {"response": {
            "kind": "handoff",
            "summary": "That is not enabled for your plan sponsor, so I am handing this to a specialist.",
            "top_intents": [intent.name],
            "reason": "capability_not_purchased",
        }}

    updates: dict = {}
    if plan.posture == "WRITE":
        rung = effective_rung(state.tenant_id, intent.name, plan.rung)
        if rung == 0:
            node_event(state, "gate_plan", allowed=False, reason="rung_zero")
            return {"response": {
                "kind": "handoff",
                "summary": "Changes to this election are not available here; a specialist can make it for you.",
                "top_intents": [intent.name],
                "reason": "rung_zero",
            }}
        posture = "DRAFT" if rung == 1 else "WRITE"
        updates["plan"] = plan.model_copy(update={"rung": rung, "posture": posture})

    node_event(state, "gate_plan", allowed=True,
               rung=updates.get("plan", plan).rung, posture=updates.get("plan", plan).posture)
    return updates
