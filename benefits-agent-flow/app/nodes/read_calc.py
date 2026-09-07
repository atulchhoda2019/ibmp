"""Deterministic calculators. Every number the answer may quote originates here or in ODL."""
from app.audit import node_event
from app.mocks import calculator
from app.state import EnvelopeItem, TurnState


def _match_projection(state: TurnState) -> dict:
    rate = state.intent.slots.get("rate") or "0.06"
    return calculator.match_projection(state.participant_ref, rate)


def _pt_cost(state: TurnState) -> dict:
    return calculator.pt_cost_estimate(state.participant_ref, int(state.intent.slots.get("visits", "1")))


CALCS = {
    "limit_room": lambda state: calculator.limit_room(state.participant_ref),
    "match_projection": _match_projection,
    "pt_cost": _pt_cost,
    "deductible": lambda state: calculator.deductible_status(state.participant_ref),
}


def requested(state: TurnState) -> dict[str, str]:
    out = {}
    for spec in state.plan.required_evidence:
        etype, _, policy = spec.partition(":")
        if etype in CALCS:
            out[etype] = policy or "fp_session"
    return out


def run(state: TurnState) -> dict:
    items = []
    for n, (etype, policy) in enumerate(sorted(requested(state).items()), start=1):
        result = CALCS[etype](state)
        items.append(EnvelopeItem(
            item_id=f"tmp-c{n}",
            kind="calc",
            source=result["source"],
            effective_from=state.service_date,
            effective_to=None,
            observed_at=result["observed_at"],
            freshness_policy=policy,
            evidence_type=etype,
            payload=result["payload"],
        ))
    node_event(state, "read_calc", items=len(items), types=[i.evidence_type for i in items])
    return {"calc_items": items}
