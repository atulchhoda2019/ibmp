"""Typed ODL reads. Only the evidence types the fired table row asked for."""
from app.audit import node_event
from app.mocks import odl
from app.state import EnvelopeItem, TurnState

READERS = {
    "elections": odl.get_elections,
    "accumulators": odl.get_accumulators,
    "balances": odl.get_balances,
    "eligibility": odl.get_eligibility,
}


def requested(state: TurnState) -> dict[str, str]:
    out = {}
    for spec in state.plan.required_evidence:
        etype, _, policy = spec.partition(":")
        if etype in READERS:
            out[etype] = policy or "fp_live"
    return out


def fetch(state: TurnState, etype: str, policy: str, index: int) -> EnvelopeItem:
    result = READERS[etype](state.participant_ref)
    return EnvelopeItem(
        item_id=f"tmp-f{index}",
        kind="fact",
        source=result["source"],
        effective_from=result.get("effective_from") or state.service_date,
        effective_to=result.get("effective_to"),
        observed_at=result["observed_at"],
        freshness_policy=policy,
        evidence_type=etype,
        payload=result["payload"],
    )


def run(state: TurnState) -> dict:
    items = [
        fetch(state, etype, policy, n)
        for n, (etype, policy) in enumerate(sorted(requested(state).items()), start=1)
    ]
    node_event(state, "read_facts", items=len(items), types=[i.evidence_type for i in items])
    return {"fact_items": items}
