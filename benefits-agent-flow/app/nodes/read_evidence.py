"""Policy evidence read. Filter first on tenant + plan + service date, rerank inside that set."""
from app.audit import node_event
from app.mocks import evidence, odl
from app.state import EnvelopeItem, TurnState

EVIDENCE_TYPE = "plan_rules"


def required_policy(state: TurnState) -> str | None:
    for spec in state.plan.required_evidence:
        etype, _, policy = spec.partition(":")
        if etype == EVIDENCE_TYPE:
            return policy or "fp_current"
    return None


def plan_ids(participant_ref: str) -> list[str]:
    record = odl.participant(participant_ref)
    return [p for p in (record["plan_id"], record["medical_plan_id"]) if p]


def fetch(state: TurnState, policy: str) -> list[EnvelopeItem]:
    result = evidence.retrieve(
        tenant_id=state.tenant_id,
        plan_ids=plan_ids(state.participant_ref),
        service_date=state.service_date,
        query=state.utterance,
    )
    return [
        EnvelopeItem(
            item_id=f"tmp-p{n}",
            kind="policy",
            source=f"{result['source']}#{passage['passage_id']}@{passage['version']}",
            effective_from=passage["effective_from"],
            effective_to=passage["effective_to"],
            observed_at=result["observed_at"],
            freshness_policy=policy,
            evidence_type=EVIDENCE_TYPE,
            payload={"text": passage["text"], "passage_id": passage["passage_id"]},
        )
        for n, passage in enumerate(result["payload"], start=1)
    ]


def run(state: TurnState) -> dict:
    policy = required_policy(state)
    if policy is None:
        return {"evidence_items": []}
    items = fetch(state, policy)
    node_event(state, "read_evidence", items=len(items),
               passages=[i.payload["passage_id"] for i in items])
    return {"evidence_items": items}
