"""Join the three parallel reads into one envelope and enforce freshness.

Stale optional evidence is dropped and disclosed; missing or stale required evidence is
refetched exactly once and then abstains. Item ids are assigned here, after the join, so
citation handles are stable regardless of the order the parallel reads finished in.
"""
from datetime import datetime

from app.audit import node_event
from app.nodes import read_calc, read_evidence, read_facts
from app.registry import freshness
from app.state import EnvelopeItem, TurnState


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def required_types(state: TurnState) -> dict[str, str]:
    out = {}
    for spec in state.plan.required_evidence:
        etype, _, policy = spec.partition(":")
        out[etype] = policy or "fp_live"
    return out


def fresh(item: EnvelopeItem, state: TurnState) -> bool:
    policy = freshness()["policies"].get(item.freshness_policy)
    if policy is None:
        return True
    if policy["effective_on_service_date"]:
        if item.effective_from > state.service_date:
            return False
        return item.effective_to is None or item.effective_to >= state.service_date
    max_age = policy["max_age_s"]
    if max_age is None:
        return True
    if max_age == 0:
        return _parse(item.observed_at) >= _parse(state.turn_started_at)
    return (_parse(state.turn_started_at) - _parse(item.observed_at)).total_seconds() <= max_age


def refetch(state: TurnState) -> tuple[list[EnvelopeItem], list[str]]:
    items: list[EnvelopeItem] = []
    items += read_evidence.run(state).get("evidence_items", [])
    items += read_facts.run(state).get("fact_items", [])
    calc = read_calc.run(state)
    items += calc.get("calc_items", [])
    return items, calc.get("not_applicable", [])


def run(state: TurnState) -> dict:
    items = [*state.evidence_items, *state.fact_items, *state.calc_items]
    required = required_types(state)
    attempts = state.retries.get("assemble", 0)

    kept, disclosures = _partition(items, state)
    missing = [t for t in required if not any(i.evidence_type == t for i in kept)]

    not_applicable = list(state.not_applicable)

    if missing and attempts < 1:
        node_event(state, "assemble", refetch=True, missing=missing)
        refetched, not_applicable = refetch(state)
        kept, disclosures = _partition(refetched, state)
        missing = [t for t in required if not any(i.evidence_type == t for i in kept)]
        attempts += 1

    if missing:
        node_event(state, "assemble", abstain=True, missing=missing, not_applicable=not_applicable)
        # Nothing on file is a different answer from retrieval failing: say which one it was.
        inapplicable = [t for t in missing if t in not_applicable]
        if inapplicable and len(inapplicable) == len(missing):
            text = ("Your coverage on file does not include the data this question needs, "
                    "so there is nothing for me to report. A specialist can confirm your coverage.")
            reason = f"not applicable to this participant: {', '.join(sorted(inapplicable))}"
        else:
            text = ("I could not retrieve the plan information this answer depends on, so I will not guess. "
                    "A specialist can pick this up with the same context.")
            reason = f"missing required evidence: {', '.join(sorted(missing))}"
        return {
            "envelope": [],
            "retries": {**state.retries, "assemble": attempts},
            "abstain_reason": reason,
            "response": {
                "kind": "answer",
                "text": text,
                "citations": [],
                "limitation": reason,
            },
        }

    envelope = [
        item.model_copy(update={"item_id": f"ev-{n}"})
        for n, item in enumerate(
            sorted(kept, key=lambda i: ({"policy": 0, "fact": 1, "calc": 2}[i.kind], i.source)), start=1
        )
    ]
    node_event(state, "assemble", items=len(envelope), dropped=len(disclosures))
    return {
        "envelope": envelope,
        "retries": {**state.retries, "assemble": attempts},
        "validation": {"disclosures": disclosures} if disclosures else None,
    }


def _partition(items: list[EnvelopeItem], state: TurnState) -> tuple[list[EnvelopeItem], list[str]]:
    kept, disclosures = [], []
    for item in items:
        if fresh(item, state):
            kept.append(item)
        else:
            policy = freshness()["policies"].get(item.freshness_policy, {})
            if policy.get("on_violation") == "drop_and_disclose":
                disclosures.append(f"{item.evidence_type} was stale and has been left out")
    return kept, disclosures
