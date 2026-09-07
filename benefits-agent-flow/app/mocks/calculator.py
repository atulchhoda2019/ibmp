"""Deterministic calculators with receipts. The model never does arithmetic; it quotes these."""
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

from app.mocks import odl
from app.tracing import tool_span


def _money(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _receipt(payload: dict, source: str) -> dict:
    return {"payload": payload, "source": source, "observed_at": datetime.now(timezone.utc).isoformat()}


@tool_span("calc.limit_room", "v2")
def limit_room(participant_ref: str) -> dict:
    record = odl.participant(participant_ref)
    plan = odl.plans()[record["plan_id"]]
    limit = Decimal(plan["annual_deferral_limit"])
    used = Decimal(record["ytd_deferral"])
    return _receipt(
        {
            "annual_limit": _money(limit),
            "ytd_deferral": _money(used),
            "room_remaining": _money(limit - used),
            "inputs": ["odl.elections.v5", record["plan_id"]],
        },
        "calc.limit_room.v2",
    )


@tool_span("calc.match_projection", "v3")
def match_projection(participant_ref: str, rate: str) -> dict:
    record = odl.participant(participant_ref)
    plan = odl.plans()[record["plan_id"]]
    comp = Decimal(record["elections"].get("annual_compensation", "0"))
    match = plan["match"]
    matched_rate = min(Decimal(rate), Decimal(match["up_to"]))
    return _receipt(
        {
            "rate": rate,
            "annual_deferral": _money(comp * Decimal(rate)),
            "employer_match": _money(comp * matched_rate * Decimal(match["rate"])),
            "match_formula": f"{match['rate']} on the first {match['up_to']}",
            "inputs": ["odl.elections.v5", record["plan_id"]],
        },
        "calc.match_projection.v3",
    )


@tool_span("calc.pt_cost_estimate", "v2")
def pt_cost_estimate(participant_ref: str, visits: int = 1) -> dict:
    record = odl.participant(participant_ref)
    plan_id = record["medical_plan_id"]
    if plan_id is None:
        return _receipt({"applicable": False}, "calc.pt_cost_estimate.v2")
    plan = odl.plans()[plan_id]
    allowed = Decimal(plan["pt_allowed_amount"]) * visits
    acc = record["accumulators"]
    remaining_deductible = Decimal(acc["deductible_total"]) - Decimal(acc["deductible_met"])
    to_deductible = min(allowed, max(remaining_deductible, Decimal("0")))
    after = allowed - to_deductible
    coinsurance = after * Decimal(acc["coinsurance_rate"])
    return _receipt(
        {
            "applicable": True,
            "visits": str(visits),
            "allowed_amount": _money(allowed),
            "applied_to_deductible": _money(to_deductible),
            "coinsurance": _money(coinsurance),
            "member_cost": _money(to_deductible + coinsurance),
            "inputs": ["odl.accumulators.v4", plan_id],
        },
        "calc.pt_cost_estimate.v2",
    )


@tool_span("calc.deductible_status", "v1")
def deductible_status(participant_ref: str) -> dict:
    record = odl.participant(participant_ref)
    acc = record["accumulators"]
    if not acc:
        return _receipt({"applicable": False}, "calc.deductible_status.v1")
    total = Decimal(acc["deductible_total"])
    met = Decimal(acc["deductible_met"])
    return _receipt(
        {
            "applicable": True,
            "deductible_total": _money(total),
            "deductible_met": _money(met),
            "deductible_remaining": _money(total - met),
            "inputs": ["odl.accumulators.v4"],
        },
        "calc.deductible_status.v1",
    )
