"""Mock Operational Data Layer. Every read returns its payload plus provenance."""
import copy
import json
import threading
from datetime import datetime, timezone

from app.registry import FIXTURES_DIR
from app.tracing import tool_span

_LOCK = threading.RLock()
_STATE: dict | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict:
    global _STATE
    with _LOCK:
        if _STATE is None:
            with (FIXTURES_DIR / "participants.json").open() as fh:
                _STATE = json.load(fh)
        return _STATE


def reset() -> None:
    """Tests call this between runs: the mock SoR writes back into the fixture copy."""
    global _STATE
    with _LOCK:
        _STATE = None


def plans() -> dict:
    with (FIXTURES_DIR / "plans.json").open() as fh:
        return json.load(fh)


def participant(participant_ref: str) -> dict:
    record = _load().get(participant_ref)
    if record is None:
        raise KeyError(f"unknown participant {participant_ref}")
    return copy.deepcopy(record)


def _envelope(payload: dict, source: str, effective_from: str = "2024-01-01") -> dict:
    return {
        "payload": payload,
        "source": source,
        "observed_at": _now(),
        "effective_from": effective_from,
        "effective_to": None,
    }


@tool_span("odl.get_elections", "v5")
def get_elections(participant_ref: str) -> dict:
    record = participant(participant_ref)
    return _envelope(
        {
            "rate": record["elections"].get("401k_rate"),
            "annual_compensation": record["elections"].get("annual_compensation"),
            "ytd_deferral": record["ytd_deferral"],
            "plan_id": record["plan_id"],
        },
        "odl.elections.v5",
    )


@tool_span("odl.get_balances", "v3")
def get_balances(participant_ref: str) -> dict:
    record = participant(participant_ref)
    return _envelope({"balances": record["balances"], "beneficiaries": record["beneficiaries"]}, "odl.balances.v3")


@tool_span("odl.get_accumulators", "v4")
def get_accumulators(participant_ref: str) -> dict:
    record = participant(participant_ref)
    return _envelope(
        {"accumulators": record["accumulators"], "medical_plan_id": record["medical_plan_id"]},
        "odl.accumulators.v4",
    )


@tool_span("odl.get_eligibility", "v2")
def get_eligibility(participant_ref: str) -> dict:
    record = participant(participant_ref)
    eligible = record["eligibility_date"] <= datetime.now(timezone.utc).date().isoformat()
    return _envelope(
        {"eligibility_date": record["eligibility_date"], "eligible": eligible, "plan_id": record["plan_id"]},
        "odl.eligibility.v2",
    )


def apply_contribution_change(participant_ref: str, rate: str) -> None:
    """Called by the mock system of record only: this is the read-after-write surface."""
    with _LOCK:
        _load()[participant_ref]["elections"]["401k_rate"] = rate
