"""Mock system of record: execute-once keyed by idempotency key, with fault injection."""
import os
import threading
from datetime import datetime, timezone

from app.mocks import odl
from app.tracing import tool_span

_LOCK = threading.Lock()
_RECEIPTS: dict[str, dict] = {}
_FAULTS_FIRED: set[str] = set()


class SorUnknownOutcome(RuntimeError):
    """The submit call did not return: the command may or may not have committed."""


def reset() -> None:
    with _LOCK:
        _RECEIPTS.clear()
        _FAULTS_FIRED.clear()


@tool_span("sor.submit", "v7")
def submit(command: dict) -> dict:
    key = command["idempotency_key"]
    with _LOCK:
        existing = _RECEIPTS.get(key)
        if existing is not None:
            return dict(existing, replayed=True)

        receipt = {
            "receipt_id": f"RCP-{key}",
            "idempotency_key": key,
            "action": command["action"],
            "applied": command["params"],
            "committed_at": datetime.now(timezone.utc).isoformat(),
            "replayed": False,
        }
        _RECEIPTS[key] = receipt
        if command["action"] == "ContributionChange":
            odl.apply_contribution_change(command["participant_ref"], command["params"]["rate"])

        if os.environ.get("SOR_TIMEOUT_ONCE") == "1" and "sor_timeout" not in _FAULTS_FIRED:
            _FAULTS_FIRED.add("sor_timeout")
            # Commit happened, the response did not: the corridor must reconcile by key.
            raise SorUnknownOutcome(key)

    return dict(receipt)


def receipts() -> dict[str, dict]:
    """Test/audit view of everything that actually committed."""
    with _LOCK:
        return {key: dict(value) for key, value in _RECEIPTS.items()}


@tool_span("sor.lookup_by_key", "v7")
def lookup_by_key(idempotency_key: str) -> dict | None:
    with _LOCK:
        receipt = _RECEIPTS.get(idempotency_key)
        return dict(receipt, replayed=True) if receipt else None
