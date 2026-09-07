"""Deterministic mock model gateway.

The composer is a template engine over the evidence envelope: it can only restate values
that a tool produced, and every sentence carries the item id it came from. Two fault flags
exercise the output gate and the retry path.
"""
import os
import threading

_LOCK = threading.Lock()
_FAULTS_FIRED: set[str] = set()


class ModelTimeout(RuntimeError):
    pass


def reset() -> None:
    with _LOCK:
        _FAULTS_FIRED.clear()


LEAD = {
    "g_retirement_readonly_v2": "Here is what the plan rules and your current elections say",
    "g_contribution_change_v3": "Here is what your contribution change would look like",
    "g_balance_readonly_v1": "Here is your current retirement account picture",
    "g_deductible_readonly_v1": "Here is where your medical deductible stands",
    "g_health_readonly_v1": "Here is what your medical plan says",
    "g_eligibility_readonly_v1": "Here is your eligibility status",
    "g_life_event_readonly_v1": "Here is what the plan says about this life event",
}


def _sentence(item: dict) -> str:
    payload = item["payload"]
    if item["kind"] == "policy":
        return payload["text"].rstrip(".")
    if item["kind"] == "fact":
        parts = [f"{k.replace('_', ' ')} is {v}" for k, v in payload.items() if isinstance(v, str) and v]
        return "Your record shows " + ", ".join(parts) if parts else "Your record was retrieved"
    parts = [f"{k.replace('_', ' ')} is {v}" for k, v in payload.items() if isinstance(v, str) and v]
    return "The calculation shows " + ", ".join(parts) if parts else "The calculation returned no values"


def compose(graph_id: str, envelope: list[dict], intent: str) -> dict:
    if os.environ.get("MODEL_TIMEOUT_ONCE") == "1":
        with _LOCK:
            if "model_timeout" not in _FAULTS_FIRED:
                _FAULTS_FIRED.add("model_timeout")
                raise ModelTimeout(graph_id)

    lead = LEAD.get(graph_id, "Here is what your plan records show")
    sentences = []
    if envelope:
        sentences.append(f"{lead} [{envelope[0]['item_id']}].")
    for item in envelope:
        sentences.append(f"{_sentence(item)} [{item['item_id']}].")

    if os.environ.get("MODEL_UNCITED_CLAIM") == "1":
        with _LOCK:
            if "uncited" not in _FAULTS_FIRED:
                _FAULTS_FIRED.add("uncited")
                sentences.append("You can also withdraw 12345.00 today with no penalty.")

    return {
        "text": " ".join(sentences),
        "model": "mock.composer.v1",
        "intent": intent,
        "graph_id": graph_id,
    }
