"""Mock policy evidence. Filter first on hard predicates, rerank only inside the authorized set."""
import json
import os
from datetime import datetime, timezone

from app.registry import FIXTURES_DIR
from app.tracing import tool_span


def _passages() -> list[dict]:
    with (FIXTURES_DIR / "passages.json").open() as fh:
        return json.load(fh)


def _effective(passage: dict, service_date: str) -> bool:
    if passage["effective_from"] > service_date:
        return False
    return passage["effective_to"] is None or passage["effective_to"] >= service_date


@tool_span("evidence.retrieve", "v6")
def retrieve(tenant_id: str, plan_ids: list[str], service_date: str, query: str, limit: int = 3) -> dict:
    fault = os.environ.get("EVIDENCE_EMPTY")
    if fault == "always" or (fault == "1" and not os.environ.get("_EVIDENCE_EMPTY_CONSUMED")):
        os.environ["_EVIDENCE_EMPTY_CONSUMED"] = "1"
        hits: list[dict] = []
    else:
        # Hard predicates: tenant, plan, and the version effective on the service date.
        authorized = [
            p
            for p in _passages()
            if p["tenant_id"] == tenant_id and p["plan_id"] in plan_ids and _effective(p, service_date)
        ]
        terms = {t for t in query.lower().replace("?", " ").split() if t}
        scored = [
            (sum(1 for k in p["keywords"] if k in query.lower() or k in terms), p["passage_id"], p)
            for p in authorized
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        hits = [p for score, _, p in scored if score > 0][:limit] or [p for _, _, p in scored][:limit]

    return {
        "payload": hits,
        "source": "evidence.policy.v6",
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
