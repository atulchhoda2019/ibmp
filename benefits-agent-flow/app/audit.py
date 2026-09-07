"""JSONL audit log: the offline mirror of the LangSmith trace, one event per node."""
import json
import os
import pathlib
import threading
import time
from typing import Any

_LOCK = threading.Lock()

EMAIL_KEYS = {"email", "user_email", "userEmail"}


def audit_path() -> pathlib.Path:
    path = pathlib.Path(os.environ.get("AUDIT_LOG", pathlib.Path(__file__).parent.parent / "var" / "audit.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def redact(value: Any) -> Any:
    """Keep field names, drop participant payload values: refs and shapes only."""
    if isinstance(value, dict):
        return {k: ("<redacted>" if k in EMAIL_KEYS else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def emit(event: dict[str, Any]) -> dict[str, Any]:
    record = {"ts": time.time(), **redact(event)}
    line = json.dumps(record, default=str)
    with _LOCK:
        with audit_path().open("a") as fh:
            fh.write(line + "\n")
    return record


def read(conversation_id: str) -> list[dict[str, Any]]:
    path = audit_path()
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("conversation_id") == conversation_id:
                out.append(record)
    return out


def node_event(state: Any, node: str, **fields: Any) -> dict[str, Any]:
    plan = getattr(state, "plan", None)
    intent = getattr(state, "intent", None)
    return emit({
        "conversation_id": state.conversation_id,
        "turn_id": state.turn_id,
        "tenant_id": state.tenant_id,
        "node": node,
        "intent": intent.name if intent else None,
        "band": intent.band if intent else None,
        "row": plan.table_row_id if plan else None,
        "graph_id": plan.graph_id if plan else None,
        **fields,
    })
