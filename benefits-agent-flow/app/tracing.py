"""LangSmith helpers. With LANGSMITH_TRACING unset everything degrades to a no-op decorator."""
import os
from typing import Any, Callable

from app.audit import redact

try:  # langsmith is optional at runtime; the mock plane never needs the network
    from langsmith import traceable as _traceable
except ImportError:  # pragma: no cover - exercised only where langsmith is absent
    _traceable = None


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(
        os.environ.get("LANGSMITH_API_KEY")
    )


def _mask_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Participant payloads enter traces as refs and field names, never raw values."""
    return redact(inputs)


def tool_span(name: str, version: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        if _traceable is None or not tracing_enabled():
            fn.tool_name = name  # type: ignore[attr-defined]
            fn.tool_version = version  # type: ignore[attr-defined]
            return fn
        wrapped = _traceable(
            name=name,
            run_type="tool",
            metadata={"tool_version": version},
            process_inputs=_mask_inputs,
        )(fn)
        wrapped.tool_name = name  # type: ignore[attr-defined]
        wrapped.tool_version = version  # type: ignore[attr-defined]
        return wrapped

    return deco


def run_config(conversation_id: str, tenant_id: str, versions: dict[str, str]) -> dict[str, Any]:
    """Root run metadata. graph_id is stamped by the planner as a child-run tag, because
    LangSmith has no API for rewriting root metadata from inside a node."""
    return {
        "configurable": {"thread_id": conversation_id},
        "run_name": f"turn:{conversation_id}",
        "tags": [tenant_id, "mocked-data-plane"],
        "metadata": dict(versions),
        "recursion_limit": 50,
    }


def run_url() -> str | None:
    if not tracing_enabled():
        return None
    project = os.environ.get("LANGSMITH_PROJECT", "default")
    return f"https://smith.langchain.com/projects/p/{project}"
