"""LangSmith helpers. With LANGSMITH_TRACING unset everything degrades to a no-op decorator."""
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from app.audit import redact

try:  # langsmith is optional at runtime; the mock plane never needs the network
    from langsmith import Client, traceable as _traceable
    from langchain_core.tracers.context import collect_runs
except ImportError:  # pragma: no cover - exercised only where langsmith is absent
    Client = None
    _traceable = None
    collect_runs = None


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


class RunCapture:
    """Holds the root run URL of the turn that just executed, so /trace can link the tree."""

    url: str | None = None


@contextmanager
def capture_run() -> Iterator[RunCapture]:
    """Collect the root run of the enclosed graph invocation and resolve its LangSmith URL."""
    capture = RunCapture()
    if not tracing_enabled() or collect_runs is None or Client is None:
        yield capture
        return
    with collect_runs() as collector:
        yield capture
    for run in collector.traced_runs:
        try:
            capture.url = Client().get_run_url(
                run=run, project_name=os.environ.get("LANGSMITH_PROJECT")
            )
        except Exception:  # pragma: no cover - a trace link must never fail a turn
            capture.url = None
        break
