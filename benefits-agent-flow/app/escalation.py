"""Frontier escalation: a deepagents research loop that may only re-read the envelope.

The ladder is deterministic (board §5): failed validator check -> one composer retry ->
frontier -> human. This is the frontier step, and it is advisory only. The table still chose
the graph; the agent never gets a corridor tool, an ODL handle or the network beyond its own
model call, so the worst it can do is produce a draft that gate_output then rejects.

Off unless ESCALATION_BACKEND=deepagents, so CI and the offline runtime never reach a model.
"""
import importlib.util
import json
import os
from typing import Any, Callable

ADVISORY_TOOLS = ("list_evidence", "read_evidence_item")

SYSTEM_PROMPT = """You are a benefits answer composer working under a governance gate.

You may only use the evidence items handed to you by the tools. Rules the gate enforces
after you answer, so breaking one wastes the turn:
- Every sentence must end with the item id it came from, in square brackets, e.g. [ev-2].
- Every number you write must appear verbatim in an evidence payload. Never compute,
  round, convert or estimate a number the evidence does not already state.
- Never mention an email address, a social security number or any other identifier.
- If the evidence cannot answer the question, say so plainly in one cited sentence rather
  than filling the gap.

Answer in at most four short sentences. No preamble, no markdown, no bullet lists."""


def backend() -> str:
    return os.environ.get("ESCALATION_BACKEND", "none").lower()


def available() -> bool:
    """True only when the operator opted in and the extra install is present."""
    if backend() != "deepagents":
        return False
    return importlib.util.find_spec("deepagents") is not None


def _tools(envelope: list[dict]) -> list[Callable]:
    """Read-only closures over the already-authorized envelope. No ODL, no SOR, no writes."""
    by_id = {item["item_id"]: item for item in envelope}

    def list_evidence() -> str:
        """List the evidence items available for this answer, without their payloads."""
        return json.dumps([
            {
                "item_id": item["item_id"],
                "kind": item["kind"],
                "evidence_type": item["evidence_type"],
                "effective_from": item["effective_from"],
                "effective_to": item["effective_to"],
            }
            for item in envelope
        ])

    def read_evidence_item(item_id: str) -> str:
        """Read the payload of one evidence item by its id, e.g. "ev-1"."""
        item = by_id.get(item_id)
        if item is None:
            return f"no such item: {item_id}. Call list_evidence for the ids you may read."
        return json.dumps(item["payload"])

    return [list_evidence, read_evidence_item]


def compose(question: str, envelope: list[dict], model: Any = None) -> dict[str, Any]:
    """Run the frontier agent over the envelope. Returns {"text", "model"} or {"error"}.

    `model` accepts a chat model instance so the loop can be exercised without a provider.
    """
    from deepagents import FilesystemPermission, create_deep_agent

    model = model or os.environ.get("ESCALATION_MODEL", "openai:gpt-4.1")
    agent = create_deep_agent(
        model=model,
        tools=_tools(envelope),
        system_prompt=SYSTEM_PROMPT,
        # deepagents ships a scratchpad filesystem; deny every write on it so the harness
        # keeps only its planning notes in memory and nothing the agent does can persist.
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
    )
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            {"recursion_limit": int(os.environ.get("ESCALATION_MAX_STEPS", "12"))},
        )
    except Exception as exc:  # a frontier failure falls through to the human handoff
        return {"error": type(exc).__name__}
    messages = result.get("messages") or []
    text = messages[-1].content if messages else ""
    if isinstance(text, list):  # content blocks
        text = " ".join(part.get("text", "") for part in text if isinstance(part, dict))
    return {"text": text.strip(), "model": model if isinstance(model, str) else type(model).__name__}
