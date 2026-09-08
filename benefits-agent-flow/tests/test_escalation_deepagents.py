"""The real deepagents harness, driven by a scripted model so no provider is called.

Skipped unless requirements-escalation.txt is installed (Python 3.11+).
"""
import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from app import escalation  # noqa: E402

ENVELOPE = [{
    "item_id": "ev-1", "kind": "fact", "evidence_type": "balances",
    "effective_from": "2026-01-01", "effective_to": None,
    "payload": {"balance": "412300.00"},
}]


class ScriptedModel(BaseChatModel):
    """Replays a fixed turn sequence and records the tools the harness bound to it."""

    script: list = []
    bound: list = []
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound.append([getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def call(name: str, args: dict, id_: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


def test_the_harness_reads_the_envelope_through_its_tools_and_answers():
    model = ScriptedModel(script=[
        call("list_evidence", {}, "1"),
        call("read_evidence_item", {"item_id": "ev-1"}, "2"),
        AIMessage(content="Your 401k balance is 412300.00 [ev-1]."),
    ])
    result = escalation.compose("what is my balance", ENVELOPE, model=model)
    assert result["text"] == "Your 401k balance is 412300.00 [ev-1]."


def test_no_corridor_or_data_plane_tool_is_reachable_from_the_harness():
    model = ScriptedModel(script=[AIMessage(content="no answer [ev-1].")])
    escalation.compose("what is my balance", ENVELOPE, model=model)
    bound = set(model.bound[0])
    assert {"list_evidence", "read_evidence_item"} <= bound
    forbidden = ("corridor", "propose", "confirm", "execute", "submit", "odl", "sor")
    assert not [name for name in bound if any(word in name for word in forbidden)]


def test_the_scratchpad_filesystem_refuses_writes():
    model = ScriptedModel(script=[
        call("write_file", {"file_path": "/notes.md", "content": "x"}, "1"),
        AIMessage(content="no answer [ev-1]."),
    ])
    escalation.compose("what is my balance", ENVELOPE, model=model)
    tool_result = str(model.seen[-1][-1].content).lower()
    assert "denied" in tool_result or "error" in tool_result


def test_a_model_that_never_answers_is_bounded_by_the_step_budget(monkeypatch):
    monkeypatch.setenv("ESCALATION_MAX_STEPS", "4")
    model = ScriptedModel(script=[call("list_evidence", {}, str(i)) for i in range(20)])
    assert escalation.compose("loop", ENVELOPE, model=model) == {"error": "GraphRecursionError"}
