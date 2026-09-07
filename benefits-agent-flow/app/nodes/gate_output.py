"""Governance gate 2: schema, citations, claim support, PII.

Every sentence must carry an item id that exists in the envelope, and every number in the
draft must appear in an envelope payload: the composer restates facts, it cannot mint them.
"""
import re

from app.audit import node_event
from app.state import TurnState

CITE = re.compile(r"\[(ev-\d+)\]")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _numbers(text: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in NUMBER.findall(text.replace(",", ""))}


def supported_numbers(state: TurnState) -> set[str]:
    supported: set[str] = set()
    for item in state.envelope:
        supported |= _numbers(str(item.payload))
        supported |= _numbers(item.effective_from or "")
        supported |= _numbers(item.effective_to or "")
    return supported


def validate(state: TurnState) -> dict:
    draft = state.draft or ""
    reasons: list[str] = []
    if not draft.strip():
        return {"passed": False, "reasons": ["empty_draft"], "citations": []}

    known = {item.item_id for item in state.envelope}
    citations: list[str] = []
    for sentence in [s.strip() for s in re.split(r"(?<=\.)\s+", draft) if s.strip()]:
        cited = CITE.findall(sentence)
        if not cited:
            reasons.append(f"uncited sentence: {sentence[:60]}")
            continue
        for item_id in cited:
            if item_id not in known:
                reasons.append(f"unknown citation {item_id}")
            elif item_id not in citations:
                citations.append(item_id)

    unsupported = sorted(_numbers(CITE.sub("", draft)) - supported_numbers(state))
    if unsupported:
        reasons.append(f"unsupported numbers: {', '.join(unsupported)}")

    if EMAIL.search(draft) or SSN.search(draft):
        reasons.append("pii_in_output")

    return {"passed": not reasons, "reasons": reasons, "citations": citations}


def run(state: TurnState) -> dict:
    if state.validation and state.validation.get("reasons") == ["model_timeout"]:
        node_event(state, "gate_output", passed=False, reasons=["model_timeout"])
        return {}
    result = validate(state)
    if state.validation and state.validation.get("disclosures"):
        result["disclosures"] = state.validation["disclosures"]
    node_event(state, "gate_output", passed=result["passed"], reasons=result["reasons"])
    return {"validation": result}
