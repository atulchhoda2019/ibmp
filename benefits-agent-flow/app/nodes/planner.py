"""Stage 0 rules, stage 1 classifier mock, stage 2 decision table, stage 3 ambiguity ladder."""
import json
import os
import pathlib
import re
from typing import Optional

from app.audit import emit, node_event
from app.registry import catalog, decision_table, intent_names
from app.state import Intent, PlanDecision, TurnState

WORD_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15",
}


def label_for(intent: str) -> str:
    return intent.replace("_", " ")


def canonicalize_rate(raw: str) -> Optional[str]:
    """'six percent', '6%', '0.06' all canonicalize to '0.06' so keys and proposals are stable."""
    text = raw.strip().lower()
    for word, digit in WORD_NUMBERS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*(%|percent)?", text)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2) or value > 1:
        value = value / 100.0
    if not 0 < value <= 1:
        return None
    return f"{value:.4f}".rstrip("0").rstrip(".") if value != int(value) else f"{value:.2f}"


def extract_slots(intent: str, utterance: str) -> dict[str, str]:
    slots: dict[str, str] = {}
    if intent in {"contribution_change", "retirement_projection"}:
        matches = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", utterance.lower())
        if matches:
            rate = canonicalize_rate(matches[-1])
            if rate:
                slots["rate"] = rate
        else:
            worded = re.search(r"\b(" + "|".join(WORD_NUMBERS) + r")\s+percent", utterance.lower())
            if worded:
                rate = canonicalize_rate(worded.group(0))
                if rate:
                    slots["rate"] = rate
    if intent == "pt_cost_estimate":
        visits = re.search(r"(\d{1,2})\s+(?:pt\s+)?(?:visits|sessions)", utterance.lower())
        slots["visits"] = visits.group(1) if visits else "1"
    return slots


def stage0_rules(utterance: str) -> Optional[Intent]:
    for rule in catalog()["rules"]:
        match = re.match(rule["pattern"], utterance, flags=re.IGNORECASE)
        if match:
            slots = extract_slots(rule["intent"], utterance)
            return Intent(
                name=rule["intent"], confidence=1.0, band="HIGH", slots=slots,
                alternates=[], source="rule",
            )
    return None


def band_for(confidence: float) -> str:
    for band, (low, high) in catalog()["bands"].items():
        if low <= confidence < high:
            return band
    return "LOW"


def stage1_classifier(utterance: str) -> Intent:
    """Keyword-scored lookup over the closed catalog. Never invents a label."""
    text = utterance.lower()
    scored: list[tuple[float, str]] = []
    for spec in catalog()["intents"]:
        if spec["name"] == "__out_of_scope__":
            continue
        # A longer phrase is stronger evidence than a bare word: "physical therapy" beats "claim".
        score = sum(1 + 0.4 * (len(kw.split()) - 1) for kw in spec["keywords"] if kw in text)
        if score:
            scored.append((score, spec["name"]))
    scored.sort(key=lambda row: (-row[0], row[1]))

    if not scored:
        return Intent(name="__out_of_scope__", confidence=0.2, band="LOW", slots={}, alternates=[])

    top, top_name = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    if runner >= 0.7 * top:
        confidence = 0.72          # near tie: MEDIUM, so the ladder asks instead of guessing
    else:
        confidence = min(0.55 + 0.20 * top + 0.12 * (top - runner), 0.97)
    name = top_name if top_name in intent_names() else "__out_of_scope__"
    return Intent(
        name=name,
        confidence=round(confidence, 2),
        band=band_for(confidence),
        slots=extract_slots(name, utterance),
        alternates=[n for _, n in scored[1:3]],
    )


def resolve_clarification(state: TurnState) -> Optional[Intent]:
    """A clarify answer re-enters the planner exactly once and resolves by exact option match."""
    if state.clarify_rounds < 1:
        return None
    offered = state.ui_context.get("clarify_options") or []
    answer = state.utterance.strip().lower()
    for option in offered:
        if answer == option["label"].lower() or answer == option["intent"].lower():
            return Intent(
                name=option["intent"], confidence=1.0, band="HIGH",
                slots=extract_slots(option["intent"], state.ui_context.get("clarify_utterance", "")),
                alternates=[], source="rule",
            )
    return None


def stage2_table(intent: Intent) -> dict:
    for row in decision_table()["rows"]:
        if row["intent"] not in ("any", intent.name):
            continue
        if row["band"] not in ("any", intent.band):
            continue
        return row
    raise RuntimeError(f"no decision table row for {intent.name}/{intent.band}: the table must be total")


def fallback_queue_path() -> pathlib.Path:
    default = pathlib.Path(__file__).parent.parent.parent / "var" / "fallback_queue.jsonl"
    path = pathlib.Path(os.environ.get("FALLBACK_QUEUE", default))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def log_fallback(state: TurnState, intent: Intent) -> None:
    with fallback_queue_path().open("a") as fh:
        fh.write(json.dumps({
            "conversation_id": state.conversation_id,
            "turn_id": state.turn_id,
            "tenant_id": state.tenant_id,
            "utterance": state.utterance,
            "top_guesses": [intent.name, *intent.alternates],
            "confidence": intent.confidence,
        }) + "\n")


def run(state: TurnState) -> dict:
    intent = resolve_clarification(state) or stage0_rules(state.utterance) or stage1_classifier(state.utterance)

    # Stage 3: a second unresolved ambiguity is not asked again, it drops to LOW.
    if intent.band == "MEDIUM" and state.clarify_rounds >= 1:
        intent = intent.model_copy(update={"band": "LOW"})

    row = stage2_table(intent)
    plan = PlanDecision(
        graph_id=row["graph"],
        table_row_id=row["row"],
        rung=row.get("rung_max", 0),
        budgets=row.get("budgets", {}),
        required_evidence=row.get("evidence", []),
        posture=row.get("posture", "READ"),
        requires_capability=row.get("requires_capability"),
        clarify=bool(row.get("clarify")),
    )

    updates: dict = {"intent": intent, "plan": plan}
    if intent.band == "LOW":
        log_fallback(state, intent)
        updates["response"] = {
            "kind": "handoff",
            "summary": "I could not place that request confidently, so I am handing it to a specialist.",
            "top_intents": [intent.name, *intent.alternates],
            "transcript": [state.utterance],
            "slots": intent.slots,
            "row": row["row"],
        }
    elif plan.clarify:
        options = [
            {"intent": name, "label": label_for(name)}
            for name in [intent.name, *intent.alternates][:3]
        ]
        updates["response"] = {
            "kind": "clarify",
            "question": "Which of these did you mean?",
            "options": options,
            "row": row["row"],
        }
        updates["clarify_rounds"] = state.clarify_rounds + 1

    node_event(state, "planner", source=intent.source, confidence=intent.confidence,
               row=row["row"], graph_id=plan.graph_id, posture=plan.posture, slots=intent.slots)
    emit({"conversation_id": state.conversation_id, "node": "planner.trace_metadata",
          "graph_id": plan.graph_id, "table_row_id": plan.table_row_id})
    return updates
