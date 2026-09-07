"""Final answer, scripted fallback, draft, or handoff. Never a fact that is not in state."""
from app.audit import node_event
from app.state import TurnState

SCRIPTED = (
    "I could not put together an answer I can stand behind for that one. "
    "A benefits specialist can take it from here with everything you have told me."
)


def run(state: TurnState) -> dict:
    if state.receipt is not None:
        response = {"kind": "receipt", "receipt": state.receipt}
        node_event(state, "respond", kind="receipt")
        return {"response": response}

    if state.response is not None:
        node_event(state, "respond", kind=state.response.get("kind"), preexisting=True)
        return {}

    if state.plan and state.plan.posture == "DRAFT":
        response = {
            "kind": "draft",
            "form": {
                "action": "ContributionChange",
                "requested": {"rate": state.intent.slots.get("rate")},
                "instructions": "Your plan sponsor requires this change to be submitted by you.",
            },
            "citations": (state.validation or {}).get("citations", []),
        }
    elif state.validation and state.validation.get("passed"):
        response = {
            "kind": "answer",
            "text": state.draft,
            "citations": state.validation.get("citations", []),
        }
        if state.validation.get("disclosures"):
            response["disclosures"] = state.validation["disclosures"]
    else:
        response = {
            "kind": "answer",
            "text": SCRIPTED,
            "citations": [],
            "scripted": True,
            "reasons": (state.validation or {}).get("reasons", []),
        }

    node_event(state, "respond", kind=response["kind"])
    return {"response": response}
