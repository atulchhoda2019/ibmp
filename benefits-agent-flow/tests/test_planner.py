"""Stage 0-3: rules bypass the classifier, the catalog is closed, the table decides."""
from app.nodes import planner
from app.registry import decision_table, intent_names
from tests.conftest import make_state, run_turn


def plan_for(utterance: str, **kwargs):
    state = make_state(utterance, **kwargs)
    updates = planner.run(state)
    return updates["intent"], updates["plan"], updates.get("response")


def test_stage0_rule_bypasses_the_classifier():
    intent, plan, _ = plan_for("change my 401k contribution to 8 percent")
    assert intent.source == "rule" and intent.confidence == 1.0
    assert intent.slots["rate"] == "0.08"
    assert plan.posture == "WRITE"


def test_rate_canonicalization_is_stable_across_surface_forms():
    assert planner.canonicalize_rate("8%") == "0.08"
    assert planner.canonicalize_rate("eight percent") == "0.08"
    assert planner.canonicalize_rate("0.08") == "0.08"
    assert planner.canonicalize_rate("250%") is None       # out of schema, not a silent clamp


def test_unknown_request_is_out_of_scope_not_a_new_label():
    intent, plan, response = plan_for("book me a flight to Denver")
    assert intent.name == "__out_of_scope__"
    assert intent.name in intent_names()
    assert plan.graph_id == "g_fallback_v1"
    assert response["kind"] == "handoff"


def test_low_band_hands_off_and_queues_instead_of_guessing(monkeypatch, tmp_path):
    monkeypatch.setenv("FALLBACK_QUEUE", str(tmp_path / "q.jsonl"))
    _, plan, response = plan_for("uhh the thing with the stuff")
    assert plan.rung == 0 and response["kind"] == "handoff"
    assert (tmp_path / "q.jsonl").read_text().strip()


def test_near_tie_lands_in_medium_and_asks():
    intent, plan, response = plan_for("what happens if I get physical therapy")
    assert intent.band == "MEDIUM"
    assert plan.clarify and response["kind"] == "clarify"
    assert [option["intent"] for option in response["options"]] == [
        "retirement_projection", "pt_cost_estimate"]


def test_a_second_unresolved_ambiguity_drops_to_low_rather_than_asking_again():
    intent, _, response = plan_for("what happens if I get physical therapy", clarify_rounds=1)
    assert intent.band == "LOW"
    assert response["kind"] == "handoff"


def test_clarify_answer_resolves_by_exact_option_match():
    state = make_state(
        "claim status",
        clarify_rounds=1,
        ui_context={"clarify_options": [{"intent": "claim_status", "label": "claim status"}],
                    "clarify_utterance": "where is it"},
    )
    updates = planner.run(state)
    assert updates["intent"].name == "claim_status"
    assert updates["intent"].source == "rule"


def test_clarification_turn_reads_nothing(graph):
    """A clarify row must short circuit before any data read: no envelope, no tool calls."""
    rows = {row["row"]: row for row in decision_table()["rows"]}
    clarify_rows = [row for row in rows.values() if row.get("clarify")]
    assert clarify_rows, "the table must offer at least one clarification row"

    state = make_state("what happens if I get physical therapy")
    result = run_turn(graph, state)
    assert result["response"]["kind"] == "clarify"
    assert result["envelope"] == []
    assert result.get("draft") is None


def test_table_is_first_match_wins_with_guards_first():
    rows = decision_table()["rows"]
    guard_indexes = [i for i, row in enumerate(rows) if row["band"] == "LOW" or row["intent"] == "__out_of_scope__"]
    ordinary = [i for i, row in enumerate(rows) if i not in guard_indexes]
    assert max(guard_indexes) < min(ordinary)
