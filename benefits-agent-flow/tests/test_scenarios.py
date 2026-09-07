"""S1-S12: the worked examples, end to end through the graph."""
from langgraph.types import Command

from app.mocks import odl
from tests.conftest import config_for, make_state, run_turn


def test_s1_projection_answers_from_tools_with_citations(graph):
    result = run_turn(graph, make_state("what happens if I raise my contribution to 8 percent"))
    assert result["plan"].table_row_id == "r-009"
    assert result["response"]["kind"] == "answer"
    assert len(result["response"]["citations"]) >= 3
    assert "5400.00" in result["response"]["text"]        # the match number came from the calculator


def test_s2_an_exact_shape_bypasses_the_classifier(graph):
    result = run_turn(graph, make_state("change my 401k contribution to 8 percent"))
    assert result["intent"].source == "rule"
    assert result["response"]["kind"] == "preview"


def test_s3_a_near_tie_asks_one_question(graph):
    result = run_turn(graph, make_state("what happens if I get physical therapy"))
    assert result["response"]["kind"] == "clarify"
    assert len(result["response"]["options"]) == 2


def test_s4_the_clarify_answer_routes_without_asking_again(graph):
    first = run_turn(graph, make_state("what happens if I get physical therapy"))
    answer = make_state(
        first["response"]["options"][1]["label"],
        participant_ref="P-1004",
        clarify_rounds=1,
        ui_context={"clarify_options": first["response"]["options"],
                    "clarify_utterance": "what will 3 pt visits cost"},
    )
    result = run_turn(graph, answer)
    assert result["intent"].name == "pt_cost_estimate"
    assert result["response"]["kind"] == "answer"


def test_s5_out_of_scope_hands_off_without_guessing(graph):
    result = run_turn(graph, make_state("book me a flight to Denver"))
    assert result["response"]["kind"] == "handoff"
    assert result["plan"].graph_id == "g_fallback_v1"


def test_s6_a_write_needs_a_confirmation_before_anything_changes(graph):
    state = make_state("change my 401k contribution to 8 percent")
    config = config_for(state)
    preview = run_turn(graph, state)["response"]
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.06"
    final = graph.invoke(Command(resume={"proposal_id": preview["proposal"]["proposal_id"],
                                         "nonce": preview["nonce"]}), config)
    assert final["response"]["kind"] == "receipt"


def test_s7_a_change_the_limit_forbids_is_refused_with_the_arithmetic(graph):
    result = run_turn(graph, make_state("change my 401k contribution to 60 percent",
                                        participant_ref="P-1002"))
    assert result["response"]["failed_validation"] is True
    assert "22800.00" in result["response"]["text"]


def test_s8_deductible_status_answers_from_the_accumulator(graph):
    result = run_turn(graph, make_state("how much of my deductible have I met this year",
                                        participant_ref="P-1004"))
    assert result["response"]["kind"] == "answer"
    assert "1200.00" in result["response"]["text"]        # 2000.00 - 800.00, computed by the tool


def test_s9_pt_cost_uses_the_calculator_not_the_model(graph):
    result = run_turn(graph, make_state("what will 3 physical therapy visits cost me",
                                        participant_ref="P-1004"))
    assert result["response"]["kind"] == "answer"
    assert any(item.evidence_type == "pt_cost" for item in result["envelope"])


def test_s10_an_ineligible_participant_is_told_the_date(graph):
    result = run_turn(graph, make_state("am I eligible for the 401k plan yet",
                                        participant_ref="P-1003"))
    assert result["response"]["kind"] == "answer"
    assert "2027-01-01" in result["response"]["text"]


def test_s11_a_sensitive_life_event_stays_read_only(graph):
    result = run_turn(graph, make_state("I am getting divorced what happens to my 401k"))
    assert result["plan"].posture == "READ" and result["plan"].rung == 0
    assert result["response"]["kind"] == "answer"


def test_s12_an_out_of_schema_slot_value_is_not_silently_clamped(graph):
    result = run_turn(graph, make_state("change my 401k contribution to 250 percent"))
    assert result.get("proposal") is None
    assert result["response"]["kind"] in ("clarify", "answer", "handoff")
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.06"
