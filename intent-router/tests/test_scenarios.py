"""Scenarios S1-S12 from the design spec §4."""
from __future__ import annotations

import pytest

from conftest import T_DISABLED, T_ENABLED, context
from router.models import OUT_OF_SCOPE

CONTRIB_UTTERANCE = "Increase my 401(k) to 8%"


def test_s1_deductible_status_uses_ui_context_and_never_caches(router, table):
    decision = router.route(
        "How much of my deductible have I met?",
        context(ui_context={"viewing_plan": "PPO-High"}),
    )

    assert decision.intent == "plan.deductible.status"
    assert decision.slots == {"plan_id": "PPO-High"}
    assert decision.graph == "G-DEDUCTIBLE"
    assert decision.budgets == {"steps": 6, "tokens": 4000}
    assert decision.cache_key is None
    assert table.rows[decision.fired_row].intent == "plan.deductible.status"


def test_s2_rule_match_bypasses_the_classifier(router, classifier):
    decision = router.route(CONTRIB_UTTERANCE, context())

    assert classifier.calls == []
    assert decision.source == "rule"
    assert decision.slots == {"rate_pct": 8.0}
    assert decision.graph == "G-CONTRIB-CHANGE"
    assert decision.entry_node == "step_up_auth"


def test_s3_stepped_up_auth_enters_at_build_proposal(router):
    decision = router.route(CONTRIB_UTTERANCE, context(auth="stepped_up"))

    assert decision.graph == "G-CONTRIB-CHANGE"
    assert decision.entry_node == "build_proposal"


@pytest.mark.parametrize("auth", ["standard", "stepped_up"])
def test_s4_disabled_capability_routes_read_only(router, auth):
    decision = router.route(CONTRIB_UTTERANCE, context(capabilities=T_DISABLED, auth=auth))

    assert decision.graph == "G-EXPLAIN-ROUTE"


def test_s5_medium_confidence_asks_one_named_options_question(router, registry):
    decision = router.route("I want to change my contributions", context())

    assert decision.graph == "G-CLARIFY"
    assert decision.no_data_reads is True
    assert router.tool_calls == []
    assert registry.get("G-CLARIFY").manifest == ()
    assert decision.clarifying_question["options"] == [
        ("Change my 401(k) contribution amount", "ret.contribution.change"),
        ("View my current elections", "plan.elections.view"),
    ]


def test_s6_answer_resolves_on_a_single_reclassification(router):
    first = router.route("I want to change my contributions", context())
    second = router.route(
        "change the amount", context(conversation_state=first.conversation_state)
    )

    assert second.graph == "G-CONTRIB-CHANGE"
    assert second.ladder_path[0] == "clarify"
    assert "reclassified" in second.ladder_path
    assert second.ladder_path.count("clarify") == 1


def test_s7_still_ambiguous_answer_falls_down_never_asks_twice(router):
    first = router.route("I want to change my contributions", context())
    second = router.route(
        "I dunno, the money thing", context(conversation_state=first.conversation_state)
    )

    assert second.graph == "G-FALLBACK"
    assert second.ladder_path.count("clarify") == 1
    assert "fall_down" in second.ladder_path


def test_s8_sensitive_life_event_is_read_only(router):
    decision = router.route(
        "I'm going through a divorce, what happens to my coverage?", context()
    )

    assert decision.graph == "G-LIFE-EVENT-RO"
    assert decision.note == "empathetic-readonly-csr-offer"


def test_s9_low_confidence_hands_off_and_flags_for_catalog_review(router):
    decision = router.route("my money stuff is wrong", context())

    assert decision.graph == "G-FALLBACK"
    assert decision.log_for_catalog_review is True
    assert decision.handoff_payload["top_alternatives"][0] == [OUT_OF_SCOPE, 0.41]
    assert decision.handoff_payload["slots"] == {}


def test_s10_frozen_guard_outranks_transactional_rows_but_not_reads(router):
    frozen = context(capabilities=T_ENABLED, auth="stepped_up", frozen=True)
    transact = router.route(CONTRIB_UTTERANCE, frozen)
    read = router.route(
        "How much of my deductible have I met?",
        context(frozen=True, ui_context={"viewing_plan": "PPO-High"}),
    )

    assert transact.graph == "G-EXPLAIN-ROUTE"
    assert transact.note == "writes-disabled-during-freeze"
    assert transact.fired_row == 1
    assert read.graph == "G-DEDUCTIBLE"


def test_s11_plan_level_read_caches_independently_of_participant(router, catalog, table, registry):
    from router.engine import IntentRouter

    utterance = "What is the PPO-High deductible?"
    first = router.route(utterance, context(participant_ref="participant-1"))
    second = router.route(utterance, context(participant_ref="participant-2"))

    assert first.cache_key is not None
    assert first.cache_key == second.cache_key

    versioned = IntentRouter(
        catalog=catalog.with_version("catalog-v2"),
        table=_table_for_catalog(table, "catalog-v2"),
        registry=registry,
        classifier=router.classifier,
    )
    assert versioned.route(utterance, context()).cache_key != first.cache_key


def test_s12_out_of_range_slot_clarifies_and_never_clamps(router):
    decision = router.route("increase my 401(k) to 250%", context())

    assert decision.graph == "G-CLARIFY"
    assert decision.slots == {}
    assert "slot_validation_failed" in decision.ladder_path
    assert decision.clarifying_question["reason"] == "slot_validation_failed"


def _table_for_catalog(table, catalog_version):
    import dataclasses

    return dataclasses.replace(table, catalog_version=catalog_version)
