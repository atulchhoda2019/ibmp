"""The model plane: served configs, model routing, composition and the escalation ladder."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from classifier.model import ModelClassifier
from errors import ConfigError
from modelplane.gateway import (
    EscalatedToHuman,
    FakeBackend,
    ModelGateway,
    OpenAIBackend,
    Scorecard,
    build_gateway,
    validate_composition,
)
from modelplane.registry import ModelRegistry, Pointer, load_model_registry
from modelplane.routing import choose, load_model_routing
from router.models import OUT_OF_SCOPE, RequestContext
from scripts.validate_config import validate_model_plane

ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = ROOT / "modelplane" / "registry.yaml"
ROUTING_PATH = ROOT / "modelplane" / "routing.yaml"

CONTEXT = RequestContext(tenant_id="tenant-acme", participant_ref="participant-1", auth_level="standard")


@pytest.fixture
def models():
    return load_model_registry(MODELS_PATH)


@pytest.fixture
def routing():
    return load_model_routing(ROUTING_PATH)


@pytest.fixture
def gateway():
    return build_gateway()


def write_registry(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(MODELS_PATH.read_text())
    mutate(raw)
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


# -- registry ---------------------------------------------------------------


def test_registry_is_versioned_and_carries_eval_metadata(models):
    config = models.get("slm-compose-benefits-v5")
    assert models.version == "models-v1"
    assert (config.base, config.adapter, config.ring) == (
        "gpt-4.1-mini", "benefits-compose-lora-v5", "prod"
    )
    assert config.eval.passing and config.eval.judge_family == "anthropic"


def test_config_without_a_passing_eval_never_loads(tmp_path):
    def fail_the_eval(raw):
        raw["served"]["slm-compose-benefits-v5"]["eval"]["score"] = 0.5

    with pytest.raises(ConfigError, match="below its threshold"):
        load_model_registry(write_registry(tmp_path, fail_the_eval))


def test_config_without_any_eval_never_loads(tmp_path):
    def drop_the_eval(raw):
        raw["served"]["slm-compose-benefits-v5"].pop("eval")

    with pytest.raises(ConfigError, match="must carry its eval record"):
        load_model_registry(write_registry(tmp_path, drop_the_eval))


def test_an_eval_judged_by_its_own_family_is_rejected(tmp_path):
    def same_family(raw):
        raw["served"]["slm-compose-benefits-v5"]["eval"]["judge_family"] = "openai"

    with pytest.raises(ConfigError, match="its own family"):
        load_model_registry(write_registry(tmp_path, same_family))


def test_a_tenant_adapter_needs_a_measured_gap(tmp_path):
    def tenant_scope_without_gap(raw):
        raw["served"]["slm-compose-benefits-v5"]["scope"] = "tenant"

    with pytest.raises(ConfigError, match="measured gap"):
        load_model_registry(write_registry(tmp_path, tenant_scope_without_gap))


def test_rollback_is_a_pointer_flip(models):
    pointer = models.serving["compose"]
    assert pointer.config in models.served and pointer.rollback_to in models.served


# -- routing ----------------------------------------------------------------


def test_the_ladder_prefers_task_over_domain(models, routing):
    domain = choose(models, routing, task="compose", domain="benefits", posture="read", tenant="t")
    task = choose(models, routing, task="compose", domain="retirement", posture="read", tenant="t")
    assert domain.config.name == "slm-compose-benefits-v5" and domain.rung_of_ladder == "domain"
    assert task.config.name == "slm-compose-retirement-v2" and task.rung_of_ladder == "task"


def test_a_tenant_row_outranks_the_task_row(models, routing):
    choice = choose(
        models, routing, task="compose", domain="retirement", posture="read", tenant="tenant-canary"
    )
    assert choice.config.name == "slm-compose-benefits-v6"
    assert choice.config.eval.measured_gap > 0


def test_a_tenant_row_without_a_measured_gap_is_illegal(tmp_path, routing):
    def erase_the_gap(raw):
        raw["served"]["slm-compose-benefits-v6"]["eval"]["measured_gap"] = 0.0

    models = load_model_registry(write_registry(tmp_path, erase_the_gap))
    with pytest.raises(ConfigError, match="without a measured gap"):
        choose(
            models,
            routing,
            task="compose",
            domain="retirement",
            posture="read",
            tenant="tenant-canary",
        )


def test_classify_and_compose_route_to_different_configs(gateway):
    classify = gateway.select(task="classify", domain="benefits", posture="read", tenant="t")
    compose = gateway.select(task="compose", domain="benefits", posture="read", tenant="t")
    assert classify.config.task == "classify" and compose.config.task == "compose"


def test_an_unroutable_task_is_a_config_error(gateway):
    with pytest.raises(ConfigError, match="no model row"):
        gateway.select(task="translate", domain="benefits", posture="read", tenant="t")


# -- composition and escalation --------------------------------------------


def test_the_composer_may_only_restate_the_facts_it_was_given():
    assert validate_composition("You have met $840 of $1,500.", {"a": "840", "b": "1500"}) is None
    assert validate_composition("You have met $900.", {"a": "840"}) == "unsupported number 900"


def test_a_clean_composition_never_escalates(gateway):
    composition = gateway.compose(
        question="plan.deductible.status",
        facts={"answer": "You've met $840 of your $1,500 deductible."},
        domain="benefits",
        posture="read",
        tenant="tenant-acme",
    )
    assert composition.escalated is False
    assert len(composition.attempts) == 1
    assert composition.cost_usd > 0


def test_one_retry_then_frontier(models, routing):
    """A served config that invents a number gets exactly one retry before the frontier."""
    served = "slm-compose-benefits-v5"
    gateway = ModelGateway(
        registry=models,
        table=routing,
        backend=FakeBackend(canned={served: "You've met $999."}),
    )
    composition = gateway.compose(
        question="plan.deductible.status",
        facts={"answer": "You've met $840."},
        domain="benefits",
        posture="read",
        tenant="tenant-acme",
    )
    assert [attempt.config for attempt in composition.attempts] == [
        served, served, models.frontier
    ]
    assert composition.escalated and composition.config == models.frontier


def test_the_frontier_failing_is_a_human_handoff(models, routing):
    invented = {name: "You've met $999." for name in models.served}
    gateway = ModelGateway(registry=models, table=routing, backend=FakeBackend(canned=invented))
    with pytest.raises(EscalatedToHuman):
        gateway.compose(
            question="plan.deductible.status",
            facts={"answer": "You've met $840."},
            domain="benefits",
            posture="read",
            tenant="tenant-acme",
        )


def test_the_scorecard_counts_catches_escalations_and_cost(models, routing):
    gateway = ModelGateway(
        registry=models,
        table=routing,
        backend=FakeBackend(canned={"slm-compose-benefits-v5": "You've met $999."}),
    )
    scorecard = Scorecard()
    scorecard.record(
        gateway.compose(
            question="plan.deductible.status",
            facts={"answer": "You've met $840."},
            domain="benefits",
            posture="read",
            tenant="tenant-acme",
        )
    )
    snapshot = scorecard.snapshot()
    assert snapshot["validator_catch"] == 2
    assert snapshot["escalation"] == 1
    assert snapshot["containment"] == 1.0
    assert snapshot["cost_per_successful_answer_usd"] > 0


# -- classifier -------------------------------------------------------------


def test_the_model_classifier_stays_inside_the_closed_catalog(catalog, models, routing):
    off_catalog = '{"intent": "ret.contribution.delete", "confidence": 0.99}'
    gateway = ModelGateway(
        registry=models,
        table=routing,
        backend=FakeBackend(canned={"slm-classify-benefits-v3": off_catalog}),
    )
    interpretation = ModelClassifier(gateway=gateway, catalog=catalog).classify("delete it", CONTEXT)
    assert interpretation.intent == OUT_OF_SCOPE and interpretation.confidence == 0.0


def test_the_model_classifier_parses_a_well_formed_answer(catalog, models, routing):
    answer = (
        '{"intent": "ret.contribution.change", "slots": {"rate_pct": "8"}, "confidence": 0.9, '
        '"alternatives": [["plan.elections.view", 0.4]]}'
    )
    gateway = ModelGateway(
        registry=models,
        table=routing,
        backend=FakeBackend(canned={"slm-classify-benefits-v3": answer}),
    )
    interpretation = ModelClassifier(gateway=gateway, catalog=catalog).classify("8%", CONTEXT)
    assert interpretation.intent == "ret.contribution.change"
    assert interpretation.slots == {"rate_pct": "8"}
    assert interpretation.top_alternatives == (("plan.elections.view", 0.4),)


def test_malformed_model_output_degrades_to_out_of_scope(catalog, models, routing):
    gateway = ModelGateway(
        registry=models,
        table=routing,
        backend=FakeBackend(canned={"slm-classify-benefits-v3": "sure thing!"}),
    )
    interpretation = ModelClassifier(gateway=gateway, catalog=catalog).classify("hi", CONTEXT)
    assert interpretation.intent == OUT_OF_SCOPE


# -- backends and the config gate ------------------------------------------


def test_the_openai_backend_refuses_to_start_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OpenAIBackend()


def test_the_config_gate_passes_on_the_shipped_plane(models, routing):
    assert validate_model_plane(models, routing) == []


def test_the_config_gate_rejects_a_canary_serving_pointer(models, routing):
    flipped = ModelRegistry(
        version=models.version,
        served=models.served,
        serving={
            **models.serving,
            "compose": Pointer(config="slm-compose-benefits-v6", rollback_to="slm-compose-benefits-v5"),
        },
        frontier=models.frontier,
    )
    assert any("canary ring" in error for error in validate_model_plane(flipped, routing))
