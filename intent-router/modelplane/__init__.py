"""The model plane: served configs, model routing, the gateway and the escalation ladder."""
from .gateway import (
    Composition,
    EscalatedToHuman,
    FakeBackend,
    ModelBackend,
    ModelGateway,
    OpenAIBackend,
    Scorecard,
    backend_from_env,
    build_gateway,
    validate_composition,
)
from .registry import EvalRecord, ModelRegistry, ServedConfig, load_model_registry
from .routing import ModelChoice, ModelRoutingTable, choose, load_model_routing

__all__ = [
    "Composition",
    "EscalatedToHuman",
    "EvalRecord",
    "FakeBackend",
    "ModelBackend",
    "ModelChoice",
    "ModelGateway",
    "ModelRegistry",
    "ModelRoutingTable",
    "OpenAIBackend",
    "Scorecard",
    "ServedConfig",
    "backend_from_env",
    "build_gateway",
    "choose",
    "load_model_registry",
    "load_model_routing",
    "validate_composition",
]
