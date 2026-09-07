from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from classifier.stub import StubClassifier  # noqa: E402
from graphs.registry import load_registry  # noqa: E402
from router.catalog import load_catalog  # noqa: E402
from router.engine import IntentRouter  # noqa: E402
from router.models import RequestContext  # noqa: E402
from router.table import load_table  # noqa: E402
from router.trace import TraceCollector  # noqa: E402

CATALOG_PATH = ROOT / "catalog" / "v1.yaml"
TABLE_PATH = ROOT / "table" / "v1.yaml"
REGISTRY_PATH = ROOT / "graphs" / "registry.yaml"

T_ENABLED = {"ret.contribution.change": {"enabled": True, "rung": 2}}
T_DISABLED = {"ret.contribution.change": {"enabled": False}}


@pytest.fixture
def catalog():
    return load_catalog(CATALOG_PATH)


@pytest.fixture
def table():
    return load_table(TABLE_PATH)


@pytest.fixture
def registry():
    return load_registry(REGISTRY_PATH)


@pytest.fixture
def trace():
    return TraceCollector()


@pytest.fixture
def classifier(catalog):
    return StubClassifier(catalog=catalog)


@pytest.fixture
def router(catalog, table, registry, classifier, trace):
    return IntentRouter(
        catalog=catalog, table=table, registry=registry, classifier=classifier, trace_sink=trace
    )


def context(
    *,
    capabilities=None,
    auth="standard",
    frozen=False,
    ui_context=None,
    conversation_state=None,
    participant_ref="participant-1",
) -> RequestContext:
    return RequestContext(
        tenant_id="tenant-acme",
        participant_ref=participant_ref,
        auth_level=auth,
        tenant_capabilities=dict(T_ENABLED if capabilities is None else capabilities),
        tenant_frozen=frozen,
        ui_context=dict(ui_context or {}),
        conversation_state=dict(conversation_state or {}),
    )
