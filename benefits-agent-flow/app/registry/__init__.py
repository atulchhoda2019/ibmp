"""Registry loading: catalog, decision table, capability bundles, freshness policies."""
import functools
import pathlib
from typing import Any

import yaml

REGISTRY_DIR = pathlib.Path(__file__).parent
FIXTURES_DIR = REGISTRY_DIR.parent.parent / "fixtures"


def _load(name: str) -> dict[str, Any]:
    with (REGISTRY_DIR / name).open() as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=None)
def catalog() -> dict[str, Any]:
    return _load("intent_catalog.yaml")


@functools.lru_cache(maxsize=None)
def decision_table() -> dict[str, Any]:
    return _load("decision_table.yaml")


@functools.lru_cache(maxsize=None)
def bundles() -> dict[str, Any]:
    return _load("bundles.yaml")


@functools.lru_cache(maxsize=None)
def freshness() -> dict[str, Any]:
    return _load("freshness.yaml")


def intent_names() -> set[str]:
    return {i["name"] for i in catalog()["intents"]}


def versions() -> dict[str, str]:
    return {
        "catalog_version": catalog()["version"],
        "table_version": decision_table()["version"],
        "bundle_version": bundles()["version"],
        "freshness_version": freshness()["version"],
    }
