"""Errors shared by the engine and the graph registry (kept dependency-free)."""
from __future__ import annotations


class ConfigError(ValueError):
    """Raised when a catalog, table or registry artifact is invalid."""


class RoutingError(RuntimeError):
    """Raised when no row matches: the engine never improvises a route."""
