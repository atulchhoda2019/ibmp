"""Structured trace: one JSON line per routing decision."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

TRACE_FIELDS = (
    "timestamp", "tenant", "intent", "source", "confidence", "band", "band_edges",
    "fired_row", "graph", "posture", "capability", "rung", "ladder_path", "versions",
    "cache_key",
)


@dataclass
class TraceCollector:
    """In-memory sink used by tests; production wires a JSONL writer of the same shape."""

    lines: List[Dict[str, Any]] = field(default_factory=list)

    def __call__(self, record: Dict[str, Any]) -> None:
        self.lines.append(record)

    def as_jsonl(self) -> str:
        return "\n".join(json.dumps(line, sort_keys=True) for line in self.lines)


TraceSink = Callable[[Dict[str, Any]], None]


def build_trace(
    *,
    tenant: str,
    intent: str,
    source: str,
    confidence: float,
    band: str,
    fired_row: int,
    band_edges: str,
    graph: str,
    posture: str,
    capability: str,
    rung: int,
    ladder_path: tuple,
    versions: Dict[str, str],
    cache_key: str | None,
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tenant": tenant,
        "intent": intent,
        "source": source,
        "confidence": confidence,
        "band": band,
        "band_edges": band_edges,
        "fired_row": fired_row,
        "graph": graph,
        "posture": posture,
        "capability": capability,
        "rung": rung,
        "ladder_path": list(ladder_path),
        "versions": dict(versions),
        "cache_key": cache_key,
    }
