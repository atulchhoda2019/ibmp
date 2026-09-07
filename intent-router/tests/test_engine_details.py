"""Behavioral details from §5: canonicalization, trace completeness, typed proposals."""
from __future__ import annotations

import asyncio

import pytest

from conftest import context
from graphs.registry import Proposal
from router.catalog import SlotSpec
from router.slots import canonicalize
from router.trace import TRACE_FIELDS


@pytest.mark.parametrize("raw,expected", [("six percent", 6.0), ("6%", 6.0), (0.06, 6.0), (6, 6.0), ("6", 6.0)])
def test_percent_canonicalization(raw, expected):
    assert canonicalize(SlotSpec(name="rate_pct", type="percent"), raw) == expected


def test_uncanonicalizable_percent_raises(router):
    with pytest.raises(ValueError):
        canonicalize(SlotSpec(name="rate_pct", type="percent"), "a lot")


def test_trace_line_is_complete(router, trace):
    router.route("Increase my 401(k) to 8%", context())

    assert len(trace.lines) == 1
    assert set(trace.lines[0]) == set(TRACE_FIELDS)
    assert trace.lines[0]["graph"] == "G-CONTRIB-CHANGE"
    assert trace.lines[0]["versions"] == {"catalog": "catalog-v1", "table": "table-v1"}
    assert trace.as_jsonl()


def test_transactional_graph_ends_at_a_typed_proposal(registry):
    graph = registry.instantiate("G-CONTRIB-CHANGE")
    result = asyncio.run(graph.call("ProposeContributionChange", rate_pct=8.0))

    assert isinstance(result, Proposal)
    assert result.confirmation_nonce
    assert graph.calls == [("ProposeContributionChange", {"rate_pct": 8.0})]


def test_graph_rejects_tools_outside_its_manifest(registry):
    graph = registry.instantiate("G-DEDUCTIBLE")
    with pytest.raises(Exception):
        asyncio.run(graph.call("ProposeContributionChange", rate_pct=8.0))


def test_async_wrapper_matches_the_sync_core(router):
    ctx = context()
    assert asyncio.run(router.route_async("Increase my 401(k) to 8%", ctx)) == router.route(
        "Increase my 401(k) to 8%", ctx
    )
