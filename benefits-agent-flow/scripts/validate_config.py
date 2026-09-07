"""Build gate for the served config: the registries must be internally consistent.

Run from benefits-agent-flow/:  python3 scripts/validate_config.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.nodes import read_calc, read_facts  # noqa: E402
from app.registry import bundles, catalog, decision_table, freshness, versions  # noqa: E402

POSTURES = {"READ", "WRITE"}
FACT_TYPES = set(read_facts.READERS)
CALC_TYPES = set(read_calc.CALCS)
POLICY_TYPES = {"plan_rules"}


def errors() -> list[str]:
    out: list[str] = []
    intents = {spec["name"] for spec in catalog()["intents"]}
    risks = {spec["name"]: spec["risk"] for spec in catalog()["intents"]}
    policies = set(freshness()["policies"])
    tenants = bundles()["tenants"]
    rows = decision_table()["rows"]

    seen: set[str] = set()
    for row in rows:
        rid = row["row"]
        if rid in seen:
            out.append(f"{rid}: duplicate row id")
        seen.add(rid)

        if row["intent"] not in intents | {"any", "__out_of_scope__"}:
            out.append(f"{rid}: intent {row['intent']} is not in the closed catalog")
        if row["posture"] not in POSTURES:
            out.append(f"{rid}: posture {row['posture']} is not one of {sorted(POSTURES)}")
        if row["posture"] == "READ" and row["rung_max"] != 0:
            out.append(f"{rid}: an advisory row may not carry rung_max {row['rung_max']}")
        if row["posture"] == "WRITE" and risks.get(row["intent"]) != "transactional":
            out.append(f"{rid}: a WRITE row needs a transactional intent")
        if row["posture"] == "WRITE" and not row.get("requires_capability"):
            out.append(f"{rid}: a WRITE row must name the capability it needs")

        for spec in row.get("evidence", []):
            etype, _, policy = spec.partition(":")
            if etype not in FACT_TYPES | CALC_TYPES | POLICY_TYPES:
                out.append(f"{rid}: no tool produces evidence type {etype}")
            if policy and policy not in policies:
                out.append(f"{rid}: unknown freshness policy {policy}")

        for key in ("max_steps", "max_tools", "max_tokens", "deadline_ms"):
            if key not in row["budgets"]:
                out.append(f"{rid}: budget {key} is missing")

    guards = [n for n, row in enumerate(rows) if row["intent"] in ("any", "__out_of_scope__")]
    ordinary = [n for n, row in enumerate(rows) if row["intent"] not in ("any", "__out_of_scope__")]
    if guards and ordinary and max(guards) > min(ordinary):
        out.append("guard rows must precede ordinary rows in a first-match-wins table")

    for tenant, bundle in tenants.items():
        for intent, rung in bundle["rungs"].items():
            if intent not in intents:
                out.append(f"{tenant}: rung set for unknown intent {intent}")
            if not 0 <= rung <= 4:
                out.append(f"{tenant}: rung {rung} for {intent} is out of range")
        for row in rows:
            cap = row.get("requires_capability")
            if row["posture"] != "WRITE" or cap not in bundle["capabilities"]:
                continue
            if bundle["rungs"].get(row["intent"], 0) > row["rung_max"]:
                out.append(f"{tenant}: bundle rung exceeds {row['row']} rung_max")

    lo_high, hi_high = catalog()["bands"]["HIGH"]
    lo_med, hi_med = catalog()["bands"]["MEDIUM"]
    lo_low, hi_low = catalog()["bands"]["LOW"]
    if not (lo_low == 0.0 and hi_low == lo_med and hi_med == lo_high and hi_high > 1.0):
        out.append("confidence bands must partition [0, 1]")

    return out


def main() -> int:
    found = errors()
    for message in found:
        print(f"FAIL {message}")
    if found:
        return 1
    print("OK", " ".join(f"{k}={v}" for k, v in versions().items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
