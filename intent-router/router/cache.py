"""Deterministic cache key derivation. Never similarity-keyed."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

from .catalog import IntentSpec


def derive_cache_key(
    *,
    intent: IntentSpec,
    slots: Mapping[str, Any],
    band: str,
    risk: str,
    catalog_version: str,
    table_version: str,
) -> Optional[str]:
    """Only READ rows at HIGH/RULE cache, and personal answers never do."""
    if risk != "READ" or band not in ("HIGH", "RULE") or intent.personal:
        return None
    scope = f"plan:{slots['plan_id']}" if intent.scope == "plan" else "global"
    payload = json.dumps(
        {
            "intent": intent.id,
            "slots": {k: slots[k] for k in sorted(slots)},
            "catalog_version": catalog_version,
            "table_version": table_version,
            "scope": scope,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()
