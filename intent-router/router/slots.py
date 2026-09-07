"""Slot canonicalization and validation. A value never gets guessed or clamped."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from .catalog import IntentSpec, SlotSpec
from .models import ConfigError, RequestContext

WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20,
}
PERCENT_WORDS = re.compile(r"^(?P<word>[a-z\- ]+?)(\s*percent)?$")

UI_CONTEXT_KEYS = {"plan_id": "viewing_plan"}


@dataclass(frozen=True)
class SlotResult:
    values: Mapping[str, Any]
    error: Optional[str] = None
    missing_required: Sequence[str] = ()

    @property
    def ok(self) -> bool:
        return self.error is None


def canonicalize(spec: SlotSpec, value: Any) -> Any:
    if spec.type == "percent":
        return _canonical_percent(value)
    if spec.type == "string":
        return str(value).strip()
    raise ConfigError(f"no canonicalizer for slot type {spec.type!r}")


def _canonical_percent(value: Any) -> float:
    """"six percent", "6%", "0.06" and 6 all canonicalize to 6.0."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number * 100 if 0 < number < 1 else number
    text = str(value).strip().lower().rstrip("%").strip()
    try:
        number = float(text)
    except ValueError:
        word = PERCENT_WORDS.match(text)
        if word is None or word.group("word").strip() not in WORD_NUMBERS:
            raise ValueError(f"cannot canonicalize {value!r} as a percent") from None
        return float(WORD_NUMBERS[word.group("word").strip()])
    return number * 100 if 0 < number < 1 else number


def resolve_slots(intent: IntentSpec, raw: Mapping[str, Any], context: RequestContext) -> SlotResult:
    """Merge declared sources (utterance → conversation state → ui_context), canonicalize, validate.

    A slot that is absent is reported as missing, not invented: the selected graph collects it.
    A slot that is present but out of schema is an error, which the ladder turns into a clarify.
    Identity, tenant and auth are never sourced here: only slots declared in the catalog are read.
    """
    values: dict[str, Any] = {}
    missing: list[str] = []
    for name, spec in intent.slots.items():
        value = raw.get(name)
        if value is None:
            value = context.conversation_state.get(name)
        if value is None and "ui_context" in spec.source:
            value = context.ui_context.get(UI_CONTEXT_KEYS.get(name, name))
        if value is None:
            if spec.required:
                missing.append(name)
            continue
        try:
            canonical = canonicalize(spec, value)
        except ValueError as exc:
            return SlotResult(values=values, error=str(exc))
        if spec.min is not None and canonical < spec.min:
            return SlotResult(values=values, error=f"{name}={canonical} below min {spec.min}")
        if spec.max is not None and canonical > spec.max:
            return SlotResult(values=values, error=f"{name}={canonical} above max {spec.max}")
        values[name] = canonical
    return SlotResult(values=values, missing_required=tuple(missing))
