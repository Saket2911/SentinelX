"""Provenance vocabulary shared by every API response.

SentinelX must never present a number without saying where it came from. Every
quantitative block in this API is tagged with one of these values and, where
useful, the artifact file that backs it.
"""

from __future__ import annotations

from typing import Any

# A value copied verbatim from a committed research artifact (JSON/TXT report).
RECORDED = "recorded_artifact"
# A value computed by this backend from recorded artifacts (e.g. a percentage.
DERIVED = "derived"
# A value that cannot be produced in this checkout because a required, but
# git-ignored, artifact (checkpoint, scaler, dataset, .npy array) is absent.
UNAVAILABLE = "unavailable"
# Reserved for a genuinely live telemetry source. Never used today: no live
# ingestion path is implemented.
LIVE = "live_telemetry"


def block(
    value: Any,
    provenance: str,
    *,
    source: str | None = None,
    note: str | None = None,
) -> dict:
    """Wrap a value with its provenance metadata."""
    out: dict[str, Any] = {"value": value, "provenance": provenance}
    if source:
        out["source"] = source
    if note:
        out["note"] = note
    return out


def unavailable(reason: str, *, requires: list[str] | None = None) -> dict:
    """Describe data that exists in the design but not in this checkout."""
    out: dict[str, Any]
    out = {
        "value": None,
        "provenance": UNAVAILABLE,
        "reason": reason,
    }
    if requires:
        out["requires"] = requires
    return out
