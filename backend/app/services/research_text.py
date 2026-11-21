"""Parsers for the human-readable research reports committed with each run.

The reports are authoritative outputs of the research scripts. Parsing them is
preferable to re-deriving numbers, and every parsed value keeps its provenance.
"""

from __future__ import annotations

import re
from typing import Any


_SECTION_PATTERN = re.compile(r"^(TRAIN|VALIDATION|TEST)\s*$", re.MULTILINE)
_LABEL_LINE = re.compile(r"^([A-Za-z0-9 _\-\.\(\)/]+):\s*([0-9]+)\s*$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def parse_label_coverage(text: str) -> dict[str, dict[str, int]]:
    """Parse the label-coverage report into per-split label counts."""
    result: dict[str, dict[str, int]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _SECTION_PATTERN.match(line)
        if match:
            current = match.group(1).lower()
            result.setdefault(current, {})
            continue
        if current is None:
            continue
        label_match = _LABEL_LINE.match(line)
        if label_match:
            label = label_match.group(1).strip()
            if label.lower().startswith(("important", "the ")):
                continue
            result[current][label] = int(label_match.group(2))
        elif line.startswith("---") or line.startswith("Label counts"):
            continue
        elif ":" in line and not line[0].isdigit():
            # Narrative lines end the current count block.
            if not line.startswith(("Benign", "MIXED", "NO_FLOWS")):
                current = current if _LABEL_LINE.match(line) else current
    return result


def _numeric(token: str) -> float | int | None:
    token = token.strip()
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    return int(value) if value.is_integer() and "." not in token else value


def parse_multi_step_report(text: str) -> list[dict[str, Any]]:
    """Parse the multi-step forecasting table (horizon/label/metric rows).

    The report is whitespace-aligned, so rows are tokenised on any whitespace and
    the label is taken as the span between the horizon and the numeric columns.
    """
    rows: list[dict[str, Any]] = []
    columns: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("Horizon_Seconds"):
            columns = stripped.split()
            continue
        if not columns:
            continue
        cells = stripped.split()
        numeric_columns = len(columns) - 2
        if numeric_columns < 1 or len(cells) < numeric_columns + 2:
            continue
        if not re.fullmatch(r"\d+", cells[0]):
            continue
        numeric_cells = cells[-numeric_columns:]
        values = [_numeric(cell) for cell in numeric_cells]
        if any(value is None for value in values):
            continue
        label = " ".join(cells[1 : len(cells) - numeric_columns]).strip()
        if not label:
            continue
        row: dict[str, Any] = {"horizon_seconds": int(cells[0]), "label": label}
        for column, value in zip(columns[2:], values):
            row[column.strip().lower()] = value
        rows.append(row)
    return rows


def parse_feature_analysis_report(text: str) -> dict[str, Any]:
    """Parse feature-wise forecasting counts and per-horizon top features."""
    horizons: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    capture_top = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        horizon_match = re.match(r"^HORIZON:\s*(\d+)\s*seconds?$", line)
        if horizon_match:
            current = {
                "horizon_seconds": int(horizon_match.group(1)),
                "improved": None,
                "worse": None,
                "equal": None,
                "total": None,
                "top_improvements": [],
            }
            horizons.append(current)
            capture_top = False
            continue
        if current is None:
            continue
        counts_match = re.match(
            r"^Features (improved over persistence|worse than persistence|equal to persistence):\s*(\d+)/(\d+)$",
            line,
        )
        if counts_match:
            key = {
                "improved over persistence": "improved",
                "worse than persistence": "worse",
                "equal to persistence": "equal",
            }[counts_match.group(1)]
            current[key] = int(counts_match.group(2))
            current["total"] = int(counts_match.group(3))
            continue
        if line.startswith("TOP 5 FEATURES BY MAE IMPROVEMENT"):
            capture_top = True
            continue
        if capture_top:
            feature_match = re.match(r"^(.+?):\s*(-?\d+(?:\.\d+)?)%$", line)
            if feature_match:
                current["top_improvements"].append(
                    {
                        "feature": feature_match.group(1).strip(),
                        "improvement_percent": float(feature_match.group(2)),
                    }
                )
                continue
            if line:
                capture_top = False
    aggregate_match = re.search(r"All checks passed:\s*(True|False)", text)
    return {
        "horizons": horizons,
        "verification_passed": (
            aggregate_match.group(1) == "True" if aggregate_match else None
        ),
    }


def parse_training_report(text: str) -> dict[str, Any]:
    """Extract the scalar key/value facts recorded in the training report."""
    facts: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key or not value or len(key) > 60:
            continue
        facts[key.lower().replace(" ", "_")] = value
    for key, value in list(facts.items()):
        if isinstance(value, str):
            match = re.fullmatch(_NUMBER, value)
            if match:
                facts[key] = _numeric(value)
    return facts
