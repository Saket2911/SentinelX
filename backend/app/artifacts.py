"""Read-only access to the committed SentinelX research artifacts.

The repository intentionally keeps large binaries (``*.pt``, ``*.joblib``,
``*.npy``, ``*.csv``) out of version control. What *is* committed are the
JSON/TXT reports produced by the research pipeline. This module is the only
place that touches the filesystem for research data, and it never writes.

Design rules:
  * Missing artifacts are reported, never invented.
  * Parsing is defensive: a malformed artifact degrades one endpoint, not the app.
  * Results are cached by (path, mtime, size) so edits are picked up live.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import MODELS_DIR, REPORTS_DIR

log = logging.getLogger("sentinelx.artifacts")

_cache: dict[tuple[str, float, int], Any] = {}


def _cache_key(path: Path) -> tuple[str, float, int]:
    st = path.stat()
    return (str(path), st.st_mtime, st.st_size)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - defensive
        log.warning("cannot read %s: %s", path, exc)
        return None


def read_json(path: Path) -> dict | None:
    """Read a JSON artifact, returning None when absent or invalid."""
    if not path.is_file():
        return None
    key = _cache_key(path)
    if key in _cache:
        return _cache[key]
    text = _read_text(path)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("invalid JSON in %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        log.warning("expected a JSON object in %s", path)
        return None
    _cache[key] = data
    return data


def read_report(path: Path) -> str | None:
    """Read a plain-text report, returning None when absent."""
    if not path.is_file():
        return None
    key = _cache_key(path)
    if key in _cache:
        return _cache[key]
    text = _read_text(path)
    if text is not None:
        _cache[key] = text
    return text


def rel(path: Path) -> str:
    """Repository-relative path string for display in the UI."""
    from .config import REPO_ROOT

    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return path.name


# --------------------------------------------------------------------------
# Named artifact helpers
# --------------------------------------------------------------------------

def model_json(version_dir: str, *parts: str) -> dict | None:
    return read_json(MODELS_DIR / version_dir / Path(*parts))


def model_report(version_dir: str, *parts: str) -> str | None:
    return read_report(MODELS_DIR / version_dir / Path(*parts))


def repo_json(*parts: str) -> dict | None:
    return read_json(REPORTS_DIR.joinpath(*parts))


def repo_report(*parts: str) -> str | None:
    return read_report(REPORTS_DIR.joinpath(*parts))


# --------------------------------------------------------------------------
# Fixed-width table parsing
# --------------------------------------------------------------------------

_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_ROW_RE = re.compile(rf"^\s*({_NUM})\s+(.+?)\s+((?:{_NUM}\s*){{13}})$", re.MULTILINE)


def parse_multistep_report(text: str) -> list[dict]:
    """Parse the V3 recursive multi-step forecasting table.

    Columns (after Horizon_Seconds and Target_Label):
      Valid_Origins, LSTM_Huber, Persistence_Huber, LSTM_MAE, Persistence_MAE,
      MAE_Improvement_Percent, LSTM_RMSE, Persistence_RMSE,
      RMSE_Improvement_Percent, LSTM_Mean_Huber ... (trailing extras ignored)
    """
    rows: list[dict] = []
    body = text.split("RESULTS")[-1]
    for m in _ROW_RE.finditer(body):
        horizon = int(float(m.group(1)))
        label = m.group(2).strip()
        nums = [float(x) for x in m.group(3).split()]
        if len(nums) < 9:
            continue
        rows.append(
            {
                "horizon_seconds": horizon,
                "target_label": label,
                "valid_origins": int(nums[0]),
                "lstm_huber": nums[1],
                "persistence_huber": nums[2],
                "lstm_mae": nums[3],
                "persistence_mae": nums[4],
                "mae_improvement_percent": nums[5],
                "lstm_rmse": nums[6],
                "persistence_rmse": nums[7],
                "rmse_improvement_percent": nums[8],
            }
        )
    return rows


# Feature-family table: Horizon, Family (may contain spaces), then 13 numbers.
_FAMILY_RE = re.compile(
    rf"^\s*({_NUM})\s+(.+?)\s+((?:{_NUM}\s*){{13}})\s*$", re.MULTILINE
)


def parse_feature_family_report(text: str) -> list[dict]:
    rows: list[dict] = []
    current_horizon = 0
    for line in text.splitlines():
        hm = re.search(r"HORIZON:\s*(\d+)\s*seconds", line)
        if hm:
            current_horizon = int(hm.group(1))
            continue
        m = _FAMILY_RE.match(line)
        if not m:
            continue
        nums = [float(x) for x in m.group(3).split()]
        if len(nums) < 12:
            continue
        rows.append(
            {
                "horizon_seconds": current_horizon or int(float(m.group(1))),
                "feature_family": m.group(2).strip(),
                "feature_count": int(nums[0]),
                "lstm_better_count": int(nums[1]),
                "persistence_better_count": int(nums[2]),
                "equal_count": int(nums[3]),
                "lstm_mean_mae": nums[5],
                "persistence_mean_mae": nums[6],
                "mae_improvement_percent": nums[7],
                "lstm_mean_rmse": nums[8],
                "persistence_mean_rmse": nums[9],
                "rmse_improvement_percent": nums[10],
            }
        )
    return rows


def parse_label_coverage(text: str) -> dict[str, dict[str, int]]:
    """Parse the dataset label coverage report into {partition: {label: count}}."""
    out: dict[str, dict[str, int]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line in {"TRAIN", "VALIDATION", "TEST"}:
            current = line.lower()
            out.setdefault(current, {})
            continue
        if current and re.match(r"^[A-Za-z][\w .\-]*:\s*\d+$", line):
            label, _, count = line.partition(":")
            try:
                out[current][label.strip()] = int(count.strip())
            except ValueError:
                continue
        elif line.startswith("TOTAL") or not line:
            current = None if not line else current
    return {k: v for k, v in out.items() if v}


def parse_run_structure(text: str) -> dict:
    """Parse label-run-structure report headline sections."""
    result: dict[str, Any] = {"runs": [], "histogram": [], "total_states": None,
                              "total_runs": None, "timestamp_range": None}
    m = re.search(r"Total states:\s*([\d,]+)", text)
    if m:
        result["total_states"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Total runs:\s*([\d,]+)", text)
    if m:
        result["total_runs"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Timestamp range:\s*(.+?)\s+to\s+(.+)", text)
    if m:
        result["timestamp_range"] = {"start": m.group(1).strip(), "end": m.group(2).strip()}

    # RUN DURATION SUMMARY rows: Label Run_Count Mean Median Minimum Maximum
    in_summary = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Label") and "Run_Count" in s:
            in_summary = True
            continue
        if in_summary:
            parts = s.split()
            if len(parts) == 6 and parts[0] in {"Benign", "Bot", "Infilteration"}:
                try:
                    result["runs"].append(
                        {
                            "label": parts[0],
                            "run_count": int(parts[1]),
                            "mean_seconds": float(parts[2]),
                            "median_seconds": float(parts[3]),
                            "min_seconds": float(parts[4]),
                            "max_seconds": float(parts[5]),
                        }
                    )
                except ValueError:
                    continue
            elif s.startswith("RUN LENGTH HISTOGRAM") or not s:
                in_summary = False

    # RUN LENGTH HISTOGRAM rows: Label Duration_Category Run_Count States_In_Runs
    in_hist = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Label") and "Duration_Category" in s:
            in_hist = True
            continue
        if in_hist:
            if s.startswith("PERCENTAGE"):
                in_hist = False
                continue
            m2 = re.match(r"^(Benign|Bot|Infilteration)\s+(.+?)\s+(\d+)\s+(\d+)$", s)
            if m2:
                result["histogram"].append(
                    {
                        "label": m2.group(1),
                        "duration_category": m2.group(2).strip(),
                        "run_count": int(m2.group(3)),
                        "states_in_runs": int(m2.group(4)),
                    }
                )
    return result


def parse_anomaly_score_eval(text: str) -> dict:
    """Parse the V2 world-model anomaly score evaluation report."""
    out: dict[str, Any] = {"thresholds": {}, "distributions": [], "detection": []}
    for m in re.finditer(r"(\d+)th percentile:\s*([\d.]+)", text):
        out["thresholds"][f"p{m.group(1)}"] = float(m.group(2))
    for m in re.finditer(
        r"^\s*(Benign|Bot|NO_FLOWS)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
        text,
        re.MULTILINE,
    ):
        out["distributions"].append(
            {
                "label": m.group(1),
                "samples": int(m.group(2)),
                "mean": float(m.group(3)),
                "median": float(m.group(4)),
                "p95": float(m.group(5)),
                "p99": float(m.group(6)),
            }
        )
    for m in re.finditer(
        rf"^\s*({_NUM})\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+({_NUM})\s+(\d+)(?:st|nd|rd|th) percentile\s*$",
        text,
        re.MULTILINE,
    ):
        out["detection"].append(
            {
                "threshold": float(m.group(1)),
                "tp": int(m.group(2)),
                "fn": int(m.group(3)),
                "fp": int(m.group(4)),
                "tn": int(m.group(5)),
                "bot_detection_rate": float(m.group(6)),
                "benign_false_positive_rate": float(m.group(7)),
                "precision": float(m.group(8)),
                "f1": float(m.group(9)),
                "percentile": int(m.group(10)),
            }
        )
    return out
