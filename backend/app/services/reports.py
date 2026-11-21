"""Report generation and download.

Reports are assembled from the same services that feed the UI, so a report can
never contain a number the dashboard would disagree with. Every report embeds the
provenance and the interpretation limits of its inputs.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..db import Database
from . import dashboard as dashboard_service
from . import events as events_service
from . import models as models_service
from . import predictions as predictions_service
from .anomaly import calibration_for
from .artifacts import ArtifactStore


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    if value is None:
        return "n/a"
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows available._\n"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(_fmt(value))
            else:
                cells.append("" if value is None else str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def _section_summary(store: ArtifactStore, database: Database, run_id: str) -> str:
    dashboard = dashboard_service.overview(store, database, run_id)
    lines = [
        "## Executive summary",
        "",
        f"**Posture:** {dashboard['status']['posture']} — {dashboard['status']['headline']}",
        "",
        dashboard["status"]["detail"],
        "",
        _table(
            dashboard["metrics"],
            [("label", "Metric"), ("display", "Value"), ("unit", "Unit"), ("detail", "Interpretation")],
        ),
        "### Interpretation limits",
        "",
        *[f"- {item}" for item in dashboard["limitations"]],
        "",
    ]
    return "\n".join(lines)


def _section_model(store: ArtifactStore, run_id: str) -> str:
    model = models_service.summary(store, run_id, run_id)
    evaluation = model["test_metrics"] or {}
    lstm = evaluation.get("lstm") or {}
    persistence = evaluation.get("persistence") or {}
    improvement = evaluation.get("improvement_percent") or {}
    architecture = model["architecture"]
    training = model["training"]
    rows = [
        {"metric": "MAE", "lstm": lstm.get("mae"), "persistence": persistence.get("mae"), "change": improvement.get("mae")},
        {"metric": "RMSE", "lstm": lstm.get("rmse"), "persistence": persistence.get("rmse"), "change": improvement.get("rmse")},
        {"metric": "Huber", "lstm": lstm.get("huber_loss"), "persistence": persistence.get("huber_loss"), "change": improvement.get("huber_loss")},
    ]
    return "\n".join(
        [
            "## Model evaluation",
            "",
            f"- Run: `{run_id}`",
            f"- Architecture: {architecture.get('model')} — {architecture.get('input_features')} inputs, "
            f"hidden {architecture.get('hidden_size')}, {architecture.get('num_layers')} LSTM layer(s), "
            f"dropout {architecture.get('dropout')}, sequence length {architecture.get('sequence_length')}",
            f"- Training: best epoch {training.get('best_epoch')}, best validation Huber "
            f"{_fmt(training.get('best_validation_huber'))}, batch size {training.get('batch_size')}, "
            f"{training.get('optimizer')} at lr {training.get('learning_rate')}",
            f"- Held-out test sequences: {_fmt(evaluation.get('samples'), 0)}",
            f"- Reported improvement over persistence uses the recorded evaluation; the test partition "
            f"was previously used for exploratory analysis.",
            "",
            _table(
                rows,
                [("metric", "Metric"), ("lstm", "LSTM"), ("persistence", "Persistence"), ("change", "Improvement %")],
            ),
            "All values are in standardized preprocessed model feature space.",
            "",
        ]
    )


def _section_anomaly(store: ArtifactStore, run_id: str) -> str:
    calibration = calibration_for(store, run_id)
    validation = store.read_json(run_id, "validation_anomaly") or {}
    test = store.read_json(run_id, "test_anomaly") or {}
    lines = [
        "## Residual anomaly analysis",
        "",
        f"- Score: {calibration.method}",
        f"- Frozen threshold: {_fmt(calibration.threshold)} ({calibration.threshold_source})",
        f"- Recorded benign residual quantiles: "
        + ", ".join(f"p{key}={_fmt(value)}" for key, value in calibration.quantiles.items()),
        "",
        "### Recorded detection metrics",
        "",
        "| Split | Target | TP | FN | FP | Precision | Recall | Specificity |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, summary, key in (
        ("Validation", validation, "detection_metrics_benign_vs_infilteration"),
        ("Test", test, "detection_metrics_benign_vs_bot"),
    ):
        metrics = summary.get(key) or {}
        if not metrics:
            continue
        lines.append(
            f"| {name} | {key.split('_vs_')[-1]} | {metrics.get('TP')} | {metrics.get('FN')} | "
            f"{metrics.get('FP')} | {_fmt(metrics.get('precision'))} | {_fmt(metrics.get('recall'))} | "
            f"{_fmt(metrics.get('specificity'))} |"
        )
    lines.extend(["", "These metrics describe a residual review heuristic, not a validated detector.", ""])
    lines.extend(f"- {note}" for note in calibration.notes)
    lines.append("")
    return "\n".join(lines)


def _section_triage(store: ArtifactStore, database: Database, run_id: str) -> str:
    page = events_service.list_events(store, database, run_id, page=1, page_size=25, include_top=True)
    rows = [
        {
            "step": event.get("step"),
            "timestamp": event.get("timestamp"),
            "label": event.get("dataset_label"),
            "residual": event.get("residual_score"),
            "priority": event.get("priority"),
            "status": event.get("status"),
            "assignee": event.get("assignee"),
        }
        for event in page["events"]
    ]
    return "\n".join(
        [
            "## Threat review candidates",
            "",
            f"- Selection: {page['filters'].get('selection')} (threshold {_fmt(page['filters'].get('threshold'))})",
            f"- Candidate pool: {page['filters'].get('candidate_pool')}",
            f"- Status counts: {json.dumps(page['status_counts'])}",
            "",
            _table(
                rows,
                [
                    ("step", "Step"),
                    ("timestamp", "Timestamp"),
                    ("label", "Dataset label"),
                    ("residual", "Residual"),
                    ("priority", "Priority"),
                    ("status", "Status"),
                    ("assignee", "Assignee"),
                ],
            ),
            "Residual candidates are review prompts. They are not confirmed incidents and the dataset "
            "label is a reference annotation, not a model output.",
            "",
        ]
    )


def _section_explainability(store: ArtifactStore, run_id: str) -> str:
    summary = store.read_json(run_id, "shap") or {}
    importance = store.read_table(run_id, "shap_importance")
    lines = [
        "## Explainability",
        "",
        f"- Method: {summary.get('method') or 'n/a'}",
        f"- Background: {summary.get('background_source') or 'n/a'} "
        f"({summary.get('background_size') or 'n/a'} samples)",
        f"- Explanation samples: {summary.get('benign_explanation_samples') or 0} benign, "
        f"{summary.get('bot_explanation_samples') or 0} bot",
        f"- Explained outputs: {', '.join(summary.get('output_features') or []) or 'n/a'}",
        "",
    ]
    if importance is not None and not importance.empty and "Target Label" in importance.columns:
        label = sorted(importance["Target Label"].astype(str).unique())[0]
        subset = (
            importance[importance["Target Label"].astype(str) == label]
            .groupby("Input Feature", as_index=False)["Mean Absolute SHAP"]
            .mean()
            .sort_values("Mean Absolute SHAP", ascending=False)
            .head(10)
        )
        lines.append(f"Top features by mean |SHAP| ({label}):")
        lines.append("")
        lines.append(_table(subset.to_dict(orient="records"), [("Input Feature", "Feature"), ("Mean Absolute SHAP", "Mean |SHAP|")]))
    else:
        lines.append(
            "Aggregated SHAP tables are not present in this checkout; only the recorded experiment "
            "summary is available."
        )
        lines.append("")
    lines.append(
        "SHAP attributions explain model behaviour relative to a benign background. They are not "
        "causal evidence and not attack probabilities."
    )
    lines.append("")
    return "\n".join(lines)


def _section_dataset(store: ArtifactStore, run_id: str) -> str:
    from .datasets import overview as dataset_overview

    overview = dataset_overview(store, store.datasets_dir, run_id)
    rows = [
        {
            "label": artifact["label"],
            "file": Path(artifact["path"]).name,
            "present": "yes" if artifact["present"] else "no",
        }
        for artifact in overview["artifacts"]
    ]
    coverage_lines = []
    for split, labels in overview["label_coverage"].items():
        total = sum(labels.values())
        coverage_lines.append(f"- {split}: {len(labels)} labels, {total:,} windows")
    return "\n".join(
        [
            "## Dataset status",
            "",
            f"- Dataset directory: `{overview['source']['detail']}`",
            f"- Model feature definition: {overview['model_features'] or 'not available'} features",
            "",
            "### Preprocessing artifacts",
            "",
            _table(rows, [("label", "Artifact"), ("file", "File"), ("present", "Present")]),
            "### Label coverage (recorded report)",
            "",
            *(coverage_lines or ["- Label coverage report not available."]),
            "",
            "Chronological partitions have different label coverage: the test partition contains Bot "
            "and Benign, the validation partition contains Benign and Infilteration. Bot is absent "
            "from training.",
            "",
        ]
    )


SECTIONS = {
    "summary": "Executive summary",
    "model_evaluation": "Model evaluation",
    "anomaly": "Residual anomaly analysis",
    "triage": "Threat review candidates",
    "explainability": "Explainability",
    "dataset": "Dataset status",
}


def create_report(
    store: ArtifactStore,
    database: Database,
    reports_dir: Path,
    run_id: str,
    title: str,
    sections: list[str],
    author: str,
) -> dict[str, Any]:
    unknown = [section for section in sections if section not in SECTIONS]
    if unknown:
        raise ValueError(f"Unknown report sections: {unknown}")
    if not sections:
        raise ValueError("At least one report section is required.")

    timestamp = datetime.now(tz=timezone.utc)
    report_id = f"{timestamp.strftime('%Y%m%dT%H%M%S')}-{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:40] or 'report'}"
    parts = [
        f"# {title}",
        "",
        f"- Generated: {timestamp.isoformat(timespec='seconds')}",
        f"- Author: {author}",
        f"- Active run: `{run_id}`",
        "- Data source: committed research artifacts and recorded evaluation outputs.",
        "",
        "> SentinelX predicts the next network state. Prediction residuals and dataset labels are "
        "review inputs, not confirmed malicious activity.",
        "",
    ]
    for section in sections:
        if section == "summary":
            parts.append(_section_summary(store, database, run_id))
        elif section == "model_evaluation":
            parts.append(_section_model(store, run_id))
        elif section == "anomaly":
            parts.append(_section_anomaly(store, run_id))
        elif section == "triage":
            parts.append(_section_triage(store, database, run_id))
        elif section == "explainability":
            parts.append(_section_explainability(store, run_id))
        elif section == "dataset":
            parts.append(_section_dataset(store, run_id))
    parts.append("---")
    parts.append("")
    parts.append("Generated by SentinelX. No metric in this report was computed on the test partition during this run.")

    markdown = "\n".join(parts)
    reports_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = reports_dir / f"{report_id}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    metadata = {
        "report_id": report_id,
        "title": title,
        "author": author,
        "created_at": timestamp.isoformat(timespec="seconds"),
        "run_id": run_id,
        "sections": sections,
        "formats": ["md", "json"],
        "size_bytes": len(markdown.encode("utf-8")),
    }
    (reports_dir / f"{report_id}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    database.audit(author, "report_created", report_id)
    return metadata


def list_reports(reports_dir: Path) -> list[dict[str, Any]]:
    if not reports_dir.is_dir():
        return []
    reports = []
    for path in sorted(reports_dir.glob("*.json"), reverse=True):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict) and "report_id" in metadata:
            reports.append(metadata)
    return reports


def report_path(reports_dir: Path, report_id: str, fmt: str = "md") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", report_id):
        raise ValueError("Invalid report id.")
    path = (reports_dir / f"{report_id}.{fmt}").resolve()
    if reports_dir.resolve() not in path.parents:
        raise ValueError("Invalid report path.")
    if not path.is_file():
        raise FileNotFoundError(report_id)
    return path
