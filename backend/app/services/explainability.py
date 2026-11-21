"""Explainability service.

Global and temporal SHAP tables are read from the run's explanation artifacts.
Local explanations require the SHAP value archive (``shap_values.npz``), which is
git-ignored; when it is missing the API says so instead of approximating.

Attributions explain model behaviour relative to a benign background. They are
not causal claims and not attack probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .artifacts import ArtifactError, ArtifactStore

SEQUENCE_LENGTH = 10

LIMITATIONS = [
    "SHAP values attribute the model's prediction to input features relative to a benign "
    "training background; they do not establish causality.",
    "Attributions are in preprocessed model feature space, not raw network units.",
    "Feature contribution is not evidence of malicious activity and is not an attack probability.",
    "Explanations describe the LSTM world model only; they say nothing about unobserved traffic.",
]


def _importance_rows(frame, label: str | None, output: str | None, top_n: int) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    work = frame
    if "Target Label" in work.columns and label:
        work = work[work["Target Label"].astype(str) == label]
    if "Predicted Output Feature" in work.columns and output:
        work = work[work["Predicted Output Feature"].astype(str) == output]
    if work.empty:
        return []
    grouped = (
        work.groupby("Input Feature", as_index=False)
        .agg(
            mean_absolute=("Mean Absolute SHAP", "mean"),
            mean_signed=("Mean Signed SHAP", "mean"),
        )
        .sort_values("mean_absolute", ascending=False)
        .head(max(1, top_n))
    )
    return [
        {
            "feature": str(row["Input Feature"]),
            "value": None,
            "contribution": float(row["mean_absolute"]),
            "direction": "increase" if row["mean_signed"] > 0 else ("decrease" if row["mean_signed"] < 0 else "neutral"),
            "rank": rank,
        }
        for rank, (_, row) in enumerate(grouped.iterrows(), start=1)
    ]


def _temporal_rows(frame, label: str | None, output: str | None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    work = frame
    if "Target Label" in work.columns and label:
        work = work[work["Target Label"].astype(str) == label]
    if "Predicted Output Feature" in work.columns and output:
        work = work[work["Predicted Output Feature"].astype(str) == output]
    if work.empty:
        return []
    grouped = (
        work.groupby("Input Time Position", as_index=False)["Mean Absolute SHAP"]
        .mean()
        .sort_values("Input Time Position")
    )
    total = float(grouped["Mean Absolute SHAP"].sum()) or 1.0
    return [
        {
            "position": int(row["Input Time Position"]),
            "mean_absolute": float(row["Mean Absolute SHAP"]),
            "share": float(row["Mean Absolute SHAP"]) / total,
            "lag_seconds": int(row["Input Time Position"]) - (SEQUENCE_LENGTH - 1),
        }
        for _, row in grouped.iterrows()
    ]


def _comparison_rows(frame, output: str | None, top_n: int) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    work = frame
    if "Predicted Output Feature" in work.columns and output:
        work = work[work["Predicted Output Feature"].astype(str) == output]
    if work.empty:
        return []
    grouped = (
        work.groupby("Input Feature", as_index=False)
        .agg(benign=("Benign", "mean"), bot=("Bot", "mean"), difference=("Absolute_Difference", "mean"))
        .sort_values("difference", ascending=False)
        .head(max(1, top_n))
    )
    return [
        {
            "feature": str(row["Input Feature"]),
            "benign": float(row["benign"]),
            "bot": float(row["bot"]),
            "difference": float(row["difference"]),
        }
        for _, row in grouped.iterrows()
    ]


def _local_archive(store: ArtifactStore, run_id: str) -> dict[str, Any] | None:
    artifact = store.run(run_id).table("shap_values") if store.run(run_id) else None
    if artifact is None or not artifact.present:
        return None
    try:
        with np.load(artifact.path, allow_pickle=False) as archive:
            required = {"shap_values", "explain_indices", "explain_labels", "output_indices", "feature_names", "output_names"}
            if not required.issubset(archive.files):
                return None
            return {
                "values": archive["shap_values"].copy(),
                "indices": archive["explain_indices"].copy(),
                "labels": archive["explain_labels"].copy().astype(str),
                "outputs": archive["output_names"].copy().astype(str),
                "features": archive["feature_names"].copy().astype(str),
                "path": str(artifact.path),
            }
    except (OSError, ValueError):
        return None


def _local(store: ArtifactStore, run_id: str, sample_index: int, output: str | None) -> dict[str, Any] | None:
    archive = _local_archive(store, run_id)
    if archive is None:
        return None
    count = archive["values"].shape[0]
    if count == 0:
        return None
    sample_index = max(0, min(int(sample_index), count - 1))
    outputs = archive["outputs"].tolist()
    output_name = output if output in outputs else outputs[0]
    output_position = outputs.index(output_name)
    values = archive["values"][sample_index, :, :, output_position]
    features = archive["features"].tolist()
    total = values.sum(axis=0)
    order = np.argsort(-np.abs(total))[:25]

    replay_meta: dict[str, Any] = {}
    from .replay import load_replay

    replay = load_replay(store, run_id)
    if replay is not None:
        replay_step = int(archive["indices"][sample_index]) + 1
        try:
            steps = replay.steps()
            matches = steps[steps == replay_step]
            if len(matches):
                position = matches.index[0]
                timestamp = replay.timestamps().loc[position]
                replay_meta = {
                    "replay_step": replay_step,
                    "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else None,
                    "dataset_label": str(replay.labels().loc[position]),
                }
        except Exception:  # pragma: no cover - defensive, alignment is optional
            replay_meta = {"replay_step": replay_step}

    return {
        "sample_index": sample_index,
        "sample_count": int(count),
        "explanation_index": int(archive["indices"][sample_index]),
        "label": str(archive["labels"][sample_index]),
        "output": output_name,
        "outputs": outputs,
        **replay_meta,
        "contributions": [
            {
                "feature": features[index],
                "value": None,
                "contribution": float(total[index]),
                "direction": "increase" if total[index] > 0 else ("decrease" if total[index] < 0 else "neutral"),
                "rank": rank,
            }
            for rank, index in enumerate(order, start=1)
        ],
        "temporal": [
            {"position": position, "contribution": float(values[position, :].sum())}
            for position in range(values.shape[0])
        ],
        "feature_matrix": [
            {
                "feature": features[index],
                "values": [float(value) for value in values[:, index]],
            }
            for index in order[:8]
        ],
        "source": archive["path"],
    }


def overview(
    store: ArtifactStore,
    run_id: str,
    label: str | None = None,
    output: str | None = None,
    sample_index: int = 0,
    top_n: int = 10,
) -> dict[str, Any]:
    run = store.run(run_id)
    if run is None:
        raise KeyError(run_id)
    summary = store.read_json(run_id, "shap") or {}
    importance_frame = store.read_table(run_id, "shap_importance")
    temporal_frame = store.read_table(run_id, "shap_temporal")
    comparison_frame = store.read_table(run_id, "shap_comparison")
    stability_frame = store.read_table(run_id, "shap_stability")
    archive = _local_archive(store, run_id)

    outputs = list(summary.get("output_features") or [])
    labels = ["Benign", "Bot"]
    if importance_frame is not None and "Target Label" in importance_frame.columns:
        labels = sorted(importance_frame["Target Label"].astype(str).unique().tolist())
    if importance_frame is not None and "Predicted Output Feature" in importance_frame.columns and not outputs:
        outputs = sorted(importance_frame["Predicted Output Feature"].astype(str).unique().tolist())

    selected_label = label if label in labels else (labels[0] if labels else None)
    selected_output = output if output in outputs else (outputs[0] if outputs else None)

    stability: list[dict[str, Any]] = []
    if stability_frame is not None and not stability_frame.empty:
        filtered = stability_frame
        if "Label" in filtered.columns and selected_label:
            filtered = filtered[filtered["Label"].astype(str) == selected_label]
        stability = filtered.to_dict(orient="records")

    samples: list[dict[str, Any]] = []
    if archive is not None:
        for position in range(len(archive["labels"])):
            samples.append(
                {
                    "sample_index": position,
                    "label": str(archive["labels"][position]),
                    "explanation_index": int(archive["indices"][position]),
                }
            )

    local = _local(store, run_id, sample_index, selected_output) if archive is not None else None
    available = bool(summary) or importance_frame is not None
    reason = None
    required: list[str] = []
    if not available:
        reason = "No SHAP artifacts are recorded for this run."
        required = [
            f"{run_id}/shap_explainability_v3/summary.json",
            f"{run_id}/shap_explainability_v3/shap_feature_importance_by_label.csv",
        ]
    elif importance_frame is None:
        reason = (
            "Aggregated SHAP importance tables are not present in this checkout; only the "
            "recorded experiment summary is available."
        )
        required = [f"{run_id}/shap_explainability_v3/shap_feature_importance_by_label.csv"]

    limitations = list(LIMITATIONS)
    if archive is None:
        limitations.append(
            "Local per-observation explanations need shap_values.npz, which is not present here."
        )

    return {
        "source": {
            "mode": "recorded_artifact" if available else "unavailable",
            "label": "Recorded SHAP explainability artifacts",
            "detail": str(run.directory),
        },
        "availability": {"available": available, "reason": reason, "required_artifacts": required},
        "method": {
            "method": summary.get("method"),
            "experiment": summary.get("experiment"),
            "checkpoint_epoch": summary.get("checkpoint_epoch"),
            "sequence_length": summary.get("sequence_length"),
            "input_features": summary.get("input_features"),
            "background_source": summary.get("background_source"),
            "background_size": summary.get("background_size"),
            "explanation_source": summary.get("explanation_source"),
            "benign_samples": summary.get("benign_explanation_samples"),
            "bot_samples": summary.get("bot_explanation_samples"),
            "nsamples": summary.get("nsamples"),
            "outputs": outputs,
            "interpretation": summary.get("interpretation"),
            "local_available": archive is not None,
            "selected_label": selected_label,
            "selected_output": selected_output,
            "available_labels": labels,
            "available_outputs": outputs,
        },
        "global_importance": _importance_rows(importance_frame, selected_label, selected_output, top_n),
        "temporal_importance": _temporal_rows(temporal_frame, selected_label, selected_output),
        "label_comparison": _comparison_rows(comparison_frame, selected_output, top_n),
        "stability": stability,
        "samples": samples,
        "local": local,
        "limitations": limitations,
    }
