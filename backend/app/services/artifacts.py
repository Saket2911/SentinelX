"""Read-only discovery of SentinelX research artifacts.

The API never regenerates research data. It discovers what a run produced on
disk, validates the shape of each artifact, and exposes availability so callers
can distinguish "not generated here" from "zero results".
"""

from __future__ import annotations

import ast
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# key -> (relative path, human label, kind)
RUN_ARTIFACTS: dict[str, tuple[str, str, str]] = {
    "config": ("config.json", "Run configuration", "json"),
    "training_report": ("training_report.txt", "Training report", "text"),
    "test_evaluation": (
        "test_evaluation/test_evaluation_summary.json",
        "Held-out test evaluation",
        "json",
    ),
    "validation_anomaly": (
        "validation_anomaly_analysis/anomaly_analysis_summary.json",
        "Validation residual anomaly analysis",
        "json",
    ),
    "test_anomaly": (
        "test_anomaly_evaluation/test_anomaly_summary.json",
        "Test residual anomaly evaluation",
        "json",
    ),
    "triage": ("triage_engine_v1/triage_summary.json", "Triage engine summary", "json"),
    "integrated_triage": (
        "integrated_triage_v1/integrated_triage_summary.json",
        "Integrated triage summary",
        "json",
    ),
    "explainable_triage_summary": (
        "explainable_triage_v2/summary.json",
        "Explainable triage summary",
        "json",
    ),
    "explanation_validation": (
        "explanation_validation_v1/summary.json",
        "Cross-split explanation validation",
        "json",
    ),
    "validation_attack_diagnostics": (
        "validation_attack_diagnostics_v1/summary.json",
        "Validation attack diagnostics",
        "json",
    ),
    "replay_manifest": (
        "historical_replay_v1/replay_manifest.json",
        "Historical replay manifest",
        "json",
    ),
    "replay_audit": (
        "historical_replay_v1/replay_integrity_audit.json",
        "Replay integrity audit",
        "json",
    ),
    "mitre_audit": (
        "mitre_attack_mapping_v1/mapping_artifact_audit.json",
        "MITRE mapping audit",
        "json",
    ),
    "shap": ("shap_explainability_v3/summary.json", "SHAP explanation summary", "json"),
    "shap_v2": ("shap_explainability_v2/summary.json", "SHAP explanation summary (v2)", "json"),
    "multi_step_report": (
        "multi_step_forecasting_v3_fixed/multi_step_forecasting_report.txt",
        "Multi-step forecasting report",
        "text",
    ),
    "feature_analysis_report": (
        "multi_step_feature_analysis_v1/feature_analysis_report.txt",
        "Feature-wise forecasting report",
        "text",
    ),
    "feature_family_report": (
        "multi_step_feature_analysis_v1/feature_family_report_v1.txt",
        "Feature family forecasting report",
        "text",
    ),
}

RUN_TABLES: dict[str, tuple[str, str]] = {
    "replay": (
        "historical_replay_v1/test_replay.csv",
        "Row-level historical replay",
    ),
    "replay_arrays": (
        "historical_replay_v1/replay_arrays.npz",
        "Replay arrays",
    ),
    "triage_test": ("triage_engine_v1/test_triage.csv", "Test triage rows"),
    "triage_validation": (
        "triage_engine_v1/validation_triage.csv",
        "Validation triage rows",
    ),
    "explainable_triage": (
        "explainable_triage_v2/explainable_triage.csv",
        "Explainable triage rows",
    ),
    "explainable_contributors": (
        "explainable_triage_v2/most_frequent_error_contributors.csv",
        "Frequent error contributors",
    ),
    "explainable_global": (
        "explainable_triage_v2/global_feature_error_summary.csv",
        "Global feature error summary",
    ),
    "mitre": (
        "mitre_attack_mapping_v1/mitre_attack_contextual_mapping.csv",
        "MITRE contextual mapping table",
    ),
    "shap_values": (
        "shap_explainability_v3/shap_values.npz",
        "SHAP value archive",
    ),
    "shap_importance": (
        "shap_explainability_v3/shap_feature_importance_by_label.csv",
        "SHAP feature importance",
    ),
    "shap_temporal": (
        "shap_explainability_v3/shap_temporal_importance_by_label.csv",
        "SHAP temporal importance",
    ),
    "shap_comparison": (
        "shap_explainability_v3/shap_benign_bot_comparison.csv",
        "SHAP benign/bot comparison",
    ),
    "shap_stability": (
        "shap_explainability_v3/stability_comparison_v2_v3.csv",
        "SHAP V2/V3 stability",
    ),
}

CHECKPOINT_NAMES = ("best_model.pt", "last_model.pt")

DATASET_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    ("model_features", "Model feature definition (V3)", "sentinelx_model_features_v3.json"),
    ("train_medians", "Training median imputation values", "sentinelx_train_medians_v3.npz"),
    ("scaler", "Frozen StandardScaler", "sentinelx_standard_scaler_v3.joblib"),
    ("train_metadata", "Training sequence metadata", "sentinelx_train_metadata_seq10_v3.csv"),
    ("validation_metadata", "Validation sequence metadata", "sentinelx_validation_metadata_seq10_v3.csv"),
    ("test_metadata", "Test sequence metadata", "sentinelx_test_metadata_seq10_v3.csv"),
    ("train_sequences", "Training sequence arrays", "sentinelx_train_X_seq10_v3.npy"),
    ("validation_sequences", "Validation sequence arrays", "sentinelx_validation_X_seq10_v3.npy"),
    ("test_sequences", "Test sequence arrays", "sentinelx_test_X_seq10_v3.npy"),
    ("multiday_states", "Expanded network states", "sentinelx_expanded_states_v3.csv"),
)


class ArtifactError(RuntimeError):
    """Raised when an artifact exists but cannot be used safely."""


@dataclass(frozen=True)
class Artifact:
    key: str
    label: str
    path: Path
    kind: str
    present: bool
    size_bytes: int | None = None

    @property
    def display_path(self) -> str:
        return str(self.path)


@dataclass
class ModelRun:
    run_id: str
    directory: Path
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    tables: dict[str, Artifact] = field(default_factory=dict)
    checkpoints: tuple[Path, ...] = ()

    def artifact(self, key: str) -> Artifact | None:
        return self.artifacts.get(key)

    def table(self, key: str) -> Artifact | None:
        return self.tables.get(key)

    @property
    def has_checkpoint(self) -> bool:
        return bool(self.checkpoints)

    def required(self, keys: Iterable[str], tables: Iterable[str] = ()) -> list[str]:
        missing = [
            f'{self.run_id}/{RUN_ARTIFACTS[key][0]}'
            for key in keys
            if key in RUN_ARTIFACTS and not self.artifacts[key].present
        ]
        missing.extend(
            f'{self.run_id}/{RUN_TABLES[key][0]}'
            for key in tables
            if key in RUN_TABLES and not self.tables[key].present
        )
        return missing


def _safe_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Unable to read JSON artifact {path}: {exc}") from exc


class ArtifactStore:
    """Discovers model runs and research artifacts. Thread-safe, mtime-cached."""

    def __init__(
        self,
        models_dir: Path,
        reports_dir: Path,
        datasets_dir: Path,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.datasets_dir = Path(datasets_dir)
        self._lock = threading.RLock()
        self._json_cache: dict[tuple[str, int, int], Any] = {}
        self._table_cache: dict[tuple[str, int, int], pd.DataFrame] = {}
        self._text_cache: dict[tuple[str, int, int], str] = {}
        self._runs: dict[str, ModelRun] | None = None

    # ------------------------------------------------------------------ discovery

    def refresh(self) -> None:
        with self._lock:
            self._runs = None
            self._json_cache.clear()
            self._table_cache.clear()
            self._text_cache.clear()

    def _discover(self) -> dict[str, ModelRun]:
        runs: dict[str, ModelRun] = {}
        if self.models_dir.is_dir():
            for directory in sorted(self.models_dir.iterdir()):
                if not directory.is_dir():
                    continue
                if not (directory / "config.json").is_file():
                    continue
                run = ModelRun(run_id=directory.name, directory=directory)
                for key, (relative, label, kind) in RUN_ARTIFACTS.items():
                    path = directory / relative
                    run.artifacts[key] = Artifact(
                        key=key,
                        label=label,
                        path=path,
                        kind=kind,
                        present=path.is_file(),
                        size_bytes=path.stat().st_size if path.is_file() else None,
                    )
                for key, (relative, label) in RUN_TABLES.items():
                    path = directory / relative
                    run.tables[key] = Artifact(
                        key=key,
                        label=label,
                        path=path,
                        kind="table" if path.suffix != ".npz" else "arrays",
                        present=path.is_file(),
                        size_bytes=path.stat().st_size if path.is_file() else None,
                    )
                checkpoint_paths = [
                    directory / name
                    for name in CHECKPOINT_NAMES
                    if (directory / name).is_file()
                ]
                if not checkpoint_paths:
                    checkpoint_paths = sorted(directory.glob("*.pt"))
                run.checkpoints = tuple(checkpoint_paths)
                runs[run.run_id] = run
        # A legacy V2 configuration without a run directory layout still counts.
        return runs

    @property
    def runs(self) -> dict[str, ModelRun]:
        with self._lock:
            if self._runs is None:
                self._runs = self._discover()
            return self._runs

    def run(self, run_id: str) -> ModelRun | None:
        return self.runs.get(run_id)

    def resolve_run_id(self, run_id: str | None, default: str) -> str:
        if run_id and run_id in self.runs:
            return run_id
        if default in self.runs:
            return default
        if self.runs:
            return next(iter(self.runs))
        raise ArtifactError(
            "No SentinelX model run was found under "
            f"{self.models_dir}. Expected a run directory containing config.json."
        )

    # ------------------------------------------------------------------ reading

    def read_json(self, run_id: str, key: str) -> Any | None:
        run = self.run(run_id)
        if run is None:
            return None
        artifact = run.artifact(key)
        if artifact is None or not artifact.present:
            return None
        stat = artifact.path.stat()
        cache_key = (str(artifact.path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if cache_key in self._json_cache:
                return self._json_cache[cache_key]
        value = _safe_json(artifact.path)
        with self._lock:
            self._json_cache[cache_key] = value
        return value

    def read_text(self, run_id: str, key: str) -> str | None:
        run = self.run(run_id)
        if run is None:
            return None
        artifact = run.artifact(key)
        if artifact is None or not artifact.present:
            return None
        stat = artifact.path.stat()
        cache_key = (str(artifact.path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if cache_key in self._text_cache:
                return self._text_cache[cache_key]
        try:
            value = artifact.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ArtifactError(f"Unable to read {artifact.path}: {exc}") from exc
        with self._lock:
            self._text_cache[cache_key] = value
        return value

    def read_table(self, run_id: str, key: str) -> pd.DataFrame | None:
        """Return a cached, read-only DataFrame. Callers must never mutate it."""
        run = self.run(run_id)
        if run is None:
            return None
        artifact = run.table(key)
        if artifact is None or not artifact.present:
            return None
        stat = artifact.path.stat()
        cache_key = (str(artifact.path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            if cache_key in self._table_cache:
                return self._table_cache[cache_key]
        try:
            frame = pd.read_csv(artifact.path, low_memory=False)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ArtifactError(f"Unable to read table {artifact.path}: {exc}") from exc
        frame.columns = [str(column).strip() for column in frame.columns]
        with self._lock:
            self._table_cache[cache_key] = frame
        return frame

    # ------------------------------------------------------------------ datasets

    def dataset_artifacts(self) -> list[Artifact]:
        artifacts: list[Artifact] = []
        for key, label, filename in DATASET_ARTIFACTS:
            path = self.datasets_dir / filename
            artifacts.append(
                Artifact(
                    key=key,
                    label=label,
                    path=path,
                    kind="table",
                    present=path.is_file(),
                    size_bytes=path.stat().st_size if path.is_file() else None,
                )
            )
        return artifacts

    def dataset_files(self) -> list[Path]:
        if not self.datasets_dir.is_dir():
            return []
        candidates: list[Path] = []
        for pattern in ("*.csv", "archive/*.csv", "raw/*.csv", "processed/*.csv"):
            candidates.extend(sorted(self.datasets_dir.glob(pattern)))
        return candidates

    def model_feature_names(self) -> list[str] | None:
        """Read the frozen V3 feature definition if present (never inferred)."""
        path = self.datasets_dir / "sentinelx_model_features_v3.json"
        if not path.is_file():
            return None
        try:
            names = _safe_json(path)
        except ArtifactError:
            return None
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return names
        return None

    # ------------------------------------------------------------------ context

    def label_coverage(self) -> dict[str, dict[str, int]]:
        path = self.reports_dir / "mitre_attack_contextual_mapping_v1" / "dataset_label_coverage_report.txt"
        if not path.is_file():
            return {}
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return {}
        from .research_text import parse_label_coverage

        return parse_label_coverage(text)

    def mitre_mappings(self, run_id: str) -> tuple[list[dict[str, Any]], str | None]:
        """Return contextual ATT&CK mappings and the source that produced them.

        Preference order: generated CSV table, then the committed generator that
        created it. The mapping is contextual guidance, never a model output.
        """
        table = self.read_table(run_id, "mitre")
        if table is not None and not table.empty:
            return table.to_dict(orient="records"), str(self.run(run_id).table("mitre").path)
        builder = self.reports_dir / "mitre_attack_contextual_mapping_v1" / "build_mapping.py"
        if not builder.is_file():
            return [], None
        try:
            tree = ast.parse(builder.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            return [], None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(getattr(target, "id", None) == "MAPPINGS" for target in node.targets):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return [], None
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)], str(builder)
        return [], None
