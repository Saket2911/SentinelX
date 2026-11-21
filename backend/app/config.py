"""Runtime configuration for the SentinelX API.

All paths are resolved once at process start. Research artifacts are treated as
read-only inputs; anything the application writes (analyst workflow, generated
reports, uploads, settings) is confined to ``var_dir``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_RUN_ID = "lstm_world_model_v3_gpu_run2"


def _project_root() -> Path:
    # backend/app/config.py -> backend/app -> backend -> repository root
    return Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    root: Path
    models_dir: Path
    reports_dir: Path
    datasets_dir: Path
    var_dir: Path
    default_run_id: str
    api_key: str | None
    log_level: str
    cors_origins: tuple[str, ...]
    max_upload_mb: int
    default_page_size: int

    @property
    def uploads_dir(self) -> Path:
        return self.var_dir / "uploads"

    @property
    def generated_reports_dir(self) -> Path:
        return self.var_dir / "reports"

    @property
    def database_path(self) -> Path:
        return self.var_dir / "sentinelx.db"

    @property
    def settings_path(self) -> Path:
        return self.var_dir / "settings.json"

    @property
    def active_model_path(self) -> Path:
        return self.var_dir / "active_model.json"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def load(cls) -> "Settings":
        root = _env_path("SENTINELX_ROOT", _project_root())
        origins = tuple(
            origin.strip()
            for origin in os.environ.get(
                "SENTINELX_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if origin.strip()
        )
        return cls(
            root=root,
            models_dir=_env_path("SENTINELX_MODELS_DIR", root / "Models"),
            reports_dir=_env_path("SENTINELX_REPORTS_DIR", root / "reports"),
            datasets_dir=_env_path("SENTINELX_DATASETS_DIR", root / "Datasets"),
            var_dir=_env_path("SENTINELX_VAR_DIR", root / "var"),
            default_run_id=os.environ.get("SENTINELX_RUN_ID", DEFAULT_RUN_ID),
            api_key=os.environ.get("SENTINELX_API_KEY") or None,
            log_level=os.environ.get("SENTINELX_LOG_LEVEL", "INFO").upper(),
            cors_origins=origins,
            max_upload_mb=_env_int("SENTINELX_MAX_UPLOAD_MB", 64),
            default_page_size=_env_int("SENTINELX_PAGE_SIZE", 25),
        )
