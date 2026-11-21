"""SentinelX API application factory."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .db import Database
from .routers import admin, alerts, dashboard, intel, registry, system
from .services.artifacts import ArtifactError, ArtifactStore
from .services.settings_store import SettingsStore
from .services.system import VERSION


LOGGER = logging.getLogger("sentinelx.api")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="SentinelX API",
        version=VERSION,
        description=(
            "Predictive network-state modelling API for SentinelX. Responses separate recorded "
            "research artifacts from dataset-derived rows and report unavailable data explicitly. "
            "Prediction residuals are review signals, never attack confirmations."
        ),
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.settings = settings
    app.state.store = ArtifactStore(
        settings.models_dir, settings.reports_dir, settings.datasets_dir
    )
    settings.var_dir.mkdir(parents=True, exist_ok=True)
    app.state.database = Database(settings.database_path)
    app.state.settings_store = SettingsStore(settings.settings_path, settings.default_run_id)
    app.state.started_at = time.monotonic()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - re-raised after logging
            LOGGER.exception("unhandled error on %s %s", request.method, request.url.path)
            raise
        duration_ms = (time.perf_counter() - started) * 1000
        LOGGER.info(
            "%s %s -> %s (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(ArtifactError)
    async def artifact_error_handler(request: Request, exc: ArtifactError) -> JSONResponse:
        LOGGER.warning("artifact error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=503,
            content={"error": "artifact_unavailable", "detail": str(exc)},
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "detail": str(exc)},
        )

    for router in (
        system.router,
        dashboard.router,
        intel.router,
        alerts.router,
        registry.router,
        admin.router,
    ):
        app.include_router(router, prefix="/api")

    dist = settings.root / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
        LOGGER.info("Serving built frontend from %s", dist)

    return app


app = create_app()
