"""SentinelX API entrypoint.

Run from the repository root:

    uvicorn backend.app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import VAR_DIR
from .routers import router
from .state import ensure_var_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("sentinelx")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_var_dirs()
    log.info("SentinelX API ready. Runtime state directory: %s", VAR_DIR)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="SentinelX API",
    version="4.0.0",
    description=(
        "Read-only analytical API over the SentinelX LSTM World Model research "
        "artifacts. Every numeric block states its provenance. No metric is "
        "simulated and no live telemetry is claimed."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https://.*\.style\.dev$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "sentinelx-api",
        "docs": "/docs",
        "api": "/api/health",
        "note": "Offline dataset mode. No live telemetry ingestion is implemented.",
    }
