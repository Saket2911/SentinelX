# SentinelX

**AI-Driven Automated SOC Analyst and Explainable Threat Triage Engine**

SentinelX learns temporal patterns in network behaviour and predicts the *next numerical network
state* rather than classifying events into attack categories. The prediction residual is then used
as an analyst review signal, explained with feature attribution, and presented in a security
operations console.

```
NETWORK FLOWS → FEATURE EXTRACTION → TIMESTAMPED STATES → TIME WINDOWS
     → LSTM WORLD MODEL → PREDICTED FUTURE STATE → RESIDUAL
     → ANOMALY SCORE → EXPLAINABILITY → TRIAGE → SOC CONSOLE
```

The objective is `P(S(t+1) | S(t), S(t-1), …, S(t-n))`.

---

## Important: what this checkout can and cannot do

The repository deliberately excludes large binaries (`*.pt`, `*.joblib`, `*.npy`, `*.csv`). As a
result **in this checkout**:

| Capability | Status |
|---|---|
| Read every recorded evaluation result | ✅ available |
| Inspect model versions, configs, training history | ✅ available |
| Explore multi-step forecasts, error quantiles, feature-family attribution | ✅ available |
| Browse label composition, capture structure, triage aggregates | ✅ available |
| Generate and download analytics reports | ✅ available |
| Load a trained checkpoint / run inference | ❌ checkpoint not present |
| Per-observation triage rows | ❌ CSV not present |
| SHAP value tensors | ❌ NPZ not present |
| Flow-level IP / port drill-down | ❌ raw CSVs not present |
| Live telemetry ingestion | ❌ not implemented |

Every number shown comes from a **committed artifact**. Nothing is simulated. Each metric is
tagged *Recorded*, *Derived* or *Unavailable* with the exact source path in a tooltip.

> **SentinelX does not detect attacks in this build.** The residual threshold produced **zero true
> positives** on the held-out test split. It is a review heuristic, and the interface says so on
> every page where it appears.

See [`SENTINELX_MASTER_AUDIT.md`](SENTINELX_MASTER_AUDIT.md) for the full audit, findings and
roadmap.

---

## Quick start

Requires Python 3.11+ and Node 18+.

### 1. Backend

```bash
python3 -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements.txt
backend/.venv/bin/uvicorn backend.app.main:app --port 8000
```

API docs: <http://127.0.0.1:8000/docs>

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Console: <http://localhost:5173> (Vite proxies `/api` → `127.0.0.1:8000`)

### 3. Tests

```bash
backend/.venv/bin/python -m pytest backend/tests -q      # backend: 29 tests
cd frontend && npx tsc --noEmit && npm run build          # typecheck + production build
```

---

## Architecture

```
backend/
  app/
    config.py       # paths + model version registry (single source of truth)
    provenance.py   # recorded / derived / unavailable vocabulary
    artifacts.py    # read-only artifact access + fixed-width report parsers
    services.py     # all domain logic (pure, unit-testable)
    routers.py      # thin HTTP layer
    schemas.py      # pydantic contracts + input validation
    state.py        # atomic settings + active-model persistence
  tests/test_api.py # 29 tests pinning recorded values and declared gaps
  var/              # runtime state (git-ignored)

frontend/src/
  lib/              # typed API client, formatters, fetch hook
  components/       # design system: ui.tsx, charts.tsx, icons.tsx
  pages/            # 10 modules, each wired to a live endpoint
  styles.css        # design tokens + all component styles

Models/             # recorded research artifacts (read-only, never written)
reports/            # recorded research artifacts (read-only, never written)
```

**Layering rule:** routers never parse; services never touch HTTP; only `artifacts.py` reads
research data, and it never writes.

### Provenance model

| Tag | Meaning |
|---|---|
| `recorded_artifact` | Copied verbatim from a committed JSON/TXT report |
| `derived` | Arithmetic computed by the API from recorded values |
| `unavailable` | Requires an artifact absent from this checkout; `reason` + `requires` explain |
| `live_telemetry` | Reserved. Never used — no live path exists. |

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Service state, artifact registry, loadability, warnings |
| GET | `/api/dashboard` | Posture, metrics, detection assessment, limitations |
| GET | `/api/network/overview` | Partition composition, flow-detail gap |
| GET | `/api/predictions` | One-step, multi-step, error quantiles |
| GET | `/api/predictions/multi-step?horizon=&label=` | Filtered forecast rows |
| GET | `/api/triage` | Review aggregates, frozen calibration, policy |
| GET | `/api/explainability` | SHAP experiments, feature-family attribution |
| GET | `/api/timeline` | Capture window, run statistics, histogram |
| GET | `/api/datasets` | Sources, partitions, preprocessing contract |
| GET | `/api/models` | All versions + comparison policy |
| POST | `/api/models/{id}/activate` | Validated activation |
| GET | `/api/reports/catalog` | Registered artifacts |
| GET | `/api/reports/artifact/{id}` | Raw artifact |
| POST | `/api/reports/generate` | Build Markdown summary |
| GET | `/api/reports/generated/{id}` | Download |
| GET/PUT | `/api/settings` | Validated, atomically persisted preferences |

---

## Restoring inference

To enable on-demand inference, place the git-ignored artifacts back:

```
Models/lstm_world_model_v3_gpu_run2/best_model.pt
Datasets/sentinelx_standard_scaler_v1.joblib
Datasets/sentinelx_model_features_v1.json
Datasets/sentinelx_{train,validation,test}_X_seq10_v3.npy
```

`GET /api/health` will report `inference_available: true` and the top-bar badge flips to
*Inference ready*. The existing inference modules (`src/sentinelx/inference/`) already implement
loading, validation and prediction; they were intentionally left untouched.

> Do not mix the 209-feature (V3) and 212-feature (V2) scalers, feature lists or sequence arrays.

---

## Data integrity guarantees

- No dataset, scaler, checkpoint or sequence array is written, moved or regenerated.
- No feature definition changed. No metric recomputed.
- No threshold tuned against the test split.
- Model activation changes reporting context only; it never retrains or copies a checkpoint.
- Uploads are disabled so a new dataset cannot silently replace audited training artifacts.

---

## Research record

| | |
|---|---|
| Active version | V3 · 209 features · seq 10 · hidden 128 · best epoch 11 |
| Test MAE / RMSE / Huber | 0.3206 / 0.5313 / 0.1207 |
| vs persistence | +9.99 % MAE · +21.75 % RMSE · +31.85 % Huber |
| Test residual detection | TP 0 / FN 20,216 / recall 0.0 — **not a validated detector** |
| Explainability | SHAP GradientExplainer, benign training background, 5 outputs |
| Threat intel | 17-label contextual MITRE ATT&CK reference (not per-event) |
