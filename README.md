# SentinelX

**AI-Driven Automated SOC Analyst and Explainable Threat Triage Engine**

SentinelX learns how a network behaves over time and forecasts the next network
state with an LSTM world model. Residuals between forecast and observation,
explainability artifacts and dataset reference labels are presented to analysts
through a SOC console with an auditable review workflow.

> SentinelX predicts network states. It does **not** output attack
> classifications, attack stages or attack probabilities. A high prediction
> residual is a review prompt, never proof of malicious activity.

---

## 1. What the system does

```
CIC-IDS-2018 flow CSVs
        │  scripts/pipeline (chronology, validation, duplicate/missing audit)
        ▼
1-second network states (209 model features + metadata)
        │  frozen preprocessing: training medians → signed log1p → StandardScaler
        ▼
10-state sequences (segment-safe, no cross-segment windows)
        │
        ▼
LSTM World Model  P(S(t+1) | S(t), S(t-1), … S(t-9))
        │
        ├── one-step forecast → residual → residual review priority
        ├── recursive multi-step forecast (1/2/5/10/30 s) vs persistence baseline
        └── SHAP attributions (global, temporal, Benign/Bot, local)
                    │
                    ▼
SentinelX API (FastAPI)  →  SOC console (React + TypeScript)
                              ├── SOC dashboard, network monitoring
                              ├── network-state playback (play/pause/step/slider)
                              ├── feature explorer, prediction analysis, explainability
                              └── threat triage, timeline, models, datasets, reports
                    │
                    └── analyst workflow: status, assignee, notes, audit trail
```

The application surface is split into a structured HTTP API and a single-page
console. The original Streamlit dashboard (`sentinelx_dashboard.py`) is preserved
unmodified as the research reference; it is not the product surface.

## 2. Repository layout

```
backend/                    FastAPI application (services, routers, tests)
frontend/                   React + TypeScript SOC console (Vite)
src/sentinelx/              research package: data, features, models, training,
                            inference, evaluation, explainability
experiments/run_experiment.py   controlled training experiments with guardrails
scripts/pipeline|training|evaluation|diagnostics|legacy   research pipeline
tests/                      Python tests for streaming, SHAP artifacts, metrics
Models/                     model runs, recorded summaries and reports
reports/                    evaluation and analyst-facing research reports
sentinelx_dashboard.py      preserved Streamlit research dashboard
sentinelx_shap_dashboard.py preserved SHAP artifact loader/view helpers
.quarantine/                reversible retirement of V1/V2 artifacts
SENTINELX_MASTER_AUDIT.md   Phase 1 codebase audit and roadmap
VERIFICATION_REPORT.md      what was executed, with results and gaps
FEATURE_STATUS.md           implemented / partial / planned checklist
```

Data, NumPy arrays, checkpoints, scalers and SHAP archives are **not** committed
(see `.gitignore`). The API reports them as unavailable rather than inventing
values.

## 3. Quick start

### 3.1 API

```bash
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt
cd backend
../.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- OpenAPI: `http://127.0.0.1:8000/api/docs`
- Health: `GET /api/health`

### 3.2 Console

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to 127.0.0.1:8000
npm run build        # type-check + production bundle in frontend/dist
```

If `frontend/dist` exists, the API also serves the built console at `/`, so a
single-process deployment is possible:

```bash
cd backend && ../.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# console: http://127.0.0.1:8000/
```

### 3.3 Research pipeline (unchanged)

```bash
python -m pip install -r requirements.txt      # includes torch, shap, streamlit
PYTHONPATH=src python -m pytest tests          # metrics/streaming/SHAP tests
streamlit run sentinelx_dashboard.py           # legacy dashboard (needs artifacts)
```

## 4. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SENTINELX_ROOT` | repo root | base for default paths |
| `SENTINELX_MODELS_DIR` | `Models` | model runs (`<run>/config.json`) |
| `SENTINELX_DATASETS_DIR` | `Datasets` | datasets, scaler, medians, feature JSON |
| `SENTINELX_REPORTS_DIR` | `reports` | research reports parsed by the API |
| `SENTINELX_VAR_DIR` | `var` | mutable state: SQLite, uploads, settings, generated reports |
| `SENTINELX_RUN_ID` | `lstm_world_model_v3_gpu_run2` | default run |
| `SENTINELX_LOG_LEVEL` | `INFO` | logging level |
| `SENTINELX_CORS_ORIGINS` | localhost Vite origins | allowed browser origins |
| `SENTINELX_API_KEY` | unset | when set, mutations require `X-API-Key` |
| `SENTINELX_MAX_UPLOAD_MB` | `64` | dataset upload limit |
| `VITE_API_BASE` | `/api` | frontend API base |
| `SENTINELX_API_TARGET` | `http://127.0.0.1:8000` | Vite dev proxy target |
| `VITE_ALLOWED_HOSTS` | empty | exact hostnames allowed by the dev server |

Research artifacts are read-only. Everything the application writes lives under
`var/`.

## 5. API overview

All analytical responses carry `source` (where the bytes came from),
`availability` (present? if not, exactly which artifacts are missing) and
`limitations` (interpretation constraints).

| Area | Endpoints |
| --- | --- |
| System | `GET /api/health`, `/api/capabilities`, `/api/system/status`, `/api/runs`, `/api/audit` |
| Dashboard | `GET /api/dashboard` |
| Network | `GET /api/network` |
| Replay | `GET /api/replay/meta`, `/api/replay/state?step=`, `/api/replay/series?feature=`, `/api/replay/contributors` |
| Predictions | `GET /api/predictions/summary`, `/api/predictions/series`, `/api/predictions/top-residuals` |
| Triage | `GET /api/events`, `GET/PATCH /api/events/{id}`, `GET /api/timeline` |
| Explainability | `GET /api/explainability?label=&output=&sample_index=` |
| Models | `GET /api/models`, `/api/models/{run}/`, `/api/models/compare`, `POST /api/models/{run}/activate`, `GET /api/models/{run}/inference-status`, `POST /api/models/{run}/predict[/raw]` |
| Datasets | `GET /api/datasets`, `GET /api/datasets/validate`, `POST /api/datasets/upload` |
| Reports | `GET/POST /api/reports`, `GET /api/reports/{id}`, `GET /api/reports/{id}/download` |
| Settings | `GET/PUT /api/settings` |

Mutating endpoints (`PATCH /events/{id}`, `POST /reports`, `POST /datasets/upload`,
`POST /models/{run}/activate`, `POST .../predict`, `PUT /settings`) require
`X-API-Key` when `SENTINELX_API_KEY` is configured.

## 6. Model card (V3, active research candidate)

| Item | Value |
| --- | --- |
| Architecture | LSTM world model, input 209 → hidden 128 → output 209 |
| Layers / dropout | 1 / 0.2 (dropout before the linear head) |
| Sequence | 10 one-second states |
| Training | AdamW, Huber loss, lr 1e-3, batch 128, ≤30 epochs, patience 6, seed 42 |
| Best epoch | 11 (validation Huber 0.43094) |
| Data | 197,389 train / 50,870 validation sequences, GTX 1050 3 GB |

Recorded held-out test comparison (31,406 sequences, standardized model space):

| Metric | LSTM | Persistence | Improvement |
| --- | ---: | ---: | ---: |
| MAE | 0.320627 | 0.356227 | 9.99 % |
| RMSE | 0.531328 | 0.679033 | 21.75 % |
| Huber | 0.120682 | 0.177071 | 31.85 % |

**Caveats, preserved deliberately:**
- The test partition was already used for exploratory analysis; it is not an
  untouched independent test.
- The frozen residual threshold (1.88139677 = benign validation p99) recalled
  **0 of the labelled Bot observations** on test (precision 0, specificity 1).
  It ranks observations for review; it is not a validated detector.
- Validation and test partitions have different label coverage; `Bot` is absent
  from training.
- Metrics are in preprocessed model feature space, not raw units.

Version separation is enforced: V3 uses 209 features and
`sentinelx_standard_scaler_v3.joblib`; V2/V1 used 212 features and are
quarantined. Runs are never blended.

## 7. Model improvement work added here

- `src/sentinelx/evaluation/forecast_metrics.py` — MAE/MSE/RMSE/Huber, per-feature
  and per-label errors, error quantiles, persistence baseline, recursive
  multi-step forecasting and evaluation, benign-only threshold calibration, and
  descriptive detection metrics. NumPy only, fully unit-tested.
- `src/sentinelx/models/variants.py` — `baseline_v3`, `residual_head`
  (persistence-prior delta head) and `stacked_two` variants for controlled tests.
- `src/sentinelx/training/experiments.py` — reproducible training with warmup +
  cosine schedule, gradient clipping, weight decay, early stopping, per-experiment
  checkpoint directories, and a conservative promotion guardrail (validation only;
  a test-partition flag blocks promotion).
- `experiments/run_experiment.py` — CLI that refuses to run without the real
  sequence arrays and never overwrites an existing checkpoint.

Measured numbers for new variants are intentionally absent: this environment has
no `Datasets/` arrays and no GPU, so the harness is delivered ready to run rather
than reported with fabricated results.

## 8. Testing

```bash
# research metrics / streaming / SHAP helpers
.venv/bin/python -m pytest                 # 50 tests

# backend API + services
cd backend && ../.venv/bin/python -m pytest   # 24 tests

# frontend type-check + production build
cd frontend && npm run build
```

`tests/test_v3_edge_cases.py` and `tests/test_v3_integration.py` are manual
end-to-end scripts (they execute at import and need `Datasets/` plus
`best_model.pt`); they are excluded from automatic collection and can be run
directly once artifacts exist.

## 9. Limitations

- Live capture, PCAP parsing and Zeek ingestion are **not implemented**. Streaming
  inference consumes pre-aggregated 209-feature state rows.
- Inference requires a local `.pt` checkpoint plus the frozen scaler/median/feature
  artifacts; git does not carry them.
- Row-level pages require the replay/triage/explanation CSVs produced by the
  research scripts; the API names the missing artifact instead of showing zeros.
- Residual triage is a heuristic with poor measured recall; dataset labels are
  reference annotations.
- MITRE ATT&CK mappings are contextual candidates for dataset labels, not
  per-observation classifications.
- The quarantine directory keeps retired V1/V2 code for reproducibility; it is not
  executed by the application.

## 10. Documentation

- `SENTINELX_MASTER_AUDIT.md` — Phase 1 audit, defects, risk register, roadmap
- `VERIFICATION_REPORT.md` — commands executed and their exact results/gaps
- `FEATURE_STATUS.md` — implemented / partial / planned checklist
- `reports/` — research evaluation reports retained from the original pipeline
