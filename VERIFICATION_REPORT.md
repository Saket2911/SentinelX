# SentinelX — Verification Report

Environment: Linux container, Python 3.12.3, Node 26.13.0, 2 vCPU, 3 GB RAM.
Repository: branch `feature/shap-dashboard-integration` @ `a1048ac`, plus the
work described in `SENTINELX_MASTER_AUDIT.md`.

Nothing in this report is copied from an earlier run: every command below was
executed on the delivered tree, and every failure/limitation is listed as such.

## 1. Commands and results

| # | Command | Result |
| --- | --- | --- |
| 1 | `python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt` | Pass — FastAPI 0.16x, pandas 3.1.1, NumPy 2.5.5, pytest, httpx |
| 2 | `.venv/bin/python -m pytest` (repo root) | **50 passed** |
| 3 | `cd backend && ../.venv/bin/python -m pytest` | **24 passed** |
| 4 | `cd frontend && npx tsc --noEmit` | Pass — no type errors (strict mode, `noUnusedLocals`) |
| 5 | `cd frontend && npm run build` | Pass — `dist/index.html` + hashed CSS/JS (678 kB JS, 190 kB gzip) |
| 6 | `uvicorn app.main:app --host 0.0.0.0 --port 8000` | Pass — API listening on 0.0.0.0:8000 |
| 7 | `GET http://127.0.0.1:8000/api/health` | 200 |
| 8 | `GET http://127.0.0.1:8000/api/dashboard` | 200, availability block present |
| 9 | `GET /` on port 8000 (built console via StaticFiles) | 200 |
| 10 | `vite --port 3001` (dev server, 0.0.0.0) | Pass — `LISTEN 0.0.0.0:3001` |
| 11 | `GET /` with the preview `Host` header | 200 |
| 12 | `GET /api/dashboard` through the Vite proxy with that Host | 200 |
| 13 | `cloud_show_dev_server(3001)` | Preview panel opened; **not** a rendered-page verification |

All 15 API surfaces were exercised during development with `fastapi.testclient`
against committed artifacts, and the same endpoints were requested over HTTP on
the running server. `GET /api/system/status` returned a 12 kB artifact inventory.

## 2. What the tests actually cover

### Backend (24 tests, `backend/tests/`)

Artifacts are created as **test fixtures** in a temporary directory (explicitly
synthetic, never served by a deployment) so the row-level paths are exercised:

- health, capabilities honesty (`live_telemetry == planned`), audit
- dashboard composition: fixture metrics, posture `review_required`, threshold
  propagation, benign/attack label counts
- empty deployment → `503 artifact_unavailable` instead of zeroed metrics
- prediction summary + parsed multi-step rows; per-label horizon-1 table
- prediction series (actual/predicted/residual/priority), unknown-feature note
- events: threshold selection, contributor columns, MITRE resolution, PATCH
  workflow (status/assignee/notes) with audit history, filtering, 404
- timeline buckets/phases and the "not reconstructed attack stages" limitation
- explainability global/temporal tables, `local_available == false` without npz
- model registry, integrity checks, comparison, activation blocked without a
  checkpoint, inference `503` without torch, wrong-shape `422`
- dataset upload + validation (contract unverifiable → not marked valid),
  oversized upload `413`, path traversal `404`
- report generation, Markdown content, download, invalid sections `422`
- settings round trip, calibration provenance, validation errors
- system status artifact inventory

### Research package (26 new tests + 24 existing)

- MAE/MSE/RMSE/Huber correctness against hand-computed values
- shape, dimensionality and finiteness rejection
- per-feature ordering, per-label aggregation, error quantiles
- persistence baseline equals the last observed state
- recursive multi-step forecasting and per-horizon metrics
- benign-only p99 threshold calibration
- detection confusion matrix including `NO_FLOWS` exclusion and the explicit
  "not a validated detector" interpretation string
- promotion guardrails: margin, checkpoint requirement, mixed gains, test-partition
  use, missing metrics

## 3. Not verified (and why)

| Item | Reason |
| --- | --- |
| Live inference / prediction output | No `best_model.pt`, no PyTorch, no `Datasets/` in this checkout. `POST /predict` returns `503` with a reason; the guard is tested. |
| Model activation success path | Blocked by the same missing checkpoint; the refusal is tested, and the success path requires artifacts. |
| Row-level dashboard pages with real data | `test_replay.csv`, triage and explanation CSVs are git-ignored and absent. The API's availability blocks name them; the fixture tests cover the populated paths. |
| Local SHAP view in the browser | Requires `shap_values.npz` (absent). Backend returns `local: null` and a limitation line. |
| Protocol/port network analytics | Requires the raw CIC-IDS-2018 flow CSV. |
| `tests/test_v3_edge_cases.py`, `tests/test_v3_integration.py` | Import-time scripts needing `Datasets/` + checkpoint; excluded from collection and documented. |
| Controlled experiment results | Cannot run without the real sequence arrays/GPU; harness and guardrails are tested with synthetic data only. |
| Browser click-through of every page | Not performed. The console build and API integration were verified; visual/interaction behaviour should be checked in the preview panel. |

## 4. Defects found and fixed during implementation

1. `network.py` used `str(value).str.lower()` twice — would have raised
   `AttributeError` on every dashboard/network request with data. Caught by tests.
2. `src/sentinelx/evaluation/__init__.py` imported torch eagerly, so the whole
   evaluation package (and any NumPy-only consumer) failed without PyTorch. Now
   lazy, matching `inference/__init__.py`.
3. Multi-step report parser assumed 2+ spaces between columns; the real report
   separates `Horizon_Seconds` and `Target_Label` with one space. Replaced with a
   token-aware parser, validated against the committed V3 report (31,377 origins
   at horizon 1).
4. Duplicate artifact key (`explainable_triage`) between JSON summaries and CSV
   tables caused the wrong status to be reported. Renamed the summary key.
5. Temporal SHAP lag was computed as `position - 10` (off-by-one); now
   `position - (sequence_length - 1)`.
6. Dataset validation marked uploads "valid" when the feature contract could not
   be checked. It now requires a verified contract.
7. Reports module had a fragile `__import__` for the dashboard service.
8. Frontend: conditional hook use in the model comparison flow (invalid React),
   an unsupported `Line` child inside `AreaChart` (now `ComposedChart`), unused
   imports failing strict TypeScript, and missing Node types for `vite.config.ts`.

## 5. UI/UX upgrade and restored replay capability

Scope: improve the console without removing pages, charts or workflows, and
restore the original Streamlit application's network-state playback and feature
explorer in the console.

| Item | Evidence |
| --- | --- |
| Per-state replay API | `GET /api/replay/meta`, `/state`, `/series`, `/contributors` serve `replay_arrays.npz` + `sentinelx_test_X_seq10_v3.npy` with availability blocks when absent |
| Playback page | `/replay`: play/pause, first/previous/next/last, reset, timeline slider, speed 1–8×, step counter, timestamp, segment, capture day, reference label, residual and threshold; charts update per state |
| Feature explorer | `/explorer`: feature picker, window navigation, MAE/median/max cards, actual-vs-predicted, signed/absolute residuals, scatter, largest-error table linking into playback, contributor ranking |
| Design system | Tokens re-based on the specified navy/charcoal palette; grouped sidebar, compact header with dataset mode, active model, review-candidate count and API health |
| Preserved pages | Dashboard, Network, Predictions, Threats, Timeline, Explainability, Models, Datasets, Reports, Settings all remain routed and unchanged in capability |
| Backend tests | `cd backend && ../.venv/bin/python -m pytest` → **35 passed** (8 new playback tests) |
| Frontend | `npx tsc --noEmit` clean; `npm run build` succeeds; `/`, `/replay`, `/explorer`, `/network`, `/predictions` return 200 through the preview host |

Playback could not be exercised with real values in this checkout: the arrays and
frozen test sequences are git-ignored and absent. The API returns
`availability.available = false` with the exact required artifact paths, and the
page states that nothing is simulated. Regenerate with
`python scripts/evaluation/29_generate_historical_replay_v3.py` where the
checkpoint and dataset exist. The browser was not opened for interaction or
visual review of the upgraded pages; only the build and HTTP routes were checked.

## 6. Reproduction

```bash
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
.venv/bin/python -m pytest
(cd backend && ../.venv/bin/python -m pytest)

cd frontend && npm install && npm run build

# single-process run (API + built console)
cd backend && ../.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# or split development run
cd frontend && npm run dev        # proxies /api to 127.0.0.1:8000
```
