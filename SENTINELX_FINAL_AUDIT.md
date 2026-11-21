# SENTINELX — FINAL AUDIT

Scope: the repository at `feature/shap-dashboard-integration`, audited by running the
application, probing every API route and driving all twelve console routes in a real
browser. This audit records what exists, what works, what was broken, and what is
blocked by artifacts that are deliberately not committed.

- Product surface: FastAPI (`backend/`) + React/TypeScript SOC console (`frontend/`).
- Research reference: `sentinelx_dashboard.py` (Streamlit, unmodified).
- Primary model: V3 LSTM world model, 209 features, hidden 128, 1 layer, dropout 0.2,
  sequence length 10. V2 (212 features) artifacts are quarantined and never mixed in.

## 1. What is present in this checkout

| Area | State |
| --- | --- |
| Backend application | 30 modules, 6.5k lines, 36 tests passing |
| Frontend console | 12 routes, 5.3k lines, typecheck + production build clean |
| Model run directory | `Models/lstm_world_model_v3_gpu_run2` with config, training report, test evaluation, anomaly, triage, SHAP summaries, replay manifest + integrity audit |
| Datasets | **absent** (`Datasets/` is git-ignored) |
| Checkpoints (`best_model.pt`) | **absent** (git-ignored) |
| Row-level tables (`*.csv`) | **absent** (git-ignored) |
| Replay arrays (`replay_arrays.npz`) | **absent** (git-ignored) |
| Frozen test sequences | **absent** (git-ignored) |
| SHAP value archive / importance CSVs | **absent** (git-ignored) |

Consequence, stated plainly: every chart that needs row-level telemetry is empty in this
checkout **because the data is not here**, not because the chart is unimplemented. Charts
backed by recorded JSON/TXT summaries (test metrics, anomaly counts, multi-step report,
error quantiles, SHAP summary) render real values. This document separates those two cases.

## 2. API inventory (all routes served)

```
GET  /api/health                 GET  /api/capabilities        GET  /api/system/status
GET  /api/dashboard              GET  /api/network             GET  /api/timeline
GET  /api/predictions/summary    GET  /api/predictions/series  GET /api/predictions/top-residuals
GET  /api/replay/meta            GET  /api/replay/state        GET /api/replay/series
GET  /api/replay/contributors
GET  /api/explainability         GET  /api/events              GET /api/events/{id}
PATCH /api/events/{id}
GET  /api/models                 GET  /api/models/compare      GET /api/models/{run_id}
POST /api/models/{run_id}/activate
GET  /api/models/{run_id}/inference-status
POST /api/models/{run_id}/predict   POST /api/models/{run_id}/predict/raw
GET  /api/datasets               POST /api/datasets/upload     GET /api/datasets/validate
GET  /api/reports                POST /api/reports             GET /api/reports/{id}
GET  /api/reports/{id}/download  GET  /api/runs                GET /api/settings
PUT  /api/settings               GET  /api/audit
```

Dependency map (research → product):

```
CIC-IDS-2018 flows → 1 s network states (209 features) → 10-state sequences
   → LSTM world model → next-state forecast → mean residual score
   → frozen p99 threshold (1.88139677) → review queue / triage → analyst workflow
   → explainability (SHAP artifacts) → FastAPI services → console charts
```

No stage is recomputed by the API. Services only read validated artifacts and report
`availability` when one is missing; nothing is simulated.

## 3. Graph inventory

`requires row data` = needs the git-ignored tables; in this checkout it shows an explicit
empty state naming the artifact, never a populated placeholder.

| # | Graph | Page | Purpose | Data source | Requires row data | Status after this pass |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Residual activity | Dashboard | Mean residual per bucket + attack-labelled share | state/replay table | yes | implemented; empty state names `test_replay.csv` |
| 2 | Label distribution bars | Dashboard | Dataset reference label mix | replay/triage table | yes | implemented |
| 3 | Detection behaviour | Dashboard | TP/FN/precision/recall at frozen threshold | test anomaly summary | no | real values (TP 0, FN 20,216) |
| 4 | Residual activity | Network | Residual trend over capture | replay table | yes | implemented |
| 5 | Label mix bars | Network | Per-window label mix | replay table | yes | implemented |
| 6 | Protocol / port distribution | Network | Raw flow-plane analytics | raw flow CSV | yes | implemented, names the missing CSV |
| 7 | Actual vs predicted | Predictions | Observed vs forecast next state | replay table | yes | implemented |
| 8 | Residual histogram / quantile curve | Predictions | Error distribution | replay rows, else recorded quantiles | partial | **new fallback**: real p50–p99.9 curve when rows absent |
| 9 | Forecast error by horizon | Predictions | Recursive MAE vs persistence | multi-step report | no | **new chart**, real 1/2/5/10/30 s data |
| 10 | Highest-residual observations | Predictions | Review ranking | replay table | yes | **new panel**, backend existed with no UI |
| 11 | Per-label table / multi-step table | Predictions | Recorded evaluation | multi-step report | no | real values |
| 12 | Playback history + forecast | Playback | 10 inputs, actual t, predicted t | replay arrays + sequences | yes | implemented, verified in browser |
| 13 | Local residual trend + threshold | Playback | Window residuals vs frozen threshold | replay table | yes | implemented |
| 14 | Contributor ranking | Playback | Top residual contributors per state | replay table | yes | implemented |
| 15 | Actual vs predicted | Feature Explorer | One feature across the replay | replay table | yes | implemented |
| 16 | Signed/absolute residual | Feature Explorer | Direction and magnitude | replay table | yes | implemented |
| 17 | Actual-vs-predicted scatter | Feature Explorer | Tracking quality | replay table | yes | implemented |
| 18 | Contributor share bars | Feature Explorer | Feature frequency in top-5 errors | contributor table / columns | yes | implemented |
| 19 | Residual activity | Timeline | 5-minute window overview | replay table | yes | implemented |
| 20 | Global SHAP importance | Explainability | Mean |SHAP| per feature | importance CSV | yes | implemented, notice now marks partial artifacts |
| 21 | Temporal importance | Explainability | Attribution by input position | temporal CSV | yes | implemented |
| 22 | Local SHAP bars | Explainability | Per-observation attribution | `shap_values.npz` | yes | implemented |
| 23 | LSTM vs persistence | Model Management | Held-out MAE/RMSE/Huber | test evaluation JSON | no | **new chart**, real values |
| 24 | Multi-step error by horizon | Model drawer | Recursive error growth | multi-step report | no | **new chart** |
| 25 | Timestamped network-state playback | Playback | PLAY/PAUSE/RESET/prev/next/slider/speed | replay arrays | yes | verified below |

## 4. Defects found and fixed in this pass

1. **Playback required a filename the pipeline never writes.** It demanded
   `Datasets/sentinelx_test_sequences_v3.npy`; the frozen V3 split writes
   `sentinelx_test_X_seq10_v3.npy`. Fixed (commit `c6d4a06`), old name kept as fallback,
   dataset inventory corrected for train/validation/test arrays.
2. **Feature Explorer blanked the entire console.** `series.data?.window.total_states`
   threw on the unavailable payload (which has no `window`), and the uncaught render error
   unmounted the whole app with no visible message. Fixed the dereference and added a
   route-level `ErrorBoundary` so a page failure can never blank the console again.
3. **Playback could wedge at the end of data.** It called `setPlaying(false)` inside a
   `setStep` updater, leaving the button showing "Pause" while disabled. Timer and stop
   condition are now separate effects; PLAY at the last state restarts from the first.
4. **Partial availability was silent.** `explainability` returns `available: true` with a
   `reason` when only the summary exists; the notice was hidden. It now renders a
   "partial artifacts" warning.
5. **Palette drift** — Predictions and Explainability hard-coded `#5aa9f0`/`#3ed3b6`
   instead of the token palette; now `#3B82F6`/`#22D3EE`.

## 5. Data-blocked, with the exact artifact and command

| Feature | Missing artifact | Producing script |
| --- | --- | --- |
| Dashboard / Network / Timeline / Events / Predictions series / Explorer | `Models/<run>/historical_replay_v1/test_replay.csv` (or `explainable_triage_v2/explainable_triage.csv`, `triage_engine_v1/test_triage.csv`) | `scripts/evaluation/29_generate_historical_replay_v3.py`, `31_build_explainable_triage_v2.py` |
| Playback | `historical_replay_v1/replay_arrays.npz` + `Datasets/sentinelx_test_X_seq10_v3.npy` | `scripts/evaluation/29_generate_historical_replay_v3.py` |
| SHAP charts | `shap_feature_importance_by_label.csv`, `shap_temporal_importance_by_label.csv`, `shap_benign_bot_comparison.csv` | `scripts/evaluation/38_shap_lstm_world_model_v3.py` |
| Local SHAP | `shap_explainability_v3/shap_values.npz` | same |
| Protocol/port panel | raw CIC-IDS-2018 flow CSV under `Datasets/` | dataset download |
| Live inference (`/predict`, activation) | `Models/<run>/best_model.pt` + PyTorch | recorded training run |

None of these can be fabricated; the interface reports the gap by name.

## 6. Verification environment note

This checkout has no telemetry artifacts, so end-to-end UI verification of populated charts
and playback used the repository's **test fixtures** (`backend/tests/conftest.py`) served
from a throwaway API on `127.0.0.1:8000` with `SENTINELX_ROOT=/tmp/ws`, then the real API
was restored. Fixture data was never committed, never served by the product, and is clearly
labelled as such in the implementation report. Charts that use recorded artifacts were
verified against the real run directory.
