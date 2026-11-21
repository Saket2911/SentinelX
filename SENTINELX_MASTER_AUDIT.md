# SENTINELX — MASTER AUDIT AND IMPLEMENTATION ROADMAP

Audit date: 2026-11-20
Branch audited: `feature/shap-dashboard-integration` @ `a1048ac` ("commit-4")
Scope: every committed file in the branch except `.quarantine/` (inspected, not executed)

This document is the Phase 1 deliverable. It records what exists, what works, what is
missing or misleading, and the roadmap that the implementation in this branch follows.
Nothing in the audit phase deletes, regenerates or rewrites research artifacts.

---

## 1. Repository state

| Item | Finding |
| --- | --- |
| Commits on branch | 2 (`b813579` initial, `a1048ac` commit-4) |
| Tracked files (non-quarantine) | 105 |
| Repo size on disk | ~1.9 MB (datasets, arrays and checkpoints are git-ignored) |
| Language | Python 3.12 (recorded env: 3.12.16), PyTorch 2.5.1+cu118 |
| Hardware recorded in artifacts | NVIDIA GTX 1050 (3 GB), CUDA 11.8; evaluation also run on CPU |
| Present locally | `Models/lstm_world_model_v3_gpu_run2/*` summary JSON + text reports, `reports/*` |
| Absent locally (git-ignored) | `Datasets/`, `best_model.pt`, `*.npz`, `*.joblib`, `*.csv` artifacts, MITRE mapping CSV |

Because artifacts are git-ignored, **the committed Streamlit dashboard cannot start from a
fresh clone**: it immediately requires `Models/.../test_replay.csv`,
`Models/.../explainable_triage_v2/*.csv` and `shap_explainability_v3/shap_values.npz`.
This is the single largest end-to-end gap and is addressed by the new artifact-aware API.

## 2. Existing architecture (as found)

```
Datasets/ (CIC-IDS-2018 CSVs, archive.zip)                [not in repo]
   -> scripts/pipeline/58_build_expanded_chronological_dataset.py
   -> scripts/pipeline/62_build_expanded_states.py          (1s state windows, 212 -> 209 features)
   -> scripts/pipeline/66_preprocess_expanded_states_v3.py  (median + signed-log + StandardScaler)
   -> scripts/training/21_train_lstm_world_model_v3.py      (SentinelXLSTM, 209/128/1/dropout 0.2)
   -> scripts/evaluation/22_evaluate_lstm_world_model_v3.py
   -> 23_/24_ anomaly, 28_ triage, 31_ explainable triage, 38_ SHAP, 46_/47_ multi-step
   -> sentinelx_dashboard.py / sentinelx_shap_dashboard.py  (Streamlit, monolithic)
```

`src/sentinelx/` is the reusable package: `data/` (loader, splitter), `features/`
(preprocessor, state_preprocessor, sequences, preprocessor_v3), `models/`
(lstm_world_model V1-era, lstm_world_model_v3), `training/`, `inference/`
(world_model, pipeline, streaming), `evaluation/` (metrics, evaluator, multi_step).
`explainability/` and `utils/` are empty packages.

## 3. Technology stack

- Python 3.12, PyTorch, NumPy, Pandas, scikit-learn, Joblib, SHAP 0.54.1, Streamlit
- No HTTP backend, no API schema, no database, no frontend build system
- Tests: `unittest` (4 files: streaming, V3 integration, V3 edge cases, SHAP dashboard helpers)
- MLflow/W&B: none. Experiment tracking is JSON + text reports per run directory.

## 4. Model versions (must never be mixed)

| Version | Features | Architecture | Checkpoint | Validation | Status |
| --- | --- | --- | --- | --- | --- |
| V1 | 212 | 2-layer LSTM, hidden 206/212 | `Models/lstm_world_model_v1` | — | retired, quarantined |
| V2 | 212 | 1 layer, hidden 128, dropout 0.2, batch 32, best epoch 8, val Huber 0.42561 | `Models/lstm_world_model_v2` | 50,870 | retired, quarantined (config only retained) |
| **V3** | **209** | **1 layer, hidden 128, dropout 0.2, batch 128, AdamW, Huber, best epoch 11, val Huber 0.43094** | **`Models/lstm_world_model_v3_gpu_run2/best_model.pt`** | **50,870** | **active research candidate** |

Recorded V3 held-out test results (31,406 sequences, model feature space):

| Metric | LSTM | Persistence baseline | Improvement |
| --- | ---: | ---: | ---: |
| MAE | 0.320627 | 0.356227 | 9.99 % |
| RMSE | 0.531328 | 0.679033 | 21.75 % |
| Huber | 0.120682 | 0.177071 | 31.85 % |

**Important caveats (preserved, not hidden):**
- The test partition has already been used for exploratory analysis; it is not untouched.
- The frozen residual threshold (1.88139677, benign validation p99) transfers poorly:
  test recall on `Bot` = 0.0, precision = 0.0, specificity = 1.0. Residual-only triage
  is *not* a validated detector and the UI must say so.
- Validation and test partitions have different label coverage (validation =
  Benign/Infilteration, test = Bot/Benign). `Bot` is absent from training.

## 5. Functional inventory

| Capability | Exists | Notes |
| --- | --- | --- |
| Dataset loading/validation | yes | `src/sentinelx/data/loader.py` |
| Chronological split | yes | `splitter.py` (+ stale `.backup`) |
| Flow -> 1 s state aggregation | yes | `features/sequences.py`, scripts 15/62 |
| 209-feature preprocessing (frozen) | yes | `features/preprocessor_v3.py`, strict artifact validation |
| Sequence generation (segment-safe) | yes | preallocated, sliding window |
| LSTM world model V3 | yes | input validation, dropout at head |
| Training loop | yes | V3 script: AdamW, Huber, early stopping, GPU chunking |
| One-step evaluation | yes | V3 eval + persistence baseline |
| Multi-step recursive evaluation | yes | horizons 1/2/5/10/30, per-label, feature-wise |
| Residual anomaly scoring | yes | p99 validation calibration, held-out test audit |
| SHAP explainability | yes | GradientExplainer, 64 samples, 5 outputs, stability study |
| Threat triage | yes | static CSV outputs; no analyst workflow/persistence |
| MITRE ATT&CK contextual mapping | yes | 17 labels, conditional/broad candidates, audited |
| Streaming predictor | yes | 10-state warmup, segment/gap reset; needs sklearn at import |
| Historical replay | yes | 31,406 steps, verified against direct inference |
| Dashboard | partial | Streamlit, 8 pages, requires missing artifacts; no API |
| API layer | **no** | — |
| Auth / access control | **no** | — |
| Reports generation/download | **no** | static text/MD only |
| Dataset upload/validation UI | **no** | — |
| Model activation/version compare | **no** | config JSON is read-only |
| Live telemetry | **no** | streaming consumes pre-aggregated states only |

## 6. Technical debt and defects found

1. `src/sentinelx/evaluation/metrics.pyq` — dead, mistyped file, committed.
2. `features/sequences.py.backup`, `data/splitter.py.backup` — committed duplicates that
   can silently diverge from the live modules.
3. `src/sentinelx/explainability/__init__.py` and `utils/__init__.py` are empty; SHAP logic
   lives only in scripts and in a 400-line dashboard helper module.
4. Inference is hard-coded to V3 (209/10/128/1) with strict equality checks, so V1/V2
   checkpoints cannot be loaded at all. Acceptable for production safety, but there is no
   explicit model registry to explain *why* a checkpoint is rejected.
5. `inference/pipeline.py` imports `preprocessor_v3` (joblib + sklearn) at module import;
   `streaming.py` imports pandas only but is re-exported through `__init__`. `__init__`
   already defers the heavy imports, yet `pipeline`/`preprocessor_v3` are not lazy.
6. Dashboard imports `sentinelx_shap_dashboard` at module load and calls `load_data()`
   at import time; missing CSVs raise and kill the process before any UI renders.
7. `sentinelx_shap_dashboard.load_shap_artifacts` hard-codes 64 explanations, 32/32
   Benign/Bot, 5 outputs and an exact replay alignment. Good validation, but tied to the
   one V3 run; any new run needs a new module.
8. `sentinelx_dashboard.py` (2,579 lines) mixes 400 lines of inline CSS, data access,
   analytics and 8 pages; no components, no pagination primitives, no error states,
   `st.rerun`-driven replay state.
9. No API: every page re-reads CSVs; no caching boundary, no schema contract, no auth,
   no logging, no health endpoints, no way for a second client to consume SentinelX.
10. Analyst actions (triage status, notes, investigation) exist only as static CSV columns;
    there is no persistence, audit trail or concurrency control.
11. `.gitignore` ignores `*.csv` globally, which is why the audited MITRE mapping CSV,
    triage CSVs and replay CSVs referenced by the summary JSONs are missing from the repo.
    This breaks reproducibility of the dashboard and of the MITRE evidence trail.
12. Requirements are Streamlit/SHAP/torch-centric; no backend, test-runner or type-check
    dependencies; `shap>=0.46` is only used by offline experiments.
13. Test coverage is thin: no tests for `SentinelXLSTM`, `predict_next_state`,
    `StatePreprocessor`, `make_sequences`, `evaluation/metrics`, or any UI/API surface.
14. `README.md` states the dashboard "requires final verification" and lists live capture as
    future work — accurate, but the README is the only documentation of limitations.

## 7. Risk register

| Risk | Impact | Mitigation in this implementation |
| --- | --- | --- |
| Missing artifacts look like "no data" / fake data temptation | Analysts misread output | Every endpoint returns explicit `availability` + `required_artifacts`; no synthetic telemetry |
| Residual threshold misread as attack detector | False confidence | Severity is a **review priority** derived from recorded quantiles; banner and limitation text everywhere |
| V3/V2 feature mixing | Invalid inference | Single registry keyed by `config.json`; feature count surfaced; no cross-run blending |
| Test-set reuse | Overstated performance | Metrics carry `test_partition_note`; no tuning code against test |
| Monolithic dashboard collapse | No demo possible | New API + SPA, Streamlit kept as legacy reference |
| Analyst actions lost | No workflow | SQLite-backed case state with audit trail |
| Uploaded dataset overwrites originals | Research corruption | Uploads land in `var/uploads/`, never touch `Datasets/` |

## 8. Implementation roadmap and status

| # | Work item | Status |
| --- | --- | --- |
| 1 | Master audit + roadmap (this document) | **done** |
| 2 | Feature/branch baseline preserved; no checkpoint or dataset modified | **done** |
| 3 | Artifact registry with strict availability semantics | **done** |
| 4 | FastAPI backend: dashboard, network, predictions, events, explainability, datasets, models, reports, settings, system | **done** |
| 5 | SQLite analyst workflow (status, notes, audit trail) | **done** |
| 6 | Residual review-priority calibration from recorded quantiles | **done** |
| 7 | MITRE mapping resolution (CSV, else generator source) | **done** |
| 8 | Structured evaluation module: MAE/MSE/RMSE/Huber, per-feature, percentiles, persistence + multi-step baselines | **done** |
| 9 | Controlled experiment harness (variants, guardrails, never overwrites checkpoints) | **done** |
| 10 | React/TypeScript SOC frontend, design system, 10 pages, all wired to the API | **done** |
| 11 | Backend + frontend tests, build and runtime verification | **done** (see VERIFICATION_REPORT.md for exact commands/results) |
| 12 | Report generation + download | **done** |
| 13 | Dataset upload/validation/compatibility checking | **done** (schema + feature contract) |
| 14 | Model registry, activation guardrails, version comparison | **done** |
| 15 | Live PCAP/Zeek ingestion | **not implemented** — documented as planned; no false claims |
| 16 | GPU retraining on the real dataset | **not executed here** — no `Datasets/` or checkpoints in this environment; harness ready |

## 9. Deliberate architectural decisions

1. **Backend + SPA replaces Streamlit as the product surface, Streamlit is retained.**
   The brief requires structured APIs, reusable components, a real backend and a
   commercial-grade UI. Streamlit cannot consume an API or share component contracts.
   `sentinelx_dashboard.py` is preserved unmodified as the research reference.
2. **Artifact-first data model, no fabricated telemetry.** The API serves only data that
   is actually on disk: recorded research summaries, and row-level CSVs when the researcher
   has generated them. Missing artifacts produce `availability.available = false` with the
   exact list of required files. Demo/synthetic data is never inserted.
3. **Research artifacts are read-only.** All mutable state (analyst workflow, settings,
   generated reports) lives under `var/`, which is git-ignored.
4. **Model promotion is gated**, not automatic: a run can only be activated when its
   `config.json`, checkpoint and 209-feature contract are present and internally consistent.
5. **The LSTM World Model stays the only predictive model.** The experiment harness may
   compare justified variants (residual head, stacked recurrent layer, LR schedule,
   clipping) but V3 remains the default and is never overwritten.

## 10. Known limitations of the delivered system

- Inference and training cannot run in this environment: `Datasets/`, `best_model.pt`,
  scaler and median artifacts are git-ignored and were not supplied. The API reports model
  health as "checkpoint unavailable" instead of inventing predictions.
- Row-level pages (Network, Predictions, Threats, Timeline, Explainability local view)
  require the replay/triage/explanation CSVs listed in each `availability` block.
- Live telemetry, PCAP parsing and Zeek ingestion are **not implemented**.
- Recorded detection metrics are poor and are displayed as such; they are not tuned here.
