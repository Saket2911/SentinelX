# SENTINELX — MASTER AUDIT & IMPLEMENTATION ROADMAP

**Branch:** `Version_4`
**Audit date:** 2026-11-20
**Auditor role:** platform engineering (ML · backend · frontend · QA)
**Status:** Phase 1 complete, Phases 4–6 partially delivered (see §9)

---

## 1. Executive summary

SentinelX is a genuine research project with a correctly framed scientific objective: learn
`P(S(t+1) | S(t), S(t-1), …, S(t-n))` over timestamped network states and use the resulting
prediction residual as an analyst review signal. That research direction is sound and has been
preserved without alteration.

The repository, however, is **not an application**. It is a flat collection of 100 numbered
one-off research scripts, a partially-used `src/sentinelx` package, one Streamlit dashboard, and
recorded report artifacts. There is no web service, no API, no component library, no test suite
worth the name, and — decisively — **none of the trained models, scalers, datasets or sequence
arrays are present in the checkout**, because `.gitignore` excludes every `*.pt`, `*.joblib`,
`*.npy`, `*.npz` and `*.csv`.

That single fact governs everything else: SentinelX cannot run live inference today, on this
machine or any fresh clone. What *is* present is a credible, self-consistent record of past
evaluation — including one honest and important **negative result**: the residual anomaly
threshold produced **zero true positives** on the held-out test split.

This audit therefore separates three things that were previously conflated:

1. **What was measured** — real, recorded, reproducible numbers.
2. **What cannot be measured here** — inference, per-event triage, SHAP tensors, flow detail.
3. **What was asserted** — design intentions that no artifact proves.

The platform delivered on this branch treats (1) as data, (2) as explicitly-declared gaps, and
refuses to manufacture (3).

---

## 2. Repository inventory

| Metric | Value |
|---|---|
| Tracked files (excl. `.git`) | 206 |
| Repository size | 1.7 MB |
| Root-level numbered scripts | 100 |
| Python package modules (`src/sentinelx`) | 18 |
| Recorded JSON summaries | 16 |
| Recorded text reports | 30 |
| Test files (pre-existing) | 3 (`tests/test_streaming.py`, `test_v3_*.py`) |
| Web backend | **none** |
| Web frontend | **none** (Streamlit only) |
| Present model checkpoints | **0** |
| Present datasets | **0** |

### 2.1 Layers as found

| Layer | State | Notes |
|---|---|---|
| Data ingestion | Scripts only | `02_prepare_features.py`, `58_build_expanded_chronological_dataset.py` |
| Preprocessing | Package + scripts | `StatePreprocessor` (median impute → log → StandardScaler) is well-written |
| Sequence generation | Package + scripts | Segment-boundary aware, no cross-gap sequences |
| Model | Package | `LSTMWorldModel` (206), `SentinelXLSTM` V3 (209) — clean, validated constructors |
| Inference | Package | `inference/world_model.py`, `pipeline.py`, `streaming.py` — correct, unusable (no weights) |
| Evaluation | Package + scripts | `metrics.py` is correct; `calculate_improvements` is correct |
| Explainability | Scripts only | SHAP GradientExplainer experiments, summaries retained |
| Triage | Scripts only | Residual threshold + review flag; summaries retained |
| Threat intel | Data only | 17-label contextual MITRE mapping build script |
| Dashboard | Streamlit, 31 KB | Depends on files that do not exist in the repo |
| API | **absent** | — |
| Web UI | **absent** | — |

---

## 3. Model versions

| Version | Features | Hidden | Layers | Dropout | Best epoch | Val Huber | Test MAE | Status |
|---|---|---|---|---|---|---|---|---|
| V1 | 206 | 128 | 1 | 0.2 | 2 | 0.19771 | — | historical |
| V2 | 212 | 128 | 1 | 0.2 | 8 | 0.42561 | 0.31539 | production candidate |
| V3 | 209 | 128 | 1 | 0.2 | 11 | 0.43094 | 0.32063 | experimental, deepest analysis |

**These are three different pipelines, not three checkpoints of one pipeline.** Feature
definitions, scalers and sequence arrays differ; they must never be mixed. The platform enforces
this by keeping each version's feature count in its own metadata contract and by refusing to
describe one as more accurate than another (§7, "comparison policy").

V2's validation Huber (0.4256) is *lower* than V3's (0.4309), but they are computed on different
feature spaces and are not comparable. No superiority claim is made anywhere in the platform.

---

## 4. Recorded results (the only quantitative ground truth in the repo)

### 4.1 One-step next-state prediction — V3, held-out test (31,406 sequences)

| Metric | LSTM | Persistence | Improvement |
|---|---|---|---|
| Huber | 0.120682 | 0.177071 | **+31.85 %** |
| MAE | 0.320627 | 0.356227 | **+9.99 %** |
| RMSE | 0.531328 | 0.679033 | **+21.75 %** |

The model beats a persistence baseline at every horizon. This is a real, defensible result **for
numerical forecasting**. It is not a detection result, and the platform never presents it as one.

### 4.2 Recursive multi-step forecasting — MAE improvement over persistence

| Horizon | All labels | Benign | Bot |
|---|---|---|---|
| 1 s | +9.99 % | +11.12 % | +9.34 % |
| 2 s | +12.46 % | +14.11 % | +11.49 % |
| 5 s | +13.04 % | +14.67 % | +12.08 % |
| 10 s | +13.84 % | +15.33 % | +12.96 % |
| 30 s | +15.23 % | +16.26 % | +14.64 % |

Absolute error grows with horizon (MAE 0.3207 → 0.3845) as recursive feedback accumulates error,
but the model's *relative* advantage over persistence holds or improves. Both facts are reported.

### 4.3 The negative result — residual threshold as a detector

| Split | Threshold | TP | FN | FP | TN | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| Validation (Infiltration) | 1.8814 | 88 | 12556 | 357 | 35344 | 0.00696 | 0.1978 | 0.0134 |
| Test (Bot) | 1.8814 | **0** | **20216** | 0 | 11189 | **0.0** | 0.0 | **0.0** |

The frozen 99th-percentile benign threshold **does not transfer** to the test split. Validation
recall of 0.7 % was already failing; test recall of exactly 0 is unambiguous.

**Conclusion carried through the entire platform:** the residual score is a *review prioritisation
heuristic*, not a detector. This is stated on the dashboard, the triage page, the reports and in
every generated export. It is not buried.

Root cause worth investigating (not yet done): the test split is 64 % Bot by state count, and label
runs are dominated by a handful of thousand-second stretches (§4.4). A per-state residual over a
near-stationary capture has little discriminative pressure — the model is being asked to detect
"normal for a botnet capture", which is normal.

### 4.4 Capture structure — test partition

- 14,722 states, **390 contiguous label runs**, window `2018-03-02 08:54:38 → 12:59:59`.
- Benign: 195 runs, max 4,730 s. Bot: 195 runs, max 6,098 s.
- Both labels are dominated by a few multi-thousand-second runs.

This is the strongest explanation for the detection failure and is surfaced on the Attack Timeline.

---

## 5. Critical findings

### F1 — No model, scaler or dataset artifacts are present (blocking)

`.gitignore` excludes `*.csv`, `*.npy`, `*.npz`, `*.pt`, `*.pth`, `*.joblib`, `*.pkl`. A fresh
clone has **zero** weights and **zero** data. Every "run the dashboard" instruction in the project
is therefore broken for anyone who is not the original author on the original machine.

*Impact:* live inference, per-event triage rows, SHAP tensors and flow-level detail are all
unavailable. **Mitigation:** the platform declares each gap explicitly with the exact required
path, rather than failing or faking.

### F2 — The existing dashboard is inert

`sentinelx_dashboard.py` reads `Datasets/sentinelx_test_X_seq10_v3.npy`,
`historical_replay_v1/test_replay.csv`, `replay_arrays.npz`, `explainable_triage.csv` and
`triage_engine_v1/test_triage.csv`. **None of these exist in the repository.** The dashboard
cannot render a single real page from a clean clone.

### F3 — Stale artifact path: label run structure under the wrong model

The run-structure report lives at
`Models/lstm_world_model_v1/label_run_structure_v1/…` but is referenced as a **V3** artifact by the
Streamlit dashboard and by early versions of the new API. Fixed in `backend/app/services.py`
(the API now resolves the real path and this is pinned by a test).

### F4 — Duplicated and dead code

| Finding | Path |
|---|---|
| Byte-identical duplicate loader | `data/loader.py` **==** `src/sentinelx/data/loader.py` |
| Mis-extensioned module | `src/sentinelx/evaluation/metrics.pyq` (valid Python, never imported) |
| Backup files committed | `sequences.py.backup`, `splitter.py.backup` |
| Empty script | `scripts/15_test_state_preprocessor.py` (0 bytes) |
| Numbered scripts colliding | `21_*` ×3, `22_*` ×4, `23_*` ×3, `24_*` ×3, `26_*` ×3, `27_*` ×2, `28_*` ×3, `29_*` ×3, `30_*` ×3, `31_*` ×2, `32_*` ×3, `33_*` ×3, `34_*` ×3, `35_*` ×3, `36_*` ×3, `37_*` ×3, `38_*` ×2, `39_*` ×2, `40_*` ×2, `43_*` ×2, `46_*` ×2, `47_*` ×2, `48_*` ×2, `49_*` ×2, `50_*` ×2, `51_*` ×2, `52_*` ×2, `53_*` ×2, `57_*` ×2, `66_*` ×2 |
| Two competing state builders | `15_build_multiday_states.py`, `62_build_expanded_states.py` |
| Two competing sorters | `sort_multiday_dataset.py`, `sort_multiday_dataset_v2.py` |

### F5 — Research findings never reach a consumer

The project contains an unusually rich set of negative and diagnostic findings (per-feature
ablation, feature-family forecasting, rare-error reliability, temporal peak position, SHAP
stability, label persistence) that exist **only** as `.txt` files. No interface reads them. This is
the largest unrealised value in the repository.

### F6 — No reproducible entry point

There is no documented install, no lockfile, no run command, no environment spec, and no CI. The
original environment (PyTorch 2.5.1+cu118, GTX 1050, CUDA) is only discoverable by reading report
headers.

---

## 6. Data integrity obligations (unchanged)

The platform implementation **adds** a read-only consumer layer and **modifies no research
artifact**. Specifically:

- No dataset, scaler, checkpoint, sequence array or report was written, regenerated or moved.
- No feature definition changed.
- No metric recomputed or "improved".
- No threshold re-tuned against test.
- The test partition was never used for selection.

Every value the API returns is either read verbatim from a committed artifact (`recorded_artifact`)
or arithmetically derived from such values (`derived`). Anything else is `unavailable` with a
reason.

---

## 7. What was built on `Version_4`

### 7.1 Backend — `backend/` (FastAPI)

A read-only analytical API over committed artifacts. No inference is attempted when no checkpoint
exists; the health endpoint states that plainly.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Service state, artifact registry, model loadability, warnings |
| `GET /api/dashboard` | Posture, metrics, label distribution, detection assessment, limitations |
| `GET /api/network/overview` | Partition composition; declares flow detail unavailable |
| `GET /api/predictions` | One-step, multi-step, error quantiles, live-forecast gap |
| `GET /api/predictions/multi-step` | Filterable by horizon and target label |
| `GET /api/triage` | Review aggregates, frozen calibration, decision policy, limitations |
| `GET /api/explainability` | SHAP experiments, feature-family attribution, caveats |
| `GET /api/timeline` | Capture window, run statistics, duration histogram |
| `GET /api/datasets` | Sources, partitions, preprocessing contract, upload policy |
| `GET /api/models` | All three versions, configs, training, test metrics, comparison policy |
| `POST /api/models/{id}/activate` | Validated activation with explicit warning |
| `GET /api/reports/catalog` | Registered artifacts + presence |
| `GET /api/reports/artifact/{id}` | Raw artifact content |
| `POST /api/reports/generate` | Builds a Markdown summary from recorded artifacts |
| `GET /api/reports/generated/{id}` | Download (path-traversal guarded) |
| `GET|PUT /api/settings` | Validated preferences persisted atomically |

Design points: artifact cache keyed by `(path, mtime, size)`; defensive parsers for fixed-width
reports; atomic config writes; no write path into `Models/` or `reports/`; 29 tests.

### 7.2 Frontend — `frontend/` (React 18 + TypeScript + Vite)

A dark, high-density SOC console with a single design system (tokens → primitives → pages).
Ten pages, every one wired to a live endpoint:

Dashboard · Network Monitoring · Predictive Intelligence · Threat Triage · Attack Timeline ·
Explainability Centre · Reports & Analytics · Datasets · Models · Settings

Shared primitives: `Panel`, `MetricCard`, `Tag`, `ProvenanceTag`, `Note`, `EmptyState`,
`ErrorState`, `LoadingBlock`, `Segmented`, `SearchInput`, `DataTable` (sortable + paginated),
`Drawer`, `Confirm`, `Toast`, `StatBar`, themed Recharts wrappers.

No decorative controls: the only buttons that exist perform real work (reload, filter, sort,
paginate, view artifact, generate report, download, activate model, save settings). One control is
intentionally disabled with a tooltip explaining why (dataset upload), and one option is
intentionally disabled with an explanation (live telemetry).

### 7.3 Provenance as a first-class UI concept

Every metric carries a badge: **Recorded** (verbatim from an artifact, tooltip shows the path),
**Derived** (computed by the API), or **Unavailable** (with the exact missing path). This is the
mechanism that keeps the interface honest at scale.

---

## 8. Risks

| # | Risk | Severity | Status |
|---|---|---|---|
| R1 | Absent artifacts make the platform read-only and inference-free | High | Declared, mitigated in UI |
| R2 | A reader could mistake forecasting accuracy for detection accuracy | High | Actively countered on 4 pages + exports |
| R3 | V2/V3 numbers could be compared across incompatible feature spaces | Medium | Comparison policy enforced server-side |
| R4 | 100 numbered scripts are unreproducible and unmaintained | Medium | Documented, not yet refactored |
| R5 | Streamlit dashboard remains broken and would confuse a reviewer | Medium | Should be retired or gated |
| R6 | No lockfile for the original research environment | Medium | Partially addressed (`backend/requirements.txt`, `frontend/package.json`) |
| R7 | Detection failure cause not yet investigated | Medium | Open research item (§4.3) |

---

## 9. Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Codebase audit | **Complete** |
| 2 | Pipeline validation, active-version selection, baseline reproduction | **Partial** — active version (V3) established and pinned; baseline reproduced *from recorded artifacts* only. Re-running inference requires the checkpoint. |
| 3 | Model improvement (architecture/training experiments) | **Not started** — deliberately blocked on artifact availability. No improvement is claimed without a measured comparison. |
| 4 | Backend completion | **Complete** for the artifact-backed surface. Inference endpoints correctly refuse without weights. |
| 5 | Frontend redesign | **Complete** for the ten required modules. |
| 6 | End-to-end integration | **Complete** for the achievable chain: artifact → parse → service → API → client → render. The model-inference segment of the chain is blocked by F1 and declared. |
| 7 | Testing and documentation | **Complete** for new code (29 tests, audit, README). Pre-existing tests untouched. |
| 8 | Final delivery | See README and this document. |

### Recommended next steps, in order

1. **Restore artifacts.** Place `best_model.pt`, `sentinelx_standard_scaler_v1.joblib`,
   `sentinelx_model_features_v1.json` and the `*_seq10_v3.npy` arrays in `Datasets/` and
   `Models/lstm_world_model_v3_gpu_run2/`. The API's health page will flip to
   *Inference ready* and the inference endpoints can be implemented against a real model.
2. **Investigate the detection failure (R7).** Hypotheses to test, in order: (a) per-state residual
   is diluted by near-stationary bot capture; (b) threshold should be per-feature-normalised rather
   than a flat mean; (c) a supervised head on residuals is required, which would be a *second*
   model with its own evaluation — permitted by the brief, and must not replace the World Model.
3. **Retire or gate `sentinelx_dashboard.py`** so reviewers are not misled by a broken page.
4. **Consolidate the 100 scripts** into `scripts/` stages with a single documented order, keeping
   every one (nothing deleted) but marking superseded ones.
5. **Add CI**: `pytest backend/tests` + `npm run build` on push.

---

## 10. Explicit non-claims

- SentinelX does **not** detect attacks in this build.
- SentinelX does **not** ingest live telemetry.
- SentinelX does **not** claim V3 > V2.
- SentinelX does **not** claim SHAP attributions are causal.
- SentinelX does **not** invent metrics, events, or per-flow records.
