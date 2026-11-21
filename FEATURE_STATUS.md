# SentinelX — Feature Status

Legend: **Implemented** = works end-to-end with the artifacts present ·
**Conditional** = implemented, requires an artifact/credential not in this
checkout · **Planned** = not implemented and not claimed.

## Application

| Feature | Status | Notes |
| --- | --- | --- |
| SOC dashboard | Implemented | Posture, metrics, residual activity, model health, dataset readiness, recent candidates, limits |
| Network monitoring | Conditional | State-plane metrics with a replay table; protocol/port panel needs the raw flow CSV |
| Network state playback | Conditional | Play/pause/reset, first/previous/next/last, timeline slider, speed, step and timestamp, per-state contributors and full 209-feature table; needs `replay_arrays.npz` + `sentinelx_test_sequences_v3.npy` |
| Feature explorer | Conditional | Feature picker, actual vs predicted, signed/absolute residuals, largest errors, contributor ranking; needs the replay arrays |
| Predictive intelligence | Implemented | Recorded metrics, multi-step table, calibration; actual-vs-predicted needs the replay table |
| Threat triage | Conditional | Threshold candidates need a replay/triage CSV; empty state explains the frozen-threshold recall caveat |
| Analyst workflow | Implemented | Status, assignee, notes, append-only history in SQLite |
| Attack timeline | Conditional | Label phases and 5-min residual buckets from the replay table |
| Explainability centre | Conditional | Method/global/temporal need SHAP summary + CSVs; local view needs `shap_values.npz` |
| Original Streamlit parity | Implemented | Every original page maps to a console page: Overview→SOC Dashboard, Historical Replay→Network State Playback, Threat Triage→Threat Triage, Feature Explorer→Feature Explorer, SHAP→Explainability, Model & Dataset→Models/Datasets |
| Model management | Implemented | Registry, integrity checks, comparison, guarded activation; activation requires a checkpoint |
| Dataset management | Implemented | Artifact checklist, label coverage, upload + validation against the feature contract |
| Reports | Implemented | Markdown + JSON generation, preview, download |
| Settings | Implemented | Preferences, threshold provenance, deployment health, capability matrix |
| API documentation | Implemented | `/api/docs` (OpenAPI) |
| Write access control | Implemented | Optional `X-API-Key` via `SENTINELX_API_KEY` |

## Machine learning

| Feature | Status | Notes |
| --- | --- | --- |
| LSTM world model (V3) | Implemented | Unchanged architecture; 209/128/1/0.2, sequence 10 |
| One-step forecasting | Conditional | Needs checkpoint + preprocessing artifacts at runtime |
| Recursive multi-step evaluation | Implemented | `forecast_metrics.multi_step_*`, plus recorded report parsing |
| Persistence baseline | Implemented | NumPy baseline + recorded numbers surfaced in the UI |
| Residual anomaly scoring | Implemented | Mean absolute residual, benign p99 calibration, priority bands |
| Detection operating point | Implemented (descriptive) | Confusion matrix helper; UI shows the poor recall explicitly |
| SHAP explanations | Conditional | Recorded summaries parsed; tables/npz needed for charts/local view |
| Controlled experiments | Implemented | Variants, LR schedule, clipping, early stopping, guardrails, CLI |
| Promotion guardrail | Implemented | Validation-only, margin-based, blocks test-partition use |
| Live telemetry | **Planned** | No packet capture, no Zeek, no claim of real-time monitoring |
| Streaming inference | Implemented (library) | `SentinelXStreamingPredictor` on pre-aggregated states; not exposed as a UI feed |

## Quality and testing

| Item | Status | Notes |
| --- | --- | --- |
| Backend tests | Implemented | 24 pytest tests: contracts, availability semantics, workflow, reports, guards |
| Metrics tests | Implemented | 21 tests for forecast metrics and promotion guardrails |
| Existing tests | Preserved | Streaming (unit) and SHAP helper tests still pass; V3 end-to-end scripts need local artifacts |
| Frontend type-check + build | Implemented | `npm run build` succeeds |
| Structured logging | Implemented | Request log with method/path/status/duration |
| Error handling | Implemented | Typed handlers for artifact, not-found, validation and inference failures |
| Documentation | Implemented | README, audit, verification report, this checklist |

## Known incomplete items

1. Live/PCAP/Zeek ingestion — planned only.
2. Row-level UI pages need research CSVs that are git-ignored; every such page
   shows the exact missing paths.
3. Local SHAP view needs `shap_values.npz`.
4. Model activation and prediction endpoints need `best_model.pt` + frozen
   preprocessing artifacts.
5. No new performance numbers are claimed: the experiment harness is ready but
   was not run here (no dataset arrays, no GPU).
