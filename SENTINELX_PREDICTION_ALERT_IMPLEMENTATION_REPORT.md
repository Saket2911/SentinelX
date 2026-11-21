# SentinelX — Prediction-Based Playback Alerting

Implementation report for prediction-residual alert generation in Network State Playback.

## 1. Branch

| Item | Value |
| --- | --- |
| Requested starting branch | `feature/shap-dashboard-integration` |
| Base commit verified | `64822c2` — identical to `origin/feature/shap-dashboard-integration` (`git ls-remote`) |
| Sandbox checkout branch | `freebuff/4cb66d9a57a674dc136b42d9` (cloud alias for the same commit; no tracked changes) |
| New working branch | `feature/prediction-based-playback-alerts` |
| Created from | base commit `64822c2`; original branch was not modified |
| Uncommitted work | Only untracked browser screenshots under `artifacts/`; left untouched and uncommitted |

The cloud checkout does not expose the remote branch name locally, so identity was
verified by commit hash: the checkout HEAD and the remote
`feature/shap-dashboard-integration` tip were both `64822c2` before branching.

## 2. What already existed and was reused

- Frozen **V3 LSTM world model** contract (209 features, 10-state sequence, hidden
  128, 1 layer, dropout 0.2) and its 10-state test sequences.
- `Models/<run>/historical_replay_v1/replay_arrays.npz` containing the LSTM
  forecast and observed next state for every replay position.
- `backend/app/services/playback.py`: per-state bundle, step alignment, segments,
  timestamps, labels, features and matrix cache.
- `backend/app/services/anomaly.py`: validation-calibrated frozen threshold
  (`1.88139677`, validation benign p99) and threshold source.
- `backend/app/db.py`: existing SQLite WAL store and audit-log convention.
- Existing FastAPI routers, dependency injection, write-access guard and response
  conventions.
- Existing SOC pages, charts, threat triage workflow, model/dataset views and SHAP
  explainability.

No second database, no duplicate table for existing state, no replacement of the
LSTM with a classifier, and no retraining.

## 3. New files

| File | Purpose |
| --- | --- |
| `backend/app/services/alerts.py` | Scoring, rule evaluation, SHAP context, idempotent alert persistence, alert summaries and anomaly history |
| `backend/app/routers/alerts.py` | Evaluate, list, detail, update and anomaly-history APIs |
| `backend/tests/test_alerts.py` | 21 new backend tests covering scoring, rules, persistence, dedupe, contracts and filters |
| `frontend/src/components/PlaybackAlert.tsx` | Shared severity pill, alert list, evidence/investigation drawer |
| `SENTINELX_PREDICTION_ALERT_IMPLEMENTATION_REPORT.md` | This report |

## 4. Modified files

| File | Change |
| --- | --- |
| `backend/app/db.py` | `playback_alert` and `playback_alert_history` tables plus idempotent insert, filtered listing, status update, counts and daily trend |
| `backend/app/services/playback.py` | Public `bundle()` accessor so alerting reuses the validated replay bundle/cache |
| `backend/app/schemas.py` | Evaluate/update request models; `DashboardResponse.alerts` |
| `backend/app/services/dashboard.py` | Real alert metrics, recent alerts and daily trend composed from SQLite |
| `backend/app/main.py` | Registers the alert router |
| `frontend/src/api/types.ts` | Alert, evaluation, anomaly-history and dashboard-alert types |
| `frontend/src/api/client.ts` | Typed evaluate/list/detail/update/anomaly APIs |
| `frontend/src/components/Charts.tsx` | `AnomalyScoreChart` (threshold + flagged/alert markers) and `AlertTrendChart` |
| `frontend/src/pages/ReplayPage.tsx` | Debounced per-position evaluation, alert banner/toast, timeline markers, session history, score/alert KPIs and detail drawer |
| `frontend/src/pages/PredictionsPage.tsx` | Anomaly score over time, threshold, flagged states and persisted-alert counts |
| `frontend/src/pages/ThreatsPage.tsx` | Generated playback alerts with investigation drawer and status updates |
| `frontend/src/pages/DashboardPage.tsx` | Alert metrics, daily alert trend and recent alerts |
| `frontend/src/styles/app.css` | Timeline alert markers |

## 5. Prediction-to-alert data flow

```
replay_arrays.npz (LSTM forecast for target t) + frozen 10-state sequence
        ↓
position lookup by replay step (no future states used)
        ↓
contract check: run config features/sequence == arrays (V2/V3 never mixed)
        ↓
10×209 input window validated (shape + finite values)
        ↓
actual[t] vs predicted[t] in standardized V3 feature space
        ↓
feature residuals + mean absolute residual anomaly score (+ MSE/RMSE)
        ↓
frozen calibrated threshold + persistence rule
        ↓
severity + trigger reason + stable alert ID
        ↓
SQLite idempotent persistence (run, model version, dataset, target step)
        ↓
API response → playback banner/toast/timeline marker
        ↓
Threat Triage, SOC Overview and Predictive Intelligence
```

### Prediction source (important)

This deployment uses the **LSTM forecasts recorded by the historical replay
pipeline**, not a fresh forward pass per UI step. They are genuine outputs of the
frozen V3 checkpoint for each recorded 10-state window. A fresh pass would require
PyTorch and the checkpoint, neither of which is present in this checkout
(`torch` is not installed and `best_model.pt` is git-ignored/absent). The API labels
the source as `recorded_lstm_replay` everywhere; nothing is simulated.

### Alignment, history and segments

- Each target state's input window is the recorded 10-state sequence for that
  position; feature order matches `replay_arrays.npz` feature names exactly.
- No future state is read for scoring; `actual[t]` is the observed target.
- Persistence is evaluated only across states of the same capture segment; it never
  spans a segment boundary.
- The context block reports input steps, segment IDs, consistency and a reference
  to the exact sequence array entry.
- Missing/short/non-finite windows are rejected with `insufficient_context`.
- A run whose `config.json` does not match the array feature count or sequence
  length is rejected with `model_contract_mismatch` (V2/V3 separation).

## 6. Anomaly-scoring method

- Metric: **mean absolute residual** across all model features
  (`mean(|actual − predicted|)`), in the standardized model feature space produced
  by the training-fitted V3 scaler.
- Additional reported values: MSE, RMSE, mean signed residual, max absolute error,
  and top feature-level signed/absolute residuals with percentage share.
- Because features are standardized on training data, they contribute on a
  comparable scale; the mean avoids one high-variance raw-unit feature dominating.
- The score is deliberately the same metric used by the recorded evaluation to
  calibrate the frozen threshold. Introducing a different uncalibrated score would
  invalidate that threshold.
- Non-finite states are rejected rather than silently scored.

## 7. Threshold calibration and alert rules

- Threshold: the run's frozen review threshold recorded by the triage artifacts,
  calibrated on **validation benign p99** (test partition isolated). No threshold is
  tuned on test data and no threshold was changed to increase alert counts.
- Stored in every alert: threshold value, threshold source, model version, rule
  version.
- Configurable via environment (defaults shown):
  - `SENTINELX_ALERT_PERSISTENCE=3`
  - `SENTINELX_ALERT_ELEVATED_RATIO=0.8`
  - `SENTINELX_ALERT_HIGH_RATIO=1.5`
  - `SENTINELX_ALERT_CRITICAL_RATIO=2.0`
- Triggers:
  1. single state at/above the calibrated threshold; or
  2. persistence: `N` consecutive same-segment states at/above `elevated_ratio ×
     threshold`.
- Severity is tied to the calibrated threshold, not arbitrary score bands:
  - **critical**: ratio ≥ 2.0, or persistent and ratio ≥ 1.5;
  - **high**: ratio ≥ 1.5, or any persistence trigger;
  - **medium**: single threshold crossing below 1.5×.
- Anomaly ≠ attack: all API responses, UI notices and the report say so.

### Recorded detection reality

The committed evaluation records **TP=0, FN=20216 (recall 0.0)** at the frozen
threshold on the test partition. Consequently the real held-out replay contains no
threshold crossing, and this implementation does **not** fabricate one. Alert logic
was verified with a controlled synthetic test fixture (see §11).

## 8. Database and API

New SQLite tables (no existing table replaced):

- `playback_alert` — unique `(run_id, model_version, dataset_id, target_step)`,
  alert ID, timestamp, segment, label, score, threshold, ratio, severity, trigger,
  investigation status, assignee, notes, feature evidence JSON, prediction/actual/
  context references, created/updated timestamps.
- `playback_alert_history` — append-only status/assignee/notes changes.

Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/playback/evaluate` | Score a target step; generate/persist alert if rules met (idempotent) |
| POST | `/api/playback/alerts/scan` | Vectorized scan of the whole replay for all rule-matching states; persists idempotently (capped at 500 by default) |
| GET | `/api/playback/alerts` | List filters: run, status, severity, dataset_id, target_step, since, until, limit, offset |
| GET | `/api/playback/alerts/{id}` | Alert detail, evidence, SHAP context and audit history |
| PATCH | `/api/playback/alerts/{id}` | Update investigation status/assignee/notes (audited) |
| GET | `/api/playback/anomalies` | Per-state score history, threshold, flagged steps, persisted alert steps |

Duplicate prevention is enforced in SQLite; a repeat evaluation returns the
existing alert with `deduplicated: true`.

### Why a playthrough could show no alerts

The calibrated threshold is the validation-benign p99 (`1.8814`) and the recorded
test evaluation had **recall 0.0** at it. In the observed capture, typical scores
sit around 0.28, so most states never satisfy the rule. A session that only
records alerts it personally visited therefore shows an empty list — even though
31k states exist and one review candidate may sit anywhere in them. The console
now scans the whole replay on page load (one idempotent request) and always shows
the replay-wide review queue, so rule hits are found without stepping through
tens of thousands of states.

### Why the dashboard showed every state as attack-labelled

The preferred row-level artifact is `explainable_triage.csv`, which carries the
residual and contributor columns but — in the shipped V3 run — no dataset label
column. `ReplayFrame.labels()` therefore returned `UNKNOWN` for all 31,406 rows,
and the network summary counted every row that was not literally “Benign” as an
attack. The labels do exist in `historical_replay_v1/test_replay.csv`, which
covers the same held-out rows.

The replay loader now joins the label column from the replay artifact by
`Target State Row` (falling back to positional alignment when both tables have
the same length). Unknown or empty labels are a separate category and are never
counted as attacks.

## 9. Frontend integration

- **Network Playback**: debounced one-request-per-settled-position evaluation;
  alert score/status KPIs; toast and inline notification; severity-coloured timeline
  markers that seek to the alert; session alert history; evidence drawer with
  predicted vs actual features, threshold ratio, trigger, references, SHAP context
  and investigation workflow. Pause/resume does not re-evaluate; reset/replay/seek
  reuse the persisted alert. A **replay-wide scan** runs once per run, so alerts
  are found even at 8× where individual states would be skipped. A **Replay review
  queue** lists the highest-residual states, dataset reference-label counts,
  threshold crossings, persisted alerts and links to Timeline/Triage.
- **Predictive Intelligence**: anomaly score chart with frozen threshold,
  flagged-state markers, persisted-alert markers, counts and links to replay.
- **Threat Triage**: generated alert table with status, severity and the same
  evidence drawer; updates the persisted investigation status.
- **SOC Overview**: total/high-priority/unresolved alert metrics, alert trend chart
  and recent alerts, all from SQLite.
- **Explainability**: the alert drawer shows recorded global SHAP magnitudes for the
  alert's residual features as supporting evidence, clearly separated from residual
  magnitude and with an explicit non-causality note.

## 10. Tests executed

Command: `cd backend && ../.venv/bin/python -m pytest tests -q` → **57 passed**
Command: `.venv/bin/python -m pytest tests -q` (root) → **50 passed**
Command: `cd frontend && npx tsc --noEmit` → clean
Command: `cd frontend && npm run build` → success (existing chunk-size warning only)

New tests include:

1. Ten-state input construction and context metadata.
2. Next-state alignment (`step`/`index`/timestamp).
3. Actual-vs-predicted comparison values.
4. Feature-order consistency and 209-feature width.
5. Residual / MSE / RMSE calculation.
6. Anomaly score and ranking.
7. Threshold evaluation below/above.
8. Persistence rule and segment-boundary exclusion.
9. Alert generation from an anomalous state.
10. Severity assignment (medium/high/critical).
11. SQLite persistence and retrieval.
12. Duplicate-alert prevention on repeat evaluation, seek and replay.
13–15. Playback pause/resume/seek/reset are covered through idempotent
   evaluate calls and verified in the browser request log (one evaluation per
   settled position).
16. Alert timestamp equals the target state timestamp.
17. Frontend notification/history verified in the browser (no frontend unit test
   runner exists in the repo).
18. SOC dashboard alert metrics and recent alerts.
19. Insufficient/non-finite context rejection.
20. Segment boundary handling.
21. V2/V3 contract mismatch rejection.
22. Filters by step, dataset and timestamp bounds.
23. Replay-wide scan finds and persists threshold candidates without playback.
24. Repeat scan creates no duplicates.
25. Scan finds persistence-rule candidates.
26. Anomaly history exposes a top-state review queue and label summary even when
    zero states are flagged.
27. When the explainable-triage table has no label column, labels are joined from
    the replay artifact and benign/attack counts stay correct.

Regression suites (playback, events, dashboard, auth, model registry, dataset
contracts, explainability) all pass.

## 11. Browser verification

Environment: FastAPI + Vite on the chat VM, using the repository's synthetic test
fixture workspace. To exercise alert generation without fabricating dataset
results, **one target state's forecast in the fixture was deliberately offset**
(mean absolute residual = 3.0). This is a controlled test fixture; the real
held-out dataset has no threshold crossing.

Verified in the browser (Chromium, 1600px):

1. Opened Network Playback and pressed Play.
2. Playback advanced through states with LSTM predictions and actual next states.
3. Anomaly score updated per state (e.g. 0.0522 below threshold).
4. Reached step 8 / `2018-03-02T10:00:07` (rendered in local time) with score 3.0.
5. Alert appeared without interrupting playback: toast “HIGH playback alert · step 8”,
   inline alert banner, timeline marker and session history row.
6. Opened alert details: threshold 1.8814, ratio 1.59×, trigger reason, segment,
   model version, predicted vs actual feature evidence, reference paths and SHAP
   context.
7. Updated status to `investigating` with a note; audit trail recorded
   `new → investigating`.
8. Paused, resumed, reset, replayed and sought to step 8 again: banner said the
   alert “was generated earlier; it was not created again”; API listing stayed at
   **1** alert.
9. Threat Triage showed the step-8 alert with severity High and status Investigating.
10. SOC Overview showed Playback alerts 1, High-priority 1, Unresolved 1, the daily
    trend and the recent alert.
11. Predictive Intelligence showed the anomaly score chart, threshold line, flagged
    step 8 link and persisted-alert count.
12. All 12 routes loaded with zero page/console errors after the change.

Replay-wide scan checks (clean fixture with no threshold crossing):

- Playback page auto-scanned 30 states and reported “no state satisfied the
  calibrated alert rules”.
- The review queue still listed the highest-residual states and the reference-label
  split (10 attack-labelled / 20 benign-labelled), with an explicit note that
  labels are annotations, not detections, and that recorded recall at this
  threshold is 0.
- After offsetting one fixture state, reloading playback auto-created exactly one
  alert (step 8), showed it in the queue with “Open alert”, and applied it to
  Threat Triage (High, 1) and SOC metrics (1 total / 1 high / 1 unresolved) without
  any manual step-by-step playthrough.
- A second scan created no duplicate.

Request-log check during play/pause/resume/seek: evaluations were one per settled
step (`2, 3, 4, 5, 8`) with no duplicate step requests and no duplicate alerts.

## 12. Honest status by requirement

| Requirement | Status |
| --- | --- |
| New branch from current branch, original untouched | **Verified** |
| Prediction-to-alert pipeline | **Implemented** (recorded LSTM forecasts; see limitation) |
| Previous-10-state input, no future states | **Verified** |
| Residual scoring + feature evidence | **Verified** |
| Calibrated threshold reuse, metadata stored, configurable | **Verified** |
| Single-state + persistence alert rules | **Verified** |
| Severity rules tied to threshold | **Verified** |
| Idempotent alert persistence + dedupe | **Verified** |
| Alert APIs (evaluate/list/detail/update/history/filters) | **Verified** |
| Playback alert UI, markers, notifications, detail | **Verified** (fixture) |
| Replay-wide scan + review queue (no playthrough needed) | **Verified** |
| Predictive Intelligence anomaly view | **Verified** |
| Threat Triage alert workflow | **Verified** |
| SOC Overview alert metrics/trend | **Verified** |
| SHAP context for alerts | **Implemented** (global recorded SHAP; not local per-alert attribution) |
| Real dataset alert demonstration | **Blocked** — recorded test detection has 0 threshold crossings |
| Live LSTM re-inference per UI step | **Blocked** — no PyTorch/checkpoint in this checkout |
| Dedicated “Anomaly Detection” page | **Implemented within Predictive Intelligence/Playback** (no new route) |
| Automated frontend unit tests | **Not started** — no frontend test runner configured |
| False-positive/detection evaluation of the new alert rule | **Blocked** — no validation row-level state table; only the committed test metrics exist |

## 13. Known limitations / remaining work

- Playback reuses recorded LSTM forecasts instead of recomputing them. On a
  deployment with the checkpoint and PyTorch, a cache-backed forward pass can be
  added behind the same evaluate API without changing the alert contract.
- No real alert can be shown from the shipped held-out data because the frozen
  threshold was never crossed (TP=0/FN=20216). Demonstrations use a clearly labelled
  synthetic fixture.
- The persistence rule is a defensible, documented heuristic; it has not been
  empirically validated against labelled validation rows (not present).
- SHAP evidence is global recorded importance for the alert's top residual
  features, not per-alert local SHAP. It is labelled as supporting context only.
- Alert time filtering is ISO string-based and does not parse time zones.
- The alert DB is local SQLite; there is no multi-user notification transport.
- Playback is historical dataset replay, never presented as live monitoring.
- The automatic replay scan persists at most 500 rule hits per request by default;
  larger deployments should page or schedule it. It runs when the playback page
  loads, not continuously in the background.
- If neither the preferred table nor the replay artifact carries a label column,
  states remain `UNKNOWN`; they are reported as unknown, never as attacks.
