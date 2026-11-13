from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# SENTINELX — MULTI-PAGE SOC ANALYST INTERFACE
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Models/lstm_world_model_v3_gpu_run2"
REPLAY_DIR = MODEL_DIR / "historical_replay_v1"
EXPLAIN_DIR = MODEL_DIR / "explainable_triage_v2"
TRIAGE_DIR = MODEL_DIR / "triage_engine_v1"
DATA_DIR = ROOT / "Datasets"

REPLAY_CSV = REPLAY_DIR / "test_replay.csv"
REPLAY_NPZ = REPLAY_DIR / "replay_arrays.npz"
EXPLAIN_CSV = EXPLAIN_DIR / "explainable_triage.csv"
TRIAGE_CSV = TRIAGE_DIR / "test_triage.csv"
TEST_X = DATA_DIR / "sentinelx_test_X_seq10_v3.npy"

st.set_page_config(
    page_title="SentinelX | Security Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --sx-bg: #f4f7fb;
    --sx-panel: #ffffff;
    --sx-ink: #17283f;
    --sx-muted: #718096;
    --sx-line: #e5ebf2;
    --sx-teal: #159b89;
    --sx-navy: #101c2e;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--sx-bg);
}

header[data-testid="stHeader"] {
    background: rgba(244,247,251,0.92);
}

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #101c2e;
    border-right: 1px solid #26364a;
}

[data-testid="stSidebar"] > div:first-child {
    background: #101c2e;
}

[data-testid="stSidebar"] * {
    color: #d7e0eb;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #9aacc1;
}

[data-testid="stSidebar"] hr {
    border-color: #2a3a4e;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 9px 11px;
    border-radius: 9px;
    margin-bottom: 4px;
    font-size: 0.94rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #1d3047;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #203d50;
    color: #ffffff;
    font-weight: 700;
}

.sx-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 2px 20px 2px;
}

.sx-brand-icon {
    width: 42px;
    height: 42px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    color: #66ddc8;
    background: #1a3c4b;
    font-size: 23px;
}

.sx-brand-name {
    color: #ffffff;
    font-family: 'Manrope', sans-serif;
    font-size: 19px;
    font-weight: 800;
    letter-spacing: 1.4px;
}

.sx-brand-sub {
    color: #91a6bb;
    font-size: 9px;
    letter-spacing: 1.5px;
    margin-top: 1px;
}

.sx-eyebrow {
    color: #159b89;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.sx-page-title {
    color: #17283f;
    font-family: 'Manrope', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin: 0;
}

.sx-subtitle {
    color: #718096;
    font-size: 0.94rem;
    margin-top: 5px;
    margin-bottom: 22px;
}

.sx-card {
    background: #ffffff;
    border: 1px solid #e5ebf2;
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 112px;
    box-shadow: 0 2px 8px rgba(20, 40, 65, 0.025);
}

.sx-card-label {
    color: #718096;
    font-size: 0.79rem;
    font-weight: 600;
    margin-bottom: 9px;
}

.sx-card-value {
    color: #17283f;
    font-family: 'Manrope', sans-serif;
    font-size: 1.8rem;
    line-height: 1.2;
    font-weight: 800;
}

.sx-card-foot {
    color: #8795a8;
    font-size: 0.75rem;
    margin-top: 7px;
}

.sx-section {
    color: #1b2d43;
    font-family: 'Manrope', sans-serif;
    font-size: 1.08rem;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 3px;
}

.sx-section-sub {
    color: #8290a2;
    font-size: 0.82rem;
    margin-bottom: 14px;
}

.sx-panel {
    background: #ffffff;
    border: 1px solid #e5ebf2;
    border-radius: 14px;
    padding: 18px 20px;
}

.sx-status {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 20px;
    background: #e7f6f1;
    color: #147a68;
    font-size: 0.72rem;
    font-weight: 700;
}

.sx-warning {
    background: #fff7e8;
    border: 1px solid #f3dfb7;
    color: #8b641f;
    border-radius: 10px;
    padding: 12px 15px;
    font-size: 0.85rem;
}

.sx-footer {
    border-top: 1px solid #e5ebf2;
    margin-top: 35px;
    padding-top: 15px;
    color: #8b98a9;
    font-size: 0.75rem;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5ebf2;
    border-radius: 13px;
    padding: 15px 17px;
    box-shadow: 0 2px 8px rgba(20, 40, 65, 0.025);
}

div[data-testid="stMetricLabel"] {
    color: #718096;
    font-size: 0.8rem;
}

div[data-testid="stMetricValue"] {
    color: #17283f;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid #dce5ed;
    font-weight: 600;
    min-height: 40px;
}

.stButton > button[kind="primary"] {
    background: #147f73;
    border-color: #147f73;
}

.stButton > button[kind="primary"]:hover {
    background: #106c62;
    border-color: #106c62;
}

.stSelectbox label, .stSlider label, .stTextInput label {
    color: #46576c;
    font-weight: 600;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e5ebf2;
    border-radius: 10px;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING + ALIGNMENT VALIDATION
# ============================================================

@st.cache_data(show_spinner="Loading SentinelX replay data...")
def load_data():
    required = [
        REPLAY_CSV, REPLAY_NPZ, EXPLAIN_CSV, TRIAGE_CSV, TEST_X
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    replay = pd.read_csv(REPLAY_CSV)
    explanation = pd.read_csv(EXPLAIN_CSV)
    triage = pd.read_csv(TRIAGE_CSV)

    keys = [
        "Target State Row",
        "Target Timestamp",
        "Target CaptureDay",
        "Segment ID",
    ]

    for name, frame in [
        ("Replay", replay),
        ("Explanation", explanation),
        ("Triage", triage),
    ]:
        absent = [c for c in keys if c not in frame.columns]
        if absent:
            raise ValueError(f"{name} data missing columns: {absent}")

    merged = replay.merge(
        explanation,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_explanation"),
    )

    triage_columns = keys + [
        "Anomaly Score",
        "Threshold",
        "Triage Status",
        "Reason",
        "Interpretation",
    ]

    merged = merged.merge(
        triage[triage_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(replay):
        raise ValueError("Replay, explanation and triage records do not align.")

    merged = merged.sort_values("Replay Step").reset_index(drop=True)

    with np.load(REPLAY_NPZ, allow_pickle=False) as archive:
        predictions = archive["predictions"].copy()
        actual = archive["actual"].copy()
        feature_names = archive["feature_names"].astype(str)

    input_sequences = np.load(TEST_X, mmap_mode="r")

    if predictions.shape != actual.shape:
        raise ValueError("Prediction and actual arrays differ in shape.")
    if predictions.shape[0] != len(merged):
        raise ValueError("Replay array count does not match replay CSV.")
    if input_sequences.shape[0] != len(merged):
        raise ValueError("Input sequence count does not match replay CSV.")
    if predictions.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match prediction dimensions.")
    if input_sequences.shape[2] != len(feature_names):
        raise ValueError("Input features do not match feature names.")
    if not np.isfinite(predictions).all() or not np.isfinite(actual).all():
        raise ValueError("Prediction arrays contain non-finite values.")

    manifest_path = REPLAY_DIR / "replay_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    return merged, predictions, actual, feature_names, input_sequences, manifest


try:
    df, predictions, actual, feature_names, input_sequences, manifest = load_data()
except Exception as exc:
    st.error(f"SentinelX could not load its verified replay data: {exc}")
    st.stop()


# ============================================================
# REUSABLE UI HELPERS
# ============================================================

def page_header(section, title, subtitle):
    st.markdown(
        f"""
        <div class="sx-eyebrow">{section}</div>
        <div class="sx-page-title">{title}</div>
        <div class="sx-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, foot):
    st.markdown(
        f"""
        <div class="sx-card">
            <div class="sx-card-label">{label}</div>
            <div class="sx-card-value">{value}</div>
            <div class="sx-card-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, subtitle=None):
    st.markdown(f'<div class="sx-section">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f'<div class="sx-section-sub">{subtitle}</div>',
            unsafe_allow_html=True,
        )


def footer():
    st.markdown(
        """
        <div class="sx-footer">
            SENTINELX &nbsp;·&nbsp; Predictive Cyber-Defence Research Prototype
            <span style="float:right">Historical Replay V1</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_timestamp(value):
    return str(value)


def get_step():
    return int(st.session_state.get("replay_step", 1))


# ============================================================
# SIDEBAR / NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="sx-brand">
        <div class="sx-brand-icon">⬡</div>
        <div>
            <div class="sx-brand-name">SENTINELX</div>
            <div class="sx-brand-sub">PREDICTIVE SECURITY OPERATIONS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**WORKSPACE**")

    pages = [
        "Overview",
        "Historical Replay",
        "Threat Triage",
        "Feature Explorer",
        "Model & Dataset",
    ]

    default_page = st.session_state.pop("navigation_target", "Overview")
    page = st.radio(
        "Navigation",
        pages,
        index=pages.index(default_page) if default_page in pages else 0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**SYSTEM STATUS**")
    st.markdown(
        '<span class="sx-status">● MODEL CHECKPOINT LOADED</span>',
        unsafe_allow_html=True,
    )
    st.caption("LSTM World Model · V3")
    st.caption(f"{len(df):,} replay observations")
    st.caption("CIC-IDS-2018 · Held-out test")

    st.markdown("---")
    st.caption("SENTINELX RESEARCH BUILD")
    st.caption("Version 1.0 · Historical Replay")


# ============================================================
# OVERVIEW PAGE
# ============================================================

if page == "Overview":

    page_header(
        "COMMAND CENTER",
        "Security Overview",
        "A high-level view of SentinelX predictive performance and historical replay activity.",
    )

    total = len(df)
    mean_mae = float(df["Prediction MAE (model space)"].mean())
    review_count = int((df["Triage Status"] == "REVIEW_REQUIRED").sum())
    within_count = total - review_count
    review_rate = review_count / total * 100

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("REPLAY OBSERVATIONS", f"{total:,}", "Held-out test observations")
    with m2:
        metric_card("MEAN PREDICTION MAE", f"{mean_mae:.4f}", "Preprocessed feature space")
    with m3:
        metric_card("REVIEW REQUIRED", f"{review_count:,}", f"{review_rate:.3f}% of replay")
    with m4:
        metric_card("MODEL FEATURES", "209", "Network state features")

    st.write("")
    left, right = st.columns([1.65, 1])

    with left:
        section_title(
            "Prediction error over replay",
            "Rolling one-step-ahead LSTM prediction error across the held-out test sequence.",
        )
        trend = df[["Replay Step", "Prediction MAE (model space)"]].copy()
        trend = trend.set_index("Replay Step")
        trend = trend.iloc[::max(1, len(trend) // 1200)]
        st.area_chart(trend, height=300)

    with right:
        section_title(
            "Triage distribution",
            "Residual threshold outcomes. These are review states, not attack classifications.",
        )
        status_counts = df["Triage Status"].value_counts()
        st.bar_chart(status_counts, horizontal=True, height=220)

        st.markdown(
            f"""
            <div class="sx-warning">
            <b>Interpretation note</b><br>
            The residual-only triage threshold did not flag any Bot or
            Benign test observations. Do not interpret this workflow as
            a validated standalone attack detector.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    section_title(
        "Replay snapshot",
        "A small sample of the most recent observations in the replay dataset.",
    )

    recent = df.tail(8)[[
        "Replay Step",
        "Target Timestamp",
        "Target Label",
        "Prediction MAE (model space)",
        "Triage Status",
    ]].copy()

    recent["Triage Status"] = recent["Triage Status"].str.replace("_", " ")
    st.dataframe(recent, use_container_width=True, hide_index=True)

    footer()


# ============================================================
# HISTORICAL REPLAY PAGE
# ============================================================

elif page == "Historical Replay":

    page_header(
        "TEMPORAL WORLD MODEL",
        "Historical Replay",
        "Step through held-out network activity and inspect the model's next-state prediction.",
    )

    total = len(df)
    if "replay_step" not in st.session_state:
        st.session_state.replay_step = 1

    control1, control2, control3, control4 = st.columns([1, 1, 1, 1.5])

    with control1:
        if st.button("⏮ First", use_container_width=True):
            st.session_state.replay_step = 1
            st.rerun()

    with control2:
        if st.button("← Previous", use_container_width=True):
            st.session_state.replay_step = max(1, get_step() - 1)
            st.rerun()

    with control3:
        if st.button("Next →", type="primary", use_container_width=True):
            st.session_state.replay_step = min(total, get_step() + 1)
            st.rerun()

    with control4:
        if st.button("Last ⏭", use_container_width=True):
            st.session_state.replay_step = total
            st.rerun()

    step = st.slider(
        "Replay timeline",
        min_value=1,
        max_value=total,
        key="replay_step",
        help="Choose an observation from the held-out historical replay.",
    )

    idx = step - 1
    row = df.iloc[idx]
    segment = int(row["Segment ID"])

    st.markdown(
        f"""
        <div class="sx-panel">
            <div class="sx-eyebrow">CURRENT REPLAY POSITION</div>
            <b>{format_timestamp(row['Target Timestamp'])}</b>
            &nbsp; · &nbsp; Capture day: <b>{row['Target CaptureDay']}</b>
            &nbsp; · &nbsp; Segment: <b>{segment}</b>
            <span style="float:right;color:#159b89;font-weight:700">
                STEP {step:,} / {total:,}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    section_title(
        "Prediction summary",
        "The LSTM observes ten historical states and predicts one next state.",
    )

    mae = float(row["Prediction MAE (model space)"])
    threshold = float(row["Threshold"])

    a, b, c = st.columns(3)
    with a:
        metric_card("PREDICTION MAE", f"{mae:.5f}", "Across all 209 features")
    with b:
        metric_card("TRIAGE STATUS", str(row["Triage Status"]).replace("_", " "), "Frozen validation threshold")
    with c:
        metric_card("PREDICTION HORIZON", "1 state", "One-step-ahead prediction")

    st.write("")
    section_title(
        "Inspect a network feature",
        "Choose a feature to compare its ten observed input values with the predicted and actual next state.",
    )

    selected_feature = st.selectbox(
        "Network state feature",
        options=list(feature_names),
        index=list(feature_names).index("Flow Count")
        if "Flow Count" in feature_names else 0,
        key="replay_feature",
    )

    feature_idx = int(np.where(feature_names == selected_feature)[0][0])
    history = np.asarray(input_sequences[idx, :, feature_idx], dtype=np.float32)
    actual_value = float(actual[idx, feature_idx])
    predicted_value = float(predictions[idx, feature_idx])
    feature_error = abs(actual_value - predicted_value)

    x, y, z = st.columns(3)
    with x:
        metric_card("ACTUAL NEXT STATE", f"{actual_value:.5f}", "Observed target value")
    with y:
        metric_card("MODEL PREDICTION", f"{predicted_value:.5f}", "LSTM output")
    with z:
        metric_card("ABSOLUTE ERROR", f"{feature_error:.5f}", "Actual vs predicted")

    chart = pd.DataFrame(
        {
            "Observed history": history,
            "Predicted next state": [np.nan] * 10 + [predicted_value],
            "Actual next state": [np.nan] * 10 + [actual_value],
        },
        index=[f"t-{i}" for i in range(10, 0, -1)] + ["t"],
    )
    st.line_chart(chart, height=330)

    st.caption(
        "Values are in signed-log and standardized model feature space. "
        "They are not raw packet counts, bytes, ports or seconds."
    )

    st.write("")
    section_title(
        "Local prediction error",
        "Nearby prediction errors from the same segment, with the frozen validation threshold.",
    )

    local = df[
        (df["Segment ID"] == segment)
        & (df["Replay Step"] >= max(1, step - 30))
        & (df["Replay Step"] <= min(total, step + 30))
    ].copy()

    local_chart = local.set_index("Replay Step")[
        ["Prediction MAE (model space)"]
    ]
    local_chart["Frozen validation threshold"] = threshold
    st.line_chart(local_chart, height=250)

    with st.expander("View all 209 feature predictions for this observation"):
        feature_table = pd.DataFrame({
            "Feature": feature_names,
            "Actual": actual[idx],
            "Predicted": predictions[idx],
            "Absolute Error": np.abs(actual[idx] - predictions[idx]),
        }).sort_values("Absolute Error", ascending=False)

        st.dataframe(feature_table, use_container_width=True, hide_index=True)

    with st.expander("Dataset reference label"):
        st.write(f"**Reference label:** {row['Target Label']}")
        st.caption(
            "This is the original dataset label for retrospective analysis. "
            "It is not predicted by the LSTM and is not used by the triage rule."
        )

    footer()


# ============================================================
# THREAT TRIAGE PAGE
# ============================================================

elif page == "Threat Triage":

    page_header(
        "ANALYST WORKSPACE",
        "Threat Triage",
        "Review observations that crossed the frozen validation residual threshold.",
    )

    review_df = df[df["Triage Status"] == "REVIEW_REQUIRED"].copy()
    within_df = df[df["Triage Status"] != "REVIEW_REQUIRED"].copy()

    a, b, c = st.columns(3)
    with a:
        metric_card("REVIEW REQUIRED", f"{len(review_df):,}", "Residual threshold exceeded")
    with b:
        metric_card("WITHIN RANGE", f"{len(within_df):,}", "No residual review trigger")
    with c:
        metric_card("FROZEN THRESHOLD", f"{float(df['Threshold'].iloc[0]):.5f}", "Calibrated on validation data")

    st.write("")
    st.markdown(
        """
        <div class="sx-warning">
        <b>Important:</b> A residual-based review trigger is not proof of an
        attack. Likewise, being below the threshold does not guarantee benign activity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    section_title("Review queue", "Filter and inspect observations requiring residual-based review.")

    if len(review_df) == 0:
        st.success("No observations currently require residual-based review.")
    else:
        f1, f2 = st.columns([1, 1])
        with f1:
            selected_day = st.selectbox(
                "Capture day",
                ["All"] + sorted(review_df["Target CaptureDay"].astype(str).unique().tolist()),
            )
        with f2:
            search_step = st.text_input("Search replay step", placeholder="Enter a step number")

        filtered = review_df.copy()
        if selected_day != "All":
            filtered = filtered[filtered["Target CaptureDay"].astype(str) == selected_day]
        if search_step.strip():
            if search_step.strip().isdigit():
                filtered = filtered[
                    filtered["Replay Step"] == int(search_step.strip())
                ]
            else:
                st.info("Enter a numeric replay step.")

        display = filtered[[
            "Replay Step",
            "Target Timestamp",
            "Target CaptureDay",
            "Anomaly Score",
            "Threshold",
            "Triage Status",
        ]].copy()

        st.dataframe(display, use_container_width=True, hide_index=True)

        if len(filtered):
            chosen = st.selectbox(
                "Open observation",
                filtered["Replay Step"].astype(int).tolist(),
            )
            detail = filtered[filtered["Replay Step"] == chosen].iloc[0]

            st.markdown("#### Observation details")
            d1, d2, d3 = st.columns(3)
            with d1:
                metric_card("REPLAY STEP", f"{int(chosen):,}", str(detail["Target Timestamp"]))
            with d2:
                metric_card("ANOMALY SCORE", f"{float(detail['Anomaly Score']):.5f}", "Mean absolute residual")
            with d3:
                metric_card("THRESHOLD", f"{float(detail['Threshold']):.5f}", "Frozen validation threshold")

            st.write("**Reason:**", detail["Reason"])
            st.write("**Interpretation:**", detail["Interpretation"])

            if st.button("Open in Historical Replay", type="primary"):
                st.session_state.replay_step = int(chosen)
                st.session_state["navigation_target"] = "Historical Replay"
                st.rerun()

    with st.expander("How the triage rule works"):
        st.write(
            "The triage engine compares the mean absolute prediction residual "
            "across 209 features with a threshold calibrated from benign "
            "validation observations. The held-out test results showed that "
            "this residual-only threshold did not detect the Bot observations. "
            "It should therefore be treated as an experimental review signal, "
            "not a validated attack detector."
        )

    footer()


# ============================================================
# FEATURE EXPLORER PAGE
# ============================================================

elif page == "Feature Explorer":

    page_header(
        "MODEL INTERPRETABILITY",
        "Feature Explorer",
        "Inspect prediction errors and understand which network-state features contribute to them.",
    )

    feature = st.selectbox(
        "Choose a model feature",
        list(feature_names),
        index=list(feature_names).index("Flow Count")
        if "Flow Count" in feature_names else 0,
    )

    feature_idx = int(np.where(feature_names == feature)[0][0])
    residuals = np.abs(actual[:, feature_idx] - predictions[:, feature_idx])
    signed = actual[:, feature_idx] - predictions[:, feature_idx]

    a, b, c = st.columns(3)
    with a:
        metric_card("FEATURE MEAN ABSOLUTE ERROR", f"{residuals.mean():.5f}", feature)
    with b:
        metric_card("FEATURE MEDIAN ABSOLUTE ERROR", f"{np.median(residuals):.5f}", "Across held-out replay")
    with c:
        metric_card("MAXIMUM ABSOLUTE ERROR", f"{residuals.max():.5f}", "Across held-out replay")

    st.write("")
    section_title(
        "Actual vs predicted",
        "Compare actual and predicted values for the selected feature over a short replay window.",
    )

    start = st.slider(
        "Start replay step",
        min_value=1,
        max_value=max(1, len(df) - 100),
        value=1,
        key="feature_start",
    )
    end = min(len(df), start + 100)
    positions = np.arange(start - 1, end)

    comparison = pd.DataFrame({
        "Actual": actual[positions, feature_idx],
        "Predicted": predictions[positions, feature_idx],
    }, index=df.iloc[positions]["Replay Step"])

    st.line_chart(comparison, height=320)

    section_title(
        "Feature error distribution",
        "Absolute residuals for the selected feature across the test replay.",
    )
    st.line_chart(
        pd.DataFrame(
            {"Absolute error": residuals[positions]},
            index=df.iloc[positions]["Replay Step"],
        ),
        height=220,
    )

    st.write("")
    section_title(
        "Most frequent error contributors",
        "Features most often appearing among the top five per-observation residual contributors.",
    )

    contributor_path = EXPLAIN_DIR / "most_frequent_error_contributors.csv"
    if contributor_path.exists():
        contributors = pd.read_csv(contributor_path)
        contributors = contributors.head(15).set_index("Feature")
        st.bar_chart(
            contributors[["Appearance Percent"]],
            horizontal=True,
            height=420,
        )
        st.dataframe(
            contributors.reset_index(),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Feature error contribution indicates model prediction difficulty. "
        "It does not establish that a feature is an attack indicator."
    )

    footer()


# ============================================================
# MODEL & DATASET PAGE
# ============================================================

elif page == "Model & Dataset":

    page_header(
        "SYSTEM INFORMATION",
        "Model & Dataset",
        "Inspect the trained world model, its evaluation and the data used by the replay.",
    )

    section_title("Model configuration", "The trained architecture used by the historical replay.")

    config1, config2, config3, config4 = st.columns(4)
    with config1:
        metric_card("ARCHITECTURE", "LSTM", "World Model V3")
    with config2:
        metric_card("HIDDEN SIZE", "128", "One recurrent layer")
    with config3:
        metric_card("INPUT WINDOW", "10", "Historical states")
    with config4:
        metric_card("OUTPUT SIZE", "209", "Next-state features")

    st.write("")
    left, right = st.columns(2)

    with left:
        section_title("Training configuration")
        st.dataframe(
            pd.DataFrame({
                "Parameter": [
                    "Checkpoint epoch",
                    "Sequence length",
                    "Input features",
                    "Hidden size",
                    "LSTM layers",
                    "Dropout",
                    "Optimizer",
                    "Learning rate",
                    "Loss",
                    "Device used for training",
                ],
                "Value": [
                    manifest.get("checkpoint_epoch", 11),
                    manifest.get("sequence_length", 10),
                    manifest.get("input_features", 209),
                    128,
                    1,
                    0.2,
                    "AdamW",
                    0.001,
                    "Huber Loss",
                    "CUDA · NVIDIA GTX 1050",
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        section_title("Replay configuration")
        st.dataframe(
            pd.DataFrame({
                "Property": [
                    "Dataset",
                    "Replay observations",
                    "Prediction horizon",
                    "Prediction method",
                    "Prediction feedback",
                    "Comparison space",
                ],
                "Value": [
                    "CIC-IDS-2018",
                    f"{manifest['number_of_steps']:,}",
                    manifest["prediction_horizon"],
                    manifest["prediction_method"],
                    "Disabled",
                    manifest["prediction_comparison_space"],
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.write("")
    section_title("Held-out test evaluation", "LSTM performance compared with the persistence baseline.")

    eval_data = pd.DataFrame({
        "Metric": ["Huber Loss", "MAE", "RMSE"],
        "LSTM": [0.12068179, 0.32062694, 0.53132802],
        "Persistence Baseline": [0.17707066, 0.35622746, 0.67903316],
    }).set_index("Metric")

    st.dataframe(eval_data, use_container_width=True)

    st.caption(
        "The evaluation metrics are measured in preprocessed model feature space. "
        "They are not errors in original network units."
    )

    st.write("")
    section_title("Dataset partitions", "Chronological partitioning used for model development.")

    split_data = pd.DataFrame({
        "Partition": ["Training", "Validation", "Held-out test"],
        "State rows": [199120, 51303, 31426],
        "Sequence samples": [197389, 50870, 31406],
        "Purpose": [
            "Model fitting",
            "Model selection and threshold calibration",
            "Final held-out evaluation",
        ],
    })

    st.dataframe(split_data, use_container_width=True, hide_index=True)

    with st.expander("Research limitations and interpretation"):
        st.markdown("""
        - The LSTM predicts the next network state; it is not a direct attack classifier.
        - The test replay uses actual historical input windows, not prediction feedback.
        - Residual-only triage did not detect Bot observations in the held-out test set.
        - Dataset labels are reference labels, not LSTM predictions.
        - Feature values and prediction errors are in transformed model space.
        - The validation and test periods contain different traffic distributions and attack labels.
        """)

    footer()
