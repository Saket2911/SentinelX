from pathlib import Path
import json
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import streamlit as st

from sentinelx_shap_dashboard import (
    aggregate_local_contributions,
    load_shap_artifacts,
    sample_position_for_replay_step,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Models" / "lstm_world_model_v3_gpu_run2"
REPLAY_DIR = MODEL_DIR / "historical_replay_v1"
EXPLAIN_DIR = MODEL_DIR / "explainable_triage_v2"
TRIAGE_DIR = MODEL_DIR / "triage_engine_v1"
SHAP_DIR = MODEL_DIR / "shap_explainability_v3"
DATA_DIR = ROOT / "Datasets"
REPORT_DIR = ROOT / "reports" / "mitre_attack_contextual_mapping_v1"

MITRE_CSV = REPORT_DIR / "mitre_attack_contextual_mapping_v1.csv"
REPLAY_CSV = REPLAY_DIR / "test_replay.csv"
REPLAY_NPZ = REPLAY_DIR / "replay_arrays.npz"
EXPLAIN_CSV = EXPLAIN_DIR / "explainable_triage.csv"
TRIAGE_CSV = TRIAGE_DIR / "test_triage.csv"
CONTRIBUTOR_CSV = EXPLAIN_DIR / "most_frequent_error_contributors.csv"
TEST_X = DATA_DIR / "sentinelx_test_X_seq10_v3.npy"
TEST_METADATA = DATA_DIR / "sentinelx_test_metadata_seq10_v3.csv"

st.set_page_config(
    page_title="SentinelX | Security Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --sx-bg: #f4f7fb;
    --sx-panel: #ffffff;
    --sx-ink: #17283f;
    --sx-muted: #5f6f82;
    --sx-line: #e5ebf2;
    --sx-teal: #11776b;
    --sx-navy: #101c2e;
    --sx-card-foot: #65758a;
    --sx-status-bg: #e7f6f1;
    --sx-status-ink: #147a68;
    --sx-warning-bg: #fff7e8;
    --sx-warning-ink: #8b641f;
    --sx-review-bg: #fff0eb;
    --sx-review-ink: #9f2d20;
    --sx-within-bg: #e7f6f1;
    --sx-within-ink: #147a68;
    --sx-input-bg: #ffffff;
}

[data-theme="dark"] {
    color-scheme: dark;
    --sx-bg: #0e1117;
    --sx-panel: #161b22;
    --sx-ink: #e6edf3;
    --sx-muted: #a1acb8;
    --sx-line: #343d47;
    --sx-teal: #5dd7c1;
    --sx-navy: #0b1220;
    --sx-card-foot: #a9b5c3;
    --sx-status-bg: #14352f;
    --sx-status-ink: #8de2c7;
    --sx-warning-bg: #332816;
    --sx-warning-ink: #f3cd81;
    --sx-review-bg: #482b2d;
    --sx-review-ink: #ffb4a9;
    --sx-within-bg: #15362b;
    --sx-within-ink: #a2e8c1;
    --sx-input-bg: #101820;
}

@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        color-scheme: dark;
        --sx-bg: #0e1117;
        --sx-panel: #161b22;
        --sx-ink: #e6edf3;
        --sx-muted: #a1acb8;
        --sx-line: #343d47;
        --sx-teal: #5dd7c1;
        --sx-navy: #0b1220;
        --sx-card-foot: #a9b5c3;
        --sx-status-bg: #14352f;
        --sx-status-ink: #8de2c7;
        --sx-warning-bg: #332816;
        --sx-warning-ink: #f3cd81;
        --sx-review-bg: #482b2d;
        --sx-review-ink: #ffb4a9;
        --sx-within-bg: #15362b;
        --sx-within-ink: #a2e8c1;
        --sx-input-bg: #101820;
    }
}

[data-theme="light"] {
    color-scheme: light;
    --sx-bg: #f4f7fb;
    --sx-panel: #ffffff;
    --sx-ink: #17283f;
    --sx-muted: #5f6f82;
    --sx-line: #e5ebf2;
    --sx-teal: #11776b;
    --sx-navy: #101c2e;
    --sx-card-foot: #65758a;
    --sx-status-bg: #e7f6f1;
    --sx-status-ink: #147a68;
    --sx-warning-bg: #fff7e8;
    --sx-warning-ink: #8b641f;
    --sx-review-bg: #fff0eb;
    --sx-review-ink: #9f2d20;
    --sx-within-bg: #e7f6f1;
    --sx-within-ink: #147a68;
    --sx-input-bg: #ffffff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--sx-bg);
    color: var(--sx-ink);
}

header[data-testid="stHeader"] {
    background: var(--sx-bg);
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
    color: var(--sx-teal);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.sx-page-title {
    color: var(--sx-ink);
    font-family: 'Manrope', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin: 0;
}

.sx-subtitle {
    color: var(--sx-muted);
    font-size: 0.94rem;
    margin-top: 5px;
    margin-bottom: 22px;
}

.sx-card {
    background: var(--sx-panel);
    border: 1px solid var(--sx-line);
    border-radius: 14px;
    padding: 18px 20px;
    min-height: 112px;
    box-shadow: 0 2px 8px rgba(20, 40, 65, 0.025);
}

.sx-card-label {
    color: var(--sx-muted);
    font-size: 0.79rem;
    font-weight: 600;
    margin-bottom: 9px;
}

.sx-card-value {
    color: var(--sx-ink);
    font-family: 'Manrope', sans-serif;
    font-size: 1.8rem;
    line-height: 1.2;
    font-weight: 800;
}

.sx-card-foot {
    color: var(--sx-card-foot);
    font-size: 0.75rem;
    margin-top: 7px;
}

.sx-section {
    color: var(--sx-ink);
    font-family: 'Manrope', sans-serif;
    font-size: 1.08rem;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 3px;
}

.sx-section-sub {
    color: var(--sx-muted);
    font-size: 0.82rem;
    margin-bottom: 14px;
}

.sx-panel {
    background: var(--sx-panel);
    color: var(--sx-ink);
    border: 1px solid var(--sx-line);
    border-radius: 14px;
    padding: 18px 20px;
}

.sx-status {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 20px;
    background: var(--sx-status-bg);
    color: var(--sx-status-ink);
    font-size: 0.72rem;
    font-weight: 700;
}

.sx-warning {
    background: var(--sx-warning-bg);
    border: 1px solid var(--sx-line);
    color: var(--sx-warning-ink);
    border-radius: 10px;
    padding: 12px 15px;
    font-size: 0.85rem;
}

.sx-footer {
    border-top: 1px solid var(--sx-line);
    margin-top: 35px;
    padding-top: 15px;
    color: var(--sx-muted);
    font-size: 0.75rem;
}

div[data-testid="stMetric"] {
    background: var(--sx-panel);
    border: 1px solid var(--sx-line);
    border-radius: 13px;
    padding: 15px 17px;
    box-shadow: 0 2px 8px rgba(20, 40, 65, 0.025);
}

div[data-testid="stMetricLabel"] {
    color: var(--sx-muted);
    font-size: 0.8rem;
}

div[data-testid="stMetricValue"] {
    color: var(--sx-ink);
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

.stButton > button {
    border-radius: 9px;
    border: 1px solid var(--sx-line);
    font-weight: 600;
    min-height: 40px;
}

.stButton > button[kind="primary"] {
    background: #147f73;
    border-color: #147f73;
    color: #ffffff;
}

.stButton > button[kind="primary"]:hover {
    background: #106c62;
    border-color: #106c62;
}

.stSelectbox label, .stSlider label, .stTextInput label {
    color: var(--sx-ink);
    font-weight: 600;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--sx-line);
    border-radius: 10px;
}

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--sx-muted);
}

.st-key-threat-triage-card {
    background: var(--sx-panel);
    color: var(--sx-ink);
    border: 1px solid var(--sx-line);
    border-radius: 14px;
    padding: 1rem 1.25rem;
}

.st-key-threat-triage-card [data-testid="stMarkdownContainer"] * {
    color: var(--sx-ink);
}

.st-key-threat-triage-card label {
    color: var(--sx-ink);
}

.st-key-threat-triage-card textarea {
    background: var(--sx-input-bg);
    color: var(--sx-ink);
    border-color: var(--sx-line);
}

.sx-triage-badge {
    display: inline-block;
    padding: 4px 9px;
    border: 1px solid transparent;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
}

.sx-triage-badge-review {
    background: var(--sx-review-bg);
    color: var(--sx-review-ink) !important;
    border-color: var(--sx-review-ink);
}

.sx-triage-badge-within {
    background: var(--sx-within-bg);
    color: var(--sx-within-ink) !important;
    border-color: var(--sx-within-ink);
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading SentinelX replay data...")
def load_data():
    required = [
        REPLAY_CSV,
        REPLAY_NPZ,
        EXPLAIN_CSV,
        TRIAGE_CSV,
        TEST_X,
        MITRE_CSV,
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    replay = pd.read_csv(REPLAY_CSV)
    explanation = pd.read_csv(EXPLAIN_CSV)
    triage = pd.read_csv(TRIAGE_CSV)
    mitre = pd.read_csv(MITRE_CSV)

    mitre_required = [
        "label",
        "tactic",
        "technique",
        "name",
        "status",
        "evidence",
        "notes",
    ]
    missing_mitre = [column for column in mitre_required if column not in mitre.columns]
    if missing_mitre:
        raise ValueError(f"MITRE mapping is missing columns: {missing_mitre}")

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
        absent = [column for column in keys if column not in frame.columns]
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

    missing_triage = [column for column in triage_columns if column not in triage.columns]
    if missing_triage:
        raise ValueError(f"Triage data missing columns: {missing_triage}")

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
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing replay manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    return (
        merged,
        predictions,
        actual,
        feature_names,
        input_sequences,
        manifest,
        mitre,
    )


try:
    df, predictions, actual, feature_names, input_sequences, manifest, mitre = load_data()
except Exception as exc:
    st.error(f"SentinelX could not load its verified replay data: {exc}")
    st.stop()


@st.cache_data(show_spinner="Loading saved V3 SHAP artifacts...")
def load_cached_shap_artifacts(expected_feature_names):
    return load_shap_artifacts(
        SHAP_DIR,
        TEST_METADATA,
        REPLAY_CSV,
        expected_feature_names,
    )


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


def visualization_note(description):
    st.caption(description)


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


def stop_replay():
    st.session_state.replay_playing = False


def dataset_supported_category(label):
    normalized = str(label).strip().casefold()
    if "ddos" in normalized:
        return "DDoS (dataset-label supported)"
    if normalized.startswith("dos attack"):
        return "DoS (dataset-label supported)"
    if "brute force" in normalized or "bruteforce" in normalized:
        return "Brute force (dataset-label supported)"
    if "sql injection" in normalized:
        return "SQL injection (dataset-label supported)"
    return None


def render_shap_explainability_page(artifacts):
    page_header(
        "MODEL INTERPRETABILITY",
        "SHAP Explainability",
        "Inspect saved V3 attributions for selected historical next-state predictions.",
    )
    st.warning(
        "Local SHAP values are available for only 64 selected observations out of "
        f"{len(df):,} replay steps. Other steps do not have saved local explanations."
    )
    st.caption(
        "SHAP explains contributions to a selected next-state feature prediction "
        "in transformed feature space. It does not predict attack probability or establish causality."
    )

    observations = artifacts["observations"]
    output_names = artifacts["output_names"]
    output_indices = artifacts["output_indices"]
    feature_names = artifacts["feature_names"]

    observation_col, output_col = st.columns(2)
    with observation_col:
        sample_position = st.selectbox(
            "Saved replay observation",
            options=list(range(len(observations))),
            format_func=lambda position: (
                f"Step {int(observations.iloc[position]['Replay Step']):,} · "
                f"{observations.iloc[position]['Target Timestamp']} · "
                f"{observations.iloc[position]['Target Label']}"
            ),
            key="shap_observation",
        )
    with output_col:
        output_position = st.selectbox(
            "Predicted output feature",
            options=list(range(len(output_names))),
            format_func=lambda position: (
                f"{output_names[position]} (model output {int(output_indices[position])})"
            ),
            key="shap_output",
        )

    selected_sample_position = int(sample_position)
    observation = observations.iloc[selected_sample_position]
    replay_step = int(observation["Replay Step"])
    mapped_sample_position = sample_position_for_replay_step(
        artifacts["explain_indices"], replay_step
    )
    if mapped_sample_position is None:
        st.info(
            f"Replay step {replay_step:,} has no saved local SHAP explanation. "
            "Choose one of the 64 listed observations to inspect an explanation."
        )
        return
    if mapped_sample_position != selected_sample_position:
        st.error("The selected replay step does not match its saved SHAP sample position.")
        return
    sample_position = mapped_sample_position

    step_check = st.number_input(
        "Check another replay step",
        min_value=1,
        max_value=len(df),
        value=min(max(get_step(), 1), len(df)),
        step=1,
        key="shap_replay_step_check",
        help="This check does not calculate SHAP for steps without a saved explanation.",
    )
    checked_position = sample_position_for_replay_step(
        artifacts["explain_indices"], int(step_check)
    )
    if checked_position is None:
        st.info(
            f"Replay step {int(step_check):,} has no saved local SHAP explanation. "
            "Only the 64 observations in the selector above have saved local values."
        )
    else:
        st.success(
            f"Replay step {int(step_check):,} has a saved local explanation "
            f"(sample position {checked_position})."
        )

    summary_col, timestamp_col, label_col = st.columns(3)
    with summary_col:
        metric_card("REPLAY STEP", f"{replay_step:,}", "Saved historical observation")
    with timestamp_col:
        metric_card(
            "TARGET TIMESTAMP",
            str(observation["Target Timestamp"]),
            str(observation["Target CaptureDay"]),
        )
    with label_col:
        metric_card(
            "DATASET REFERENCE LABEL",
            str(observation["Target Label"]),
            "Retrospective reference, not a model prediction",
        )
    visualization_note(
        "Replay step and timestamp identify the matched observation; the label is the "
        "retrospective dataset reference, not an LSTM output."
    )

    output_name = str(output_names[int(output_position)])
    local_values = artifacts["shap_values"][
        sample_position, :, :, int(output_position)
    ]
    contributions = aggregate_local_contributions(local_values, feature_names)
    contributions["Direction"] = np.where(
        contributions["Signed Contribution"] > 0,
        "Positive",
        np.where(contributions["Signed Contribution"] < 0, "Negative", "Zero"),
    )

    local_tab, group_tab = st.tabs(
        ["Per-observation explanation", "Group-level analysis"]
    )
    with local_tab:
        section_title(
            f"Local attribution · {output_name}",
            "Feature importance ranks mean absolute attribution across the 10 historical positions; "
            "signed contributions are summed across those positions.",
        )
        top_count = st.slider(
            "Features to display",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            key="shap_top_count",
        )
        top_features = contributions.head(top_count)
        chart = pd.DataFrame({
            "Positive contribution": top_features["Signed Contribution"].clip(lower=0).to_numpy(),
            "Negative contribution": top_features["Signed Contribution"].clip(upper=0).to_numpy(),
        }, index=top_features["Input Feature"])
        st.bar_chart(chart, horizontal=True, height=440)
        visualization_note(
            "Each bar is a feature's signed SHAP contribution summed over 10 input positions. "
            "Bars to the right of zero are positive and those to the left are negative; "
            "features are ranked by mean absolute attribution, not attack likelihood."
        )

        display_columns = [
            "Input Feature",
            "Direction",
            "Signed Contribution",
            "Mean Absolute Attribution",
        ]
        contribution_table = top_features[display_columns].style.format({
            "Signed Contribution": "{:+.6g}",
            "Mean Absolute Attribution": "{:.6g}",
        })
        st.dataframe(contribution_table, width="stretch", hide_index=True)
        visualization_note(
            "Rows are input features. Direction and signed contribution show the net "
            "effect across the 10 positions; mean absolute attribution shows average "
            "contribution magnitude regardless of sign."
        )

        with st.expander("Inspect SHAP contributions at all 10 historical positions"):
            selected_features = st.multiselect(
                "Input features",
                options=contributions["Input Feature"].tolist(),
                default=top_features["Input Feature"].head(5).tolist(),
                key="shap_time_features",
            )
            if selected_features:
                positions = [
                    int(np.flatnonzero(feature_names == feature)[0])
                    for feature in selected_features
                ]
                position_labels = [
                    f"t-{artifacts['shap_values'].shape[1] - position}"
                    for position in range(artifacts["shap_values"].shape[1])
                ]
                position_table = pd.DataFrame(
                    local_values[:, positions],
                    index=position_labels,
                    columns=selected_features,
                )
                st.dataframe(
                    position_table.style.format("{:+.6g}"),
                    width="stretch",
                )
                visualization_note(
                    "Rows are historical positions (t-10 oldest through t-1 newest); "
                    "columns are selected features and cells are signed SHAP contributions."
                )
                st.caption(
                    "Position 0 is the oldest input state (t-10); position 9 is the most recent (t-1)."
                )
            else:
                st.info("Select at least one feature to inspect its time-position contributions.")

        section_title(
            "Temporal SHAP heatmap",
            "Choose the top-ranked inputs to keep labels readable, or select all 209 features.",
        )
        heatmap_size = st.selectbox(
            "Features in heatmap",
            options=[10, 20, 50, len(feature_names)],
            index=1,
            format_func=lambda count: (
                f"All {count} features" if count == len(feature_names)
                else f"Top {count} features"
            ),
            key="shap_heatmap_size",
        )
        heatmap_features = contributions.head(int(heatmap_size))["Input Feature"].tolist()
        heatmap_indices = [
            int(np.flatnonzero(feature_names == feature)[0])
            for feature in heatmap_features
        ]
        heatmap_labels = [
            f"t-{artifacts['shap_values'].shape[1] - position}"
            for position in range(artifacts["shap_values"].shape[1])
        ]
        heatmap = pd.DataFrame(
            local_values[:, heatmap_indices],
            index=heatmap_labels,
            columns=heatmap_features,
        )
        color_limit = max(float(np.abs(heatmap.to_numpy()).max()), 1e-12)

        def heatmap_color(value):
            if value == 0:
                return "background-color: rgba(148, 163, 184, 0.12)"
            alpha = 0.12 + 0.78 * min(abs(float(value)) / color_limit, 1.0)
            color = "196, 74, 74" if value > 0 else "55, 116, 190"
            return f"background-color: rgba({color}, {alpha:.3f})"

        st.dataframe(
            heatmap.style.map(heatmap_color).format("{:+.4g}"),
            width="stretch",
        )
        visualization_note(
            "Rows are the 10 historical positions and columns are selected input features; "
            "cell values are SHAP contributions. Red indicates positive and blue negative "
            "contributions; use the feature selector to avoid overcrowding."
        )

    with group_tab:
        st.caption(
            "These saved CSV summaries aggregate across Benign and Bot groups. "
            "They are not explanations for the selected individual observation."
        )
        section_title(
            "Feature importance by dataset label",
            "The saved group summaries compare mean absolute SHAP magnitude for Benign and Bot reference-label groups.",
        )
        importance = artifacts["feature_importance"]
        benign = (
            importance[
                (importance["Target Label"] == "Benign")
                & (importance["Predicted Output Feature"] == output_name)
            ]
            .sort_values("Mean Absolute SHAP", ascending=False)
            .head(15)
        )
        bot = (
            importance[
                (importance["Target Label"] == "Bot")
                & (importance["Predicted Output Feature"] == output_name)
            ]
            .sort_values("Mean Absolute SHAP", ascending=False)
            .head(15)
        )
        benign_col, bot_col = st.columns(2)
        with benign_col:
            st.markdown("**Benign · top features**")
            st.bar_chart(
                benign.set_index("Input Feature")[["Mean Absolute SHAP"]],
                horizontal=True,
                height=420,
            )
            visualization_note(
                "Bars rank input features by mean absolute SHAP for the selected output "
                "among Benign reference-label observations; larger bars mean stronger "
                "average contribution magnitude."
            )
            st.dataframe(benign, width="stretch", hide_index=True)
            visualization_note(
                "Each row is an input feature; columns show its mean absolute SHAP, "
                "mean signed SHAP, and rank for this output and label group."
            )
        with bot_col:
            st.markdown("**Bot · top features**")
            st.bar_chart(
                bot.set_index("Input Feature")[["Mean Absolute SHAP"]],
                horizontal=True,
                height=420,
            )
            visualization_note(
                "Bars rank input features by mean absolute SHAP for the selected output "
                "among Bot reference-label observations; larger bars mean stronger "
                "average contribution magnitude."
            )
            st.dataframe(bot, width="stretch", hide_index=True)
            visualization_note(
                "Each row is an input feature; columns show its mean absolute SHAP, "
                "mean signed SHAP, and rank for this output and label group."
            )

        section_title("Benign-versus-Bot attribution differences")
        comparison = (
            artifacts["comparison"][
                artifacts["comparison"]["Predicted Output Feature"] == output_name
            ]
            .sort_values("Absolute_Difference", ascending=False)
            .head(15)
        )
        st.dataframe(comparison, width="stretch", hide_index=True)
        visualization_note(
            "Rows are input features for the selected output. Benign and Bot are the "
            "groups' mean signed SHAP values; Bot_minus_Benign_SHAP is their signed "
            "difference and Absolute_Difference is its magnitude."
        )

        section_title("Temporal importance by label")
        temporal = artifacts["temporal_importance"]
        temporal = temporal[
            temporal["Predicted Output Feature"] == output_name
        ]
        temporal_chart = temporal.pivot(
            index="Input Time Position",
            columns="Target Label",
            values="Mean Absolute SHAP",
        ).reindex(range(artifacts["shap_values"].shape[1]))
        temporal_chart.index = [
            f"t-{artifacts['shap_values'].shape[1] - position}"
            for position in temporal_chart.index
        ]
        st.line_chart(temporal_chart, height=280)
        visualization_note(
            "The x-axis is historical position (t-10 oldest to t-1 newest); the y-axis "
            "is mean absolute SHAP. Separate lines compare Benign and Bot reference-label groups."
        )

        section_title("V2-versus-V3 ranking comparison")
        stability = artifacts["stability"]
        stability = stability[stability["Output"] == output_name]
        st.dataframe(stability, width="stretch", hide_index=True)
        visualization_note(
            "Each row compares group-level V2 and V3 feature rankings for a label and "
            "output. Top10 Overlap, Jaccard, and Spearman summarize ranking agreement; "
            "they do not prove individual-attribution stability."
        )
        st.caption(
            "Ranking overlap and correlation compare group-level feature rankings. "
            "They do not prove stability of individual-observation attributions."
        )


def render_predictive_threat_monitoring():
    section_title(
        "Predictive Threat Monitoring",
        "Historical V3 forecasts, residual-based review status, and saved explanation evidence.",
    )
    if df.empty:
        st.info("No historical replay observations are available to monitor.")
        return
    st.caption(
        "LSTM next-state prediction, residual review, SHAP contribution, and dataset "
        "reference label are separate signals. The label is retrospective."
    )

    try:
        shap_artifacts = load_cached_shap_artifacts(
            tuple(str(name) for name in feature_names)
        )
    except (FileNotFoundError, ValueError, OSError, KeyError, BadZipFile) as exc:
        shap_artifacts = None
        st.warning(
            "Saved local SHAP is unavailable because its artifacts could not be "
            f"validated: {exc}"
        )

    if shap_artifacts is None:
        output_names = feature_names.astype(str)
        output_indices = np.arange(len(feature_names), dtype=np.int64)
    else:
        output_names = shap_artifacts["output_names"]
        output_indices = shap_artifacts["output_indices"]

    review_rows = df[df["Triage Status"] == "REVIEW_REQUIRED"]
    if len(review_rows):
        default_step = int(review_rows.iloc[0]["Replay Step"])
    else:
        default_step = int(df.loc[df["Prediction MAE (model space)"].idxmax(), "Replay Step"])

    control_col, output_col = st.columns([1, 1])
    with control_col:
        selected_step = int(st.number_input(
            "Historical replay step",
            min_value=1,
            max_value=len(df),
            value=default_step,
            step=1,
            key="overview_monitoring_step",
            help="Select an actual observation from the saved held-out replay.",
        ))
    with output_col:
        default_output = (
            int(np.flatnonzero(output_names == "Flow Count")[0])
            if "Flow Count" in output_names else 0
        )
        output_position = int(st.selectbox(
            "Predicted next-state feature",
            options=list(range(len(output_names))),
            index=default_output,
            format_func=lambda position: (
                f"{output_names[position]} "
                f"(model output {int(output_indices[position])})"
            ),
            key="overview_monitoring_output",
        ))

    row_index = selected_step - 1
    observation = df.iloc[row_index]
    if int(observation["Replay Step"]) != selected_step:
        st.error("Selected replay step does not match the saved replay row.")
        return

    output_name = str(output_names[output_position])
    output_index = int(output_indices[output_position])
    timestamp_col, status_col, label_col = st.columns(3)
    with timestamp_col:
        metric_card("REPLAY STEP", f"{selected_step:,}", str(observation["Target Timestamp"]))
    with status_col:
        metric_card(
            "RESIDUAL REVIEW STATUS",
            str(observation["Triage Status"]).replace("_", " "),
            f"Anomaly score {float(observation['Anomaly Score']):.5f} · "
            f"threshold {float(observation['Threshold']):.5f}",
        )
    with label_col:
        metric_card(
            "DATASET REFERENCE LABEL",
            str(observation["Target Label"]),
            "Retrospective label; not an LSTM prediction",
        )
    visualization_note(
        "Replay step and timestamp identify the selected historical row. Residual status "
        "is an unusual-behaviour review signal; the dataset label is retrospective."
    )

    actual_value = float(actual[row_index, output_index])
    predicted_value = float(predictions[row_index, output_index])
    signed_residual = actual_value - predicted_value
    actual_col, prediction_col, residual_col = st.columns(3)
    with actual_col:
        metric_card("OBSERVED NEXT STATE", f"{actual_value:.5f}", output_name)
    with prediction_col:
        metric_card("V3 LSTM PREDICTION", f"{predicted_value:.5f}", output_name)
    with residual_col:
        metric_card(
            "ABSOLUTE RESIDUAL",
            f"{abs(signed_residual):.5f}",
            "Observed minus predicted, model space",
        )
    visualization_note(
        "Observed and predicted values are for the selected next-state feature in "
        "transformed model space; absolute residual is their absolute difference."
    )

    window_start = max(0, row_index - 15)
    window_end = min(len(df), row_index + 16)
    window_positions = np.arange(window_start, window_end)
    prediction_window = pd.DataFrame(
        {
            "Observed next state": actual[window_positions, output_index],
            "V3 predicted next state": predictions[window_positions, output_index],
        },
        index=df.iloc[window_positions]["Replay Step"],
    )
    prediction_window.index.name = "Replay Step"
    section_title(
        "Actual vs predicted next-state feature",
        "Compare the selected LSTM output with its observed historical value around the chosen replay step.",
    )
    st.line_chart(prediction_window, height=260)
    visualization_note(
        "The x-axis is replay step and the y-axis is the selected feature's transformed "
        "value. The two lines are observed and predicted values; their separation is the residual."
    )

    evidence_rows = []
    for rank in range(1, 6):
        feature = observation.get(f"Top {rank} Feature")
        if pd.isna(feature):
            continue
        evidence_rows.append({
            "Rank": rank,
            "Residual feature": str(feature),
            "Signed residual (actual - predicted)": float(
                observation[f"Top {rank} Signed Residual"]
            ),
            "Absolute residual": float(observation[f"Top {rank} Absolute Error"]),
        })

    evidence_col, shap_col = st.columns([1.2, 1])
    with evidence_col:
        st.markdown("**Largest residual contributors**")
        if evidence_rows:
            st.dataframe(
                pd.DataFrame(evidence_rows),
                width="stretch",
                hide_index=True,
            )
            visualization_note(
                "Rows rank the largest per-feature residuals; signed residual is actual "
                "minus predicted, while absolute residual is the error magnitude."
            )
        else:
            st.info("Per-observation residual feature rankings are unavailable.")
        if observation["Triage Status"] == "REVIEW_REQUIRED":
            st.warning(
                "Residual threshold exceeded. This is an unusual-behaviour review "
                "indication, not an attack classification."
            )
        else:
            st.info(
                "Residual is within the calibrated review range; this does not "
                "establish that the activity is benign."
            )

    with shap_col:
        st.markdown("**Saved SHAP feature contributions**")
        if shap_artifacts is None:
            st.info("A local SHAP explanation could not be loaded.")
        else:
            sample_position = sample_position_for_replay_step(
                shap_artifacts["explain_indices"], selected_step
            )
            shap_output_position = np.flatnonzero(
                shap_artifacts["output_names"] == output_name
            )
            if sample_position is None:
                st.info(
                    f"No local SHAP explanation is saved for replay step "
                    f"{selected_step:,}. Local SHAP is available for only 64 selected observations."
                )
            elif not len(shap_output_position):
                st.info(
                    f"No local SHAP values are saved for output feature {output_name}."
                )
            else:
                local_values = shap_artifacts["shap_values"][
                    sample_position,
                    :,
                    :,
                    int(shap_output_position[0]),
                ]
                local_contributions = aggregate_local_contributions(
                    local_values,
                    shap_artifacts["feature_names"],
                ).head(5)
                st.dataframe(
                    local_contributions,
                    width="stretch",
                    hide_index=True,
                )
                visualization_note(
                    "Rows are the top input features; signed contribution sums SHAP over "
                    "10 positions and mean absolute attribution measures average magnitude. "
                    "SHAP is not an attack probability or causal finding."
                )

    if st.button(
        "Open selected observation in Threat Triage",
        type="primary",
        key="overview_open_monitoring_evidence",
    ):
        st.session_state["triage_observation_step"] = selected_step
        st.session_state["replay_step"] = selected_step
        st.session_state["navigation_target"] = "Threat Triage"
        st.rerun()

    section_title(
        "Replay timeline · residual-based review status",
        "Each point is a historical replay observation; color shows the saved residual review status.",
    )
    timeline_stride = max(1, len(df) // 1000)
    timeline_positions = set(range(0, len(df), timeline_stride))
    timeline_positions.update(
        df.index[df["Triage Status"] == "REVIEW_REQUIRED"].tolist()
    )
    timeline_positions.add(row_index)
    timeline = df.iloc[sorted(timeline_positions)][[
        "Replay Step",
        "Prediction MAE (model space)",
        "Triage Status",
    ]]
    st.scatter_chart(
        timeline,
        x="Replay Step",
        y="Prediction MAE (model space)",
        color="Triage Status",
        height=280,
    )
    visualization_note(
        "The x-axis is replay step and the y-axis is mean absolute prediction error; "
        "the legend's colors separate saved residual review statuses. This is a retrospective "
        "timeline, not a forecast."
    )
    st.caption(
        f"Timeline is sampled every {timeline_stride} replay steps for readability; "
        "all REVIEW_REQUIRED observations and the selected step are retained."
    )
    if len(review_rows):
        review_display = review_rows[[
            "Replay Step",
            "Target Timestamp",
            "Target Label",
            "Anomaly Score",
            "Threshold",
            "Triage Status",
        ]]
        st.dataframe(review_display, width="stretch", hide_index=True)
        visualization_note(
            "Each row is a threshold-crossing historical observation. Timestamp and replay "
            "step locate it; dataset label is retrospective, and score/threshold are residual based."
        )
    else:
        st.info("No observations crossed the saved residual review threshold.")

    st.caption(
        "This is retrospective replay, not evidence of early attack prediction. "
        "Residual review status is evaluated at the corresponding target observation."
    )


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
        "SHAP Explainability",
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
        '<span class="sx-status">● REPLAY ARTIFACTS VERIFIED</span>',
        unsafe_allow_html=True,
    )
    st.caption("LSTM World Model · V3")
    st.caption(f"{len(df):,} aligned replay observations")
    st.caption("CIC-IDS-2018 · Historical test replay")

    st.markdown("---")
    st.caption("SENTINELX RESEARCH BUILD")
    st.caption("Version 1.0 · Historical Replay")


if page == "Overview":

    page_header(
        "COMMAND CENTER",
        "Security Overview",
        "A high-level view of SentinelX predictive performance and historical replay activity.",
    )

    total = len(df)
    mean_mae = float(df["Prediction MAE (model space)"].mean())
    review_count = int((df["Triage Status"] == "REVIEW_REQUIRED").sum())
    review_rate = review_count / total * 100 if total else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("REPLAY OBSERVATIONS", f"{total:,}", "Held-out test observations")
    with m2:
        metric_card("MEAN PREDICTION MAE", f"{mean_mae:.4f}", "Preprocessed feature space")
    with m3:
        metric_card("REVIEW REQUIRED", f"{review_count:,}", f"{review_rate:.3f}% of replay")
    with m4:
        metric_card("MODEL FEATURES", f"{len(feature_names)}", "Network state features")
    visualization_note(
        "Replay observations counts held-out rows, mean prediction MAE summarizes average "
        "absolute next-state error, review count is residual-threshold based, and model "
        "features counts predicted state variables."
    )

    st.write("")
    left, right = st.columns([1.65, 1])

    with left:
        section_title(
            "Prediction error over replay",
            "Per-observation one-step-ahead LSTM mean absolute error across the historical test sequence.",
        )
        trend = df[["Replay Step", "Prediction MAE (model space)"]].copy()
        trend = trend.set_index("Replay Step")
        trend = trend.iloc[::max(1, len(trend) // 1200)]
        st.area_chart(trend, height=300)
        visualization_note(
            "The x-axis is replay step and the y-axis is per-observation mean absolute "
            "error across features. Higher values mean a less accurate next-state forecast; "
            "the displayed series is sampled for readability."
        )

    with right:
        section_title(
            "Triage distribution",
            "Residual threshold outcomes. These are review states, not attack classifications.",
        )
        status_counts = df["Triage Status"].value_counts()
        st.bar_chart(status_counts, horizontal=True, height=220)
        visualization_note(
            "Bars count observations in each residual-review status; the x-axis is count "
            "and the y-axis is status. These statuses are not attack classifications."
        )

        label_status = pd.crosstab(df["Target Label"], df["Triage Status"])
        st.dataframe(label_status,width="stretch")
        visualization_note(
            "Rows are retrospective dataset reference labels, columns are residual-review "
            "statuses, and cells count replay observations in each label/status combination."
        )

        st.info(
            f"{review_count:,} observations crossed the residual review threshold. "
            "The threshold is an experimental review signal, not a validated attack detector."
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
    st.dataframe(recent,width="stretch", hide_index=True)
    visualization_note(
        "Each row is one of the latest replay observations; columns give replay step, "
        "timestamp, retrospective reference label, prediction MAE, and residual-review status."
    )

    render_predictive_threat_monitoring()

    footer()


elif page == "Historical Replay":

    @st.fragment(run_every=1)
    def historical_replay():

        page_header(
            "TEMPORAL WORLD MODEL",
            "Historical Replay",
            "Step through held-out network activity and inspect the model's next-state prediction.",
        )

        total = len(df)

        if "replay_step" not in st.session_state:
            st.session_state.replay_step = 1
        if "replay_playing" not in st.session_state:
            st.session_state.replay_playing = False
        if "replay_speed" not in st.session_state:
            st.session_state.replay_speed = 1

        if st.session_state.replay_playing:
            current = get_step()
            if current >= total:
                st.session_state.replay_playing = False
            else:
                st.session_state.replay_step = min(
                    total,
                    current + int(st.session_state.replay_speed),
                )
                if st.session_state.replay_step >= total:
                    st.session_state.replay_playing = False

        play1, play2, play3 = st.columns([1.2, 1, 1])

        with play1:
            label = "⏸ Pause" if st.session_state.replay_playing else "▶ Play"
            if st.button(
                label,
                key="replay_play_pause",
                type="primary",
               width="stretch",
            ):
                st.session_state.replay_playing = not st.session_state.replay_playing

        with play2:
            st.selectbox(
                "Playback speed",
                [1, 2, 5, 10],
                key="replay_speed",
                format_func=lambda value: f"{value}×",
            )

        with play3:
            if st.button("↺ Reset", key="replay_reset",width="stretch"):
                st.session_state.replay_step = 1
                st.session_state.replay_playing = False
                st.rerun()

        control1, control2, control3, control4 = st.columns([1, 1, 1, 1.5])

        with control1:
            if st.button("⏮ First",width="stretch"):
                st.session_state.replay_step = 1
                st.session_state.replay_playing = False
                st.rerun()

        with control2:
            if st.button("← Previous",width="stretch"):
                st.session_state.replay_step = max(1, get_step() - 1)
                st.session_state.replay_playing = False
                st.rerun()

        with control3:
            if st.button("Next →",width="stretch"):
                st.session_state.replay_step = min(total, get_step() + 1)
                st.session_state.replay_playing = False
                st.rerun()

        with control4:
            if st.button("Last ⏭",width="stretch"):
                st.session_state.replay_step = total
                st.session_state.replay_playing = False
                st.rerun()

        step = st.slider(
            "Replay timeline",
            min_value=1,
            max_value=total,
            key="replay_step",
            on_change=stop_replay,
            help="Choose an observation from the historical replay.",
        )

        st.progress(
            (step - 1) / total,
            text=f"Replay progress: {step:,} / {total:,}",
        )
        visualization_note(
            "The progress bar shows the selected historical replay step relative to "
            "the full held-out sequence; it does not represent real-time monitoring."
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
                <span style="float:right;color:var(--sx-teal);font-weight:700">
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
            metric_card("PREDICTION MAE", f"{mae:.5f}", "Across all model features")
        with b:
            metric_card(
                "TRIAGE STATUS",
                str(row["Triage Status"]).replace("_", " "),
                "Frozen validation threshold",
            )
        with c:
            metric_card("PREDICTION HORIZON", "1 state", "One-step-ahead prediction")
        visualization_note(
            "Prediction MAE is the average absolute error across features; triage status "
            "shows the saved residual-threshold decision; the horizon is one next state."
        )

        with st.expander("Explain this prediction's largest residual contributors"):
            explanation_rows = []

            for rank in range(1, 6):
                feature_column = f"Top {rank} Feature"
                residual_column = f"Top {rank} Signed Residual"
                error_column = f"Top {rank} Absolute Error"
                contribution_column = f"Top {rank} Contribution Percent"

                required_columns = [
                    feature_column,
                    residual_column,
                    error_column,
                    contribution_column,
                ]

                if all(column in df.columns for column in required_columns):
                    explanation_rows.append({
                        "Rank": rank,
                        "Feature": row[feature_column],
                        "Signed Residual": row[residual_column],
                        "Absolute Error": row[error_column],
                        "Contribution (%)": row[contribution_column],
                    })

            if explanation_rows:
                st.dataframe(
                    pd.DataFrame(explanation_rows),
                   width="stretch",
                    hide_index=True,
                )
                visualization_note(
                    "Rows rank features by residual contribution; signed residual gives "
                    "direction, absolute error gives magnitude, and contribution percent "
                    "is each feature's share of the total residual."
                )
            else:
                st.info("Per-observation explanation columns are unavailable.")

            st.caption(
                "These are the largest contributors to prediction residuals. "
                "They are not attack probabilities, causal explanations or proof of malicious activity."
            )

        st.write("")
        section_title(
            "Inspect a network feature",
            "Compare its ten observed input values with the predicted and actual next state.",
        )

        selected_feature = st.selectbox(
            "Network state feature",
            options=list(feature_names),
            index=list(feature_names).index("Flow Count")
            if "Flow Count" in feature_names else 0,
            key="replay_feature",
        )

        feature_idx = int(np.where(feature_names == selected_feature)[0][0])
        history = np.asarray(
            input_sequences[idx, :, feature_idx],
            dtype=np.float32,
        )
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
        visualization_note(
            "The cards show observed next-state value, LSTM prediction, and their "
            "absolute difference for the selected feature; values are transformed."
        )

        chart = pd.DataFrame(
            {
                "Observed history": np.append(history, np.nan),
                "Predicted next state": [np.nan] * 10 + [predicted_value],
                "Actual next state": [np.nan] * 10 + [actual_value],
            },
            index=[f"t-{i}" for i in range(10, 0, -1)] + ["t"],
        )

        st.line_chart(chart, height=330)

        visualization_note(
            "The x-axis is input time position (t-10 oldest to t-1 newest, then t); "
            "the y-axis is the selected feature's transformed value. History, predicted "
            "next state, and actual next state are separate series, not raw network units."
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
        visualization_note(
            "The x-axis is replay step within the same segment and the y-axis is mean "
            "absolute error across features. The horizontal threshold line marks the "
            "frozen validation review cutoff."
        )

        with st.expander("View all model feature predictions for this observation"):
            feature_table = pd.DataFrame({
                "Feature": feature_names,
                "Actual": actual[idx],
                "Predicted": predictions[idx],
                "Absolute Error": np.abs(actual[idx] - predictions[idx]),
            }).sort_values("Absolute Error", ascending=False)

            st.dataframe(
                feature_table,
               width="stretch",
                hide_index=True,
            )
            visualization_note(
                "Each row is a model feature for this observation; Actual and Predicted "
                "are next-state values, and Absolute Error ranks the largest differences first."
            )

        with st.expander("Dataset reference label"):
            st.write(f"**Reference label:** {row['Target Label']}")
            st.caption(
                "This is the original dataset label for retrospective analysis. "
                "It is not predicted by the LSTM and is not used by the triage rule."
            )

        footer()

    historical_replay()


elif page == "Threat Triage":

    page_header(
        "ANALYST WORKSPACE",
        "Threat Triage",
        "Inspect V3 next-state predictions, observed residual evidence, and analyst assessment.",
    )

    review_df = df[df["Triage Status"] == "REVIEW_REQUIRED"].copy()
    within_df = df[df["Triage Status"] != "REVIEW_REQUIRED"].copy()

    a, b, c = st.columns(3)
    with a:
        metric_card("REVIEW REQUIRED", f"{len(review_df):,}", "Residual threshold exceeded")
    with b:
        metric_card("WITHIN RANGE", f"{len(within_df):,}", "No residual review trigger")
    with c:
        metric_card(
            "FROZEN THRESHOLD",
            f"{float(df['Threshold'].iloc[0]):.5f}",
            "Calibrated on validation data",
        )

    visualization_note(
        "Review-required and within-range counts describe saved residual-threshold "
        "outcomes; the frozen threshold is the validation-calibrated cutoff, not an "
        "attack-probability boundary."
    )

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
    section_title(
        "Threat triage assessment",
        "An observation-level view built from the saved V3 replay, residual explanation, and threshold outputs.",
    )

    if not len(df):
        st.info("No replay observations are available for triage.")
    else:
        default_step = (
            int(review_df["Replay Step"].iloc[0])
            if len(review_df)
            else int(df.loc[df["Prediction MAE (model space)"].idxmax(), "Replay Step"])
        )
        if "triage_observation_step" not in st.session_state:
            st.session_state["triage_observation_step"] = default_step
        selected_step = int(st.number_input(
            "Replay step to assess",
            min_value=1,
            max_value=len(df),
            step=1,
            key="triage_observation_step",
            help="Select an observation from the saved held-out V3 replay.",
        ))

        selected_indices = df.index[df["Replay Step"] == selected_step]
        if len(selected_indices):
            observation_index = int(selected_indices[0])
            observation = df.loc[observation_index]
            reference_label = str(observation["Target Label"])
            observation_mae = float(observation["Prediction MAE (model space)"])
            observation_threshold = float(observation["Threshold"])

            residuals = actual[observation_index] - predictions[observation_index]
            absolute_residuals = np.abs(residuals)
            rank_order = np.argsort(absolute_residuals)[::-1]

            explanation_features = []
            for rank in range(1, 6):
                feature = observation.get(f"Top {rank} Feature")
                if pd.notna(feature) and str(feature) in feature_names:
                    explanation_features.append(str(feature))

            if explanation_features:
                ranked_features = explanation_features
            else:
                ranked_features = [
                    str(feature_names[index]) for index in rank_order[:5]
                ]

            evidence_rows = []
            for rank, feature in enumerate(ranked_features, start=1):
                feature_index = int(np.where(feature_names == feature)[0][0])
                actual_value = float(actual[observation_index, feature_index])
                predicted_value = float(predictions[observation_index, feature_index])
                evidence = {
                    "Rank": rank,
                    "Feature": feature,
                    "Actual next state": actual_value,
                    "V3 predicted next state": predicted_value,
                    "Signed residual (actual - predicted)": actual_value - predicted_value,
                    "Absolute residual": abs(actual_value - predicted_value),
                }

                contribution = observation.get(
                    f"Top {rank} Contribution Percent"
                )
                if pd.notna(contribution):
                    evidence["Residual contribution (%)"] = float(contribution)
                evidence_rows.append(evidence)

            top_feature = evidence_rows[0] if evidence_rows else None
            if observation["Triage Status"] == "REVIEW_REQUIRED":
                observed_behaviour = (
                    f"The mean absolute residual is {observation_mae:.5f}, above "
                    f"the frozen validation review threshold of "
                    f"{observation_threshold:.5f}."
                )
            elif top_feature:
                direction = (
                    "higher" if top_feature["Signed residual (actual - predicted)"] > 0
                    else "lower"
                )
                observed_behaviour = (
                    f"No overall residual-threshold trigger. The largest feature "
                    f"difference is {top_feature['Feature']}: the observed next-state "
                    f"value is {abs(top_feature['Signed residual (actual - predicted)']):.5f} "
                    f"model-space units {direction} than the V3 prediction."
                )
            else:
                observed_behaviour = "Feature-level residual evidence is unavailable."

            category = dataset_supported_category(reference_label)
            label_mapping = mitre[
                mitre["label"].astype(str).str.strip() == reference_label.strip()
            ]
            verified_mapping = label_mapping[
                label_mapping["status"].astype(str).str.casefold().isin(
                    ["verified", "confirmed"]
                )
            ]

            with st.container(border=True, key="threat-triage-card"):
                st.markdown("#### Threat Triage Card")
                status_col, label_col = st.columns(2)
                with status_col:
                    st.markdown("**Assessment status**")
                    badge_class = (
                        "sx-triage-badge-review"
                        if observation["Triage Status"] == "REVIEW_REQUIRED"
                        else "sx-triage-badge-within"
                    )
                    badge_label = (
                        "REVIEW REQUIRED"
                        if observation["Triage Status"] == "REVIEW_REQUIRED"
                        else "WITHIN RANGE"
                    )
                    st.markdown(
                        f'<span class="sx-triage-badge {badge_class}">'
                        f"{badge_label}</span>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"Residual MAE {observation_mae:.5f}; threshold "
                        f"{observation_threshold:.5f}"
                    )
                with label_col:
                    st.markdown("**Observed dataset reference label**")
                    st.write(reference_label)

                st.markdown("**Observed unusual behaviour**")
                st.write(observed_behaviour)

                st.markdown("**Suspected attack category**")
                if category:
                    st.write(
                        f"{category}. This is supported by the actual dataset label, "
                        "not inferred by the LSTM."
                    )
                elif len(verified_mapping):
                    mapping = verified_mapping.iloc[0]
                    mapped_category = " / ".join(
                        str(mapping[column]).strip()
                        for column in ("tactic", "technique", "name")
                        if column in mapping.index
                        and pd.notna(mapping[column])
                        and str(mapping[column]).strip()
                    )
                    st.write(
                        f"{mapped_category or reference_label} "
                        "(verified threat-specific mapping)."
                    )
                else:
                    st.write(
                        "Unavailable: this observation has no dataset-label-supported "
                        "attack category or verified threat-specific mapping."
                    )

                st.markdown("**Supporting feature evidence**")
                if evidence_rows:
                    st.dataframe(
                        pd.DataFrame(evidence_rows),
                        width="stretch",
                        hide_index=True,
                    )
                    visualization_note(
                        "Rows rank the largest feature residuals. Actual and predicted "
                        "are transformed next-state values; signed residual is actual "
                        "minus predicted, and absolute residual is its magnitude."
                    )
                    for evidence in evidence_rows:
                        direction = (
                            "higher" if evidence["Signed residual (actual - predicted)"] > 0
                            else "lower"
                        )
                        st.caption(
                            f"{evidence['Feature']}: observed value is "
                            f"{abs(evidence['Signed residual (actual - predicted)']):.5f} "
                            f"model-space units {direction} than predicted."
                        )
                else:
                    st.info("Per-feature prediction and residual evidence is unavailable.")

                st.markdown("**Analyst assessment**")
                st.text_area(
                    "Analyst assessment",
                    value="Potential anomalous network behaviour",
                    key="triage_analyst_assessment",
                    label_visibility="collapsed",
                    help="Editable analyst interpretation; it is not generated by the model.",
                )

            st.caption(
                "Predictions and residuals are in transformed model feature space. "
                "The dataset label is retrospective reference data. Neither the residual "
                "trigger nor the feature ranking proves malicious activity."
            )

            with st.expander("View actual vs predicted next state for all features"):
                full_state = pd.DataFrame({
                    "Feature": feature_names,
                    "Observed actual next state": actual[observation_index],
                    "V3 predicted next state": predictions[observation_index],
                    "Signed residual (actual - predicted)": residuals,
                    "Absolute residual": absolute_residuals,
                }).sort_values("Absolute residual", ascending=False)
                st.dataframe(
                    full_state,
                    width="stretch",
                    hide_index=True,
                )
                visualization_note(
                    "Each row is a model feature for this historical observation. The "
                    "table reports observed and predicted next-state values, signed "
                    "residual, and absolute residual; it is sorted by largest error."
                )
        else:
            st.warning("The selected replay step is not present in the aligned replay data.")

    st.write("")
    section_title(
        "Review queue",
        "Filter and inspect observations requiring residual-based review.",
    )

    if len(review_df) == 0:
        st.success("No observations currently require residual-based review.")
    else:
        f1, f2 = st.columns([1, 1])

        with f1:
            selected_day = st.selectbox(
                "Capture day",
                ["All"] + sorted(
                    review_df["Target CaptureDay"].astype(str).unique().tolist()
                ),
            )

        with f2:
            search_step = st.text_input(
                "Search replay step",
                placeholder="Enter a step number",
            )

        filtered = review_df.copy()

        if selected_day != "All":
            filtered = filtered[
                filtered["Target CaptureDay"].astype(str) == selected_day
            ]

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
            "Target Label",
            "Anomaly Score",
            "Threshold",
            "Triage Status",
        ]].copy()

        st.dataframe(
            display,
           width="stretch",
            hide_index=True,
        )
        visualization_note(
            "Each row is a residual-threshold review candidate; columns locate it by "
            "step and timestamp and show retrospective label, anomaly score, threshold, "
            "and saved triage status."
        )

        if len(filtered):
            chosen = st.selectbox(
                "Open observation",
                filtered["Replay Step"].astype(int).tolist(),
            )
            detail = filtered[
                filtered["Replay Step"] == chosen
            ].iloc[0]

            st.markdown("#### Observation details")

            d1, d2, d3 = st.columns(3)
            with d1:
                metric_card(
                    "REPLAY STEP",
                    f"{int(chosen):,}",
                    str(detail["Target Timestamp"]),
                )
            with d2:
                metric_card(
                    "ANOMALY SCORE",
                    f"{float(detail['Anomaly Score']):.5f}",
                    "Mean absolute residual",
                )
            with d3:
                metric_card(
                    "THRESHOLD",
                    f"{float(detail['Threshold']):.5f}",
                    "Frozen validation threshold",
                )
            visualization_note(
                "Anomaly score is mean absolute prediction residual; threshold is the "
                "validation-calibrated review cutoff. Neither is an attack probability."
            )

            st.write(f"**Dataset reference label:** {detail['Target Label']}")
            st.caption(
                "The dataset label is retrospective reference information. "
                "It is not the LSTM's prediction or the triage decision."
            )

            st.write("**Reason:**", detail["Reason"])
            st.write("**Interpretation:**", detail["Interpretation"])

            if st.button("Open in Historical Replay", type="primary"):
                st.session_state.replay_step = int(chosen)
                st.session_state["navigation_target"] = "Historical Replay"
                st.rerun()

    with st.expander("How the triage rule works"):
        st.write(
            "The triage engine compares the mean absolute prediction residual "
            "across the model features with a threshold calibrated from benign "
            "validation observations. The held-out test results showed that "
            "this residual-only threshold did not flag the Bot observations. "
            "It should therefore be treated as an experimental review signal, "
            "not a validated attack detector."
        )

    footer()


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
        key="explorer_feature",
    )

    feature_idx = int(np.where(feature_names == feature)[0][0])
    residuals = np.abs(actual[:, feature_idx] - predictions[:, feature_idx])
    signed = actual[:, feature_idx] - predictions[:, feature_idx]

    a, b, c = st.columns(3)
    with a:
        metric_card(
            "FEATURE MEAN ABSOLUTE ERROR",
            f"{residuals.mean():.5f}",
            feature,
        )
    with b:
        metric_card(
            "FEATURE MEDIAN ABSOLUTE ERROR",
            f"{np.median(residuals):.5f}",
            "Across held-out replay",
        )
    with c:
        metric_card(
            "MAXIMUM ABSOLUTE ERROR",
            f"{residuals.max():.5f}",
            "Across held-out replay",
        )
    visualization_note(
        "Mean absolute error is the average feature prediction gap, median absolute "
        "error is the middle gap, and maximum absolute error is the largest gap across "
        "the held-out replay; all use transformed values."
    )

    st.write("")
    section_title(
        "Actual vs predicted",
        "Compare actual and predicted values for the selected feature over a replay window.",
    )

    max_start = max(1, len(df) - 100)
    start = st.slider(
        "Start replay step",
        min_value=1,
        max_value=max_start,
        value=1,
        key="feature_start",
    )

    end = min(len(df), start + 100)
    positions = np.arange(start - 1, end)

    comparison = pd.DataFrame(
        {
            "Actual": actual[positions, feature_idx],
            "Predicted": predictions[positions, feature_idx],
        },
        index=df.iloc[positions]["Replay Step"],
    )

    st.line_chart(comparison, height=320)
    visualization_note(
        "The x-axis is replay step and the y-axis is the selected feature's transformed "
        "value; the Actual and Predicted lines show the observed and LSTM next-state values."
    )

    section_title(
        "Feature residual analysis",
        "Signed residuals show prediction direction; absolute residuals show error magnitude.",
    )

    residual_chart = pd.DataFrame(
        {
            "Signed residual": signed[positions],
            "Absolute residual": residuals[positions],
        },
        index=df.iloc[positions]["Replay Step"],
    )

    st.line_chart(residual_chart, height=260)
    visualization_note(
        "The x-axis is replay step. Signed residual is actual minus predicted (positive "
        "means observed is higher); absolute residual shows error magnitude without direction."
    )

    st.write("")
    section_title(
        "Largest errors for this feature",
        "Observations with the largest absolute prediction errors for the selected feature.",
    )

    largest_errors = pd.DataFrame({
        "Replay Step": df["Replay Step"],
        "Timestamp": df["Target Timestamp"],
        "Reference Label": df["Target Label"],
        "Signed Residual": signed,
        "Absolute Error": residuals,
        "Triage Status": df["Triage Status"],
    }).nlargest(10, "Absolute Error")

    st.dataframe(
        largest_errors,
       width="stretch",
        hide_index=True,
    )
    visualization_note(
        "Each row is one of the ten largest selected-feature errors; columns give step, "
        "timestamp, retrospective reference label, signed and absolute residual, and review status."
    )

    st.write("")
    section_title(
        "Most frequent error contributors",
        "Features most often appearing among the top five per-observation residual contributors.",
    )

    if CONTRIBUTOR_CSV.exists():
        contributors = pd.read_csv(CONTRIBUTOR_CSV)

        required_contributor_columns = [
            "Feature",
            "Top 5 Appearance Count",
            "Appearance Percent",
        ]

        missing_contributor_columns = [
            column for column in required_contributor_columns
            if column not in contributors.columns
        ]

        if missing_contributor_columns:
            st.warning(
                "Contributor report is missing columns: "
                + ", ".join(missing_contributor_columns)
            )
        else:
            contributors = contributors.sort_values(
                "Appearance Percent",
                ascending=False,
            )

            top_contributors = contributors.head(15).copy()

            st.bar_chart(
                top_contributors.set_index("Feature")[["Appearance Percent"]],
                horizontal=True,
                height=420,
            )
            visualization_note(
                "Bars rank features by the percentage of replay observations in which "
                "they appeared among the five largest residual contributors; axes show "
                "feature and appearance percentage."
            )

            st.dataframe(
                top_contributors,
               width="stretch",
                hide_index=True,
            )
            visualization_note(
                "Rows are the most frequent residual contributors; columns show feature, "
                "appearance count, and percent of observations. This measures prediction "
                "error contribution, not attack relevance."
            )
    else:
        st.warning("The most frequent error contributors report was not found.")

    st.caption(
        "Residual contribution indicates model prediction difficulty. "
        "It does not establish that a feature is an attack indicator."
    )

    footer()


elif page == "SHAP Explainability":

    try:
        shap_artifacts = load_cached_shap_artifacts(
            tuple(str(name) for name in feature_names)
        )
    except (FileNotFoundError, ValueError, OSError, KeyError, BadZipFile) as exc:
        page_header(
            "MODEL INTERPRETABILITY",
            "SHAP Explainability",
            "Inspect saved V3 attributions for selected historical next-state predictions.",
        )
        st.error(f"SHAP artifacts could not be loaded or validated: {exc}")
        st.info(
            "The existing replay, residual, and triage pages remain available. "
            "No SHAP values are calculated by the dashboard."
        )
        footer()
    else:
        render_shap_explainability_page(shap_artifacts)


elif page == "Model & Dataset":

    page_header(
        "SYSTEM INFORMATION",
        "Model & Dataset",
        "Inspect the trained world model, its evaluation, dataset partitions and contextual threat mapping.",
    )

    input_features = int(manifest.get("input_features", len(feature_names)))
    sequence_length = int(manifest.get("sequence_length", input_sequences.shape[1]))
    checkpoint_epoch = int(manifest.get("checkpoint_epoch", 11))

    section_title(
        "Model configuration",
        "The trained architecture used by the historical replay.",
    )

    config1, config2, config3, config4 = st.columns(4)

    with config1:
        metric_card("ARCHITECTURE", "LSTM", "World Model V3")
    with config2:
        metric_card("HIDDEN SIZE", "128", "One recurrent layer")
    with config3:
        metric_card("INPUT WINDOW", str(sequence_length), "Historical states")
    with config4:
        metric_card("OUTPUT SIZE", str(len(feature_names)), "Next-state features")
    visualization_note(
        "Architecture identifies the recurrent model; hidden size is its internal "
        "representation width, input window is the number of past states, and output "
        "size is the number of predicted features."
    )

    st.write("")
    left, right = st.columns(2)

    with left:
        section_title("Training configuration")

        training_config = pd.DataFrame({
            "Parameter": [
                "Best checkpoint epoch",
                "Total training epochs",
                "Sequence length",
                "Input features",
                "Hidden size",
                "LSTM layers",
                "Dropout",
                "Optimizer",
                "Learning rate",
                "Loss",
                "Batch size",
                "Device used for training",
            ],
            "Value": [
                checkpoint_epoch,
                17,
                sequence_length,
                input_features,
                128,
                1,
                0.2,
                "AdamW",
                0.001,
                "Huber Loss",
                128,
                "CUDA (GPU training run)",
            ],
        })

        st.dataframe(
            training_config,
           width="stretch",
            hide_index=True,
        )
        visualization_note(
            "Each row is a training setting and Value is the recorded configuration. "
            "Sequence length counts past states, dropout is regularization, and Huber "
            "Loss is the training error measure."
        )

    with right:
        section_title("Replay configuration")

        replay_config = pd.DataFrame({
            "Property": [
                "Dataset",
                "Replay observations",
                "Prediction horizon",
                "Prediction method",
                "Predictions fed back as inputs",
                "Comparison space",
            ],
            "Value": [
                "CIC-IDS-2018",
                f"{manifest['number_of_steps']:,}",
                manifest["prediction_horizon"],
                manifest["prediction_method"],
                "No",
                manifest["prediction_comparison_space"],
            ],
        })

        st.dataframe(
            replay_config,
           width="stretch",
            hide_index=True,
        )
        visualization_note(
            "Each row describes how the saved replay was produced; Value gives the "
            "dataset, sample count, forecast horizon, method, input-feedback setting, "
            "or comparison space."
        )

    st.write("")
    section_title(
        "Held-out test evaluation",
        "One-step LSTM performance compared with the persistence baseline.",
    )

    lstm_metrics = {
        "Huber Loss": 0.12068179,
        "MAE": 0.32062694,
        "RMSE": 0.53132802,
    }

    persistence_metrics = {
        "Huber Loss": 0.17707066,
        "MAE": 0.35622746,
        "RMSE": 0.67903316,
    }

    eval_data = pd.DataFrame({
        "Metric": list(lstm_metrics.keys()),
        "LSTM": list(lstm_metrics.values()),
        "Persistence Baseline": list(persistence_metrics.values()),
    })

    eval_data["Improvement (%)"] = [
        (
            (persistence_metrics[metric] - lstm_metrics[metric])
            / persistence_metrics[metric]
        ) * 100
        for metric in lstm_metrics
    ]

    st.dataframe(
        eval_data,
       width="stretch",
        hide_index=True,
    )

    visualization_note(
        "Rows compare evaluation measures and columns show LSTM and persistence "
        "baseline values plus relative improvement. MAE is average absolute error, RMSE "
        "weights larger errors more, and Huber Loss combines squared and linear penalties; "
        "lower values mean smaller prediction errors. Improvement is the percentage "
        "reduction from the persistence baseline."
    )
    st.caption(
        "Metrics are measured in preprocessed model feature space, not original "
        "network units. The test partition has been used for exploratory analysis "
        "and should not be described as an untouched independent test."
    )

    st.write("")
    section_title(
        "Dataset partitions",
        "Chronological partitioning used for model development.",
    )

    split_data = pd.DataFrame({
        "Partition": ["Training", "Validation", "Held-out test"],
        "State rows": [199120, 51303, 31426],
        "Sequence samples": [197389, 50870, 31406],
        "Purpose": [
            "Model fitting",
            "Model selection and threshold calibration",
            "Held-out replay and exploratory evaluation",
        ],
    })

    st.dataframe(
        split_data,
       width="stretch",
        hide_index=True,
    )
    visualization_note(
        "Each row is a chronological dataset partition; columns show state rows, "
        "sequence samples, and the partition's role. These counts describe data splits, "
        "not model predictions."
    )

    st.write("")
    section_title(
        "MITRE ATT&CK contextual mapping",
        "Dataset labels are mapped to possible threat context where supporting evidence is available.",
    )

    st.caption(
        "These mappings are contextual research references. They are not generated "
        "by the LSTM and do not assign a technique to individual replay observations."
    )

    status_values = sorted(mitre["status"].dropna().astype(str).unique().tolist())

    selected_statuses = st.multiselect(
        "Filter mapping status",
        status_values,
        default=status_values,
        key="mitre_status_filter",
    )

    filtered_mitre = mitre[
        mitre["status"].astype(str).isin(selected_statuses)
    ].copy()

    mitre_display_columns = [
        column for column in [
            "label",
            "tactic",
            "technique",
            "name",
            "status",
            "evidence",
        ]
        if column in filtered_mitre.columns
    ]

    st.dataframe(
        filtered_mitre[mitre_display_columns],
       width="stretch",
        hide_index=True,
    )
    visualization_note(
        "Each row maps a dataset reference label to possible ATT&CK tactic/technique "
        "context; columns show mapping status and evidence. These are contextual references, "
        "not LSTM predictions or per-observation diagnoses."
    )

    if len(filtered_mitre):
        selected_label = st.selectbox(
            "Inspect a dataset label",
            sorted(filtered_mitre["label"].dropna().astype(str).unique()),
            key="mitre_label",
        )

        selected_mapping = filtered_mitre[
            filtered_mitre["label"].astype(str) == selected_label
        ].iloc[0]

        st.markdown("#### Mapping details")
        st.write(f"**Dataset label:** {selected_mapping['label']}")
        st.write(f"**Tactic:** {selected_mapping['tactic']}")
        st.write(f"**Technique:** {selected_mapping['technique']}")
        st.write(f"**Technique name:** {selected_mapping['name']}")
        st.write(f"**Mapping status:** {selected_mapping['status']}")
        st.write(f"**Evidence:** {selected_mapping['evidence']}")
        st.write(f"**Notes:** {selected_mapping['notes']}")

        if "source" in selected_mapping.index:
            st.write(f"**Source:** {selected_mapping['source']}")

    else:
        st.info("No MITRE mappings match the selected status filters.")

    with st.expander("Research limitations and interpretation"):
        st.markdown("""
        - The LSTM predicts the next network state; it is not a direct attack classifier.
        - The test replay uses actual historical input windows, not prediction feedback.
        - Residual-only triage did not flag the Bot observations in the held-out test set.
        - One NO_FLOWS observation crossed the test review threshold; this is not evidence of an attack.
        - Dataset labels are reference labels, not LSTM predictions.
        - MITRE ATT&CK mappings are contextual candidates and depend on supporting evidence.
        - Feature values and prediction errors are in transformed model space.
        - The validation and test periods contain different traffic distributions and attack labels.
        - The test partition has been used for exploratory analysis, not as an untouched independent test.
        """)

    footer()