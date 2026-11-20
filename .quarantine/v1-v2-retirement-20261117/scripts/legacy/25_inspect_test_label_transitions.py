from pathlib import Path
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")

BASE = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "temporal_error_analysis_v1"
)

RUNS_FILE = BASE / "consecutive_label_runs.csv"

STATES_FILE = (
    ROOT / "Datasets" / "sentinelx_multiday_states.csv"
)

if not RUNS_FILE.exists():
    raise FileNotFoundError(RUNS_FILE)

if not STATES_FILE.exists():
    raise FileNotFoundError(STATES_FILE)

runs = pd.read_csv(RUNS_FILE)

runs["start_time"] = pd.to_datetime(runs["start_time"])
runs["end_time"] = pd.to_datetime(runs["end_time"])

print("=" * 65)
print("SENTINELX - TEST LABEL TRANSITION INSPECTION")
print("=" * 65)

print("\nTOTAL RUNS:", len(runs))

print("\nRUN COUNTS BY LABEL")
print(runs["label"].value_counts().to_string())

print("\nRUN LENGTH DISTRIBUTION BY LABEL")
print(
    runs.groupby("label")["states"]
    .describe()
    .to_string()
)

print("\nONE-STATE RUNS BY LABEL")
print(
    runs[runs["states"] == 1]["label"]
    .value_counts()
    .to_string()
)

print("\nFIRST 20 RUNS")
print(runs.head(20).to_string(index=False))

print("\nLAST 20 RUNS")
print(runs.tail(20).to_string(index=False))

print("\n15 LONGEST RUNS")
print(
    runs.nlargest(15, "states")[
        ["start_time", "end_time", "label", "states", "mean_error"]
    ].to_string(index=False)
)

print("\nFIRST BOT RUN")
bot_runs = runs[runs["label"] == "Bot"]

if len(bot_runs):
    print(bot_runs.head(1).to_string(index=False))
    print("\nLAST BOT RUN")
    print(bot_runs.tail(1).to_string(index=False))

print("\n" + "=" * 65)
print("CHECKING ORIGINAL NETWORK STATES")
print("=" * 65)

# Read only the columns needed for this diagnostic.
states = pd.read_csv(
    STATES_FILE,
    usecols=lambda c: c in {
        "Timestamp",
        "Label",
        "Flow Count",
        "Has Flows",
        "Segment ID"
    }
)

states["Timestamp"] = pd.to_datetime(
    states["Timestamp"], errors="raise"
)

# Inspect the start of the label-transition region.
start = pd.Timestamp("2018-03-02 10:13:20")
end = pd.Timestamp("2018-03-02 10:13:45")

sample = states[
    (states["Timestamp"] >= start)
    & (states["Timestamp"] <= end)
]

print("\nORIGINAL STATES AROUND 10:13:28")
print(sample.to_string(index=False))

# Inspect the final Bot period.
start = pd.Timestamp("2018-03-02 12:54:50")
end = pd.Timestamp("2018-03-02 12:55:10")

sample = states[
    (states["Timestamp"] >= start)
    & (states["Timestamp"] <= end)
]

print("\nORIGINAL STATES AROUND 12:55")
print(sample.to_string(index=False))

# Flow-count statistics by label in the test time range.
test_start = pd.Timestamp("2018-03-02 08:54:38")
test_end = pd.Timestamp("2018-03-02 12:59:59")

test_states = states[
    (states["Timestamp"] >= test_start)
    & (states["Timestamp"] <= test_end)
]

if "Flow Count" in test_states.columns:
    print("\nFLOW COUNT BY LABEL IN TEST TIME RANGE")

    print(
        test_states.groupby("Label")["Flow Count"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .to_string()
    )

if "Segment ID" in test_states.columns:
    print("\nSEGMENT DISTRIBUTION IN TEST TIME RANGE")
    print(
        test_states.groupby("Label")["Segment ID"]
        .value_counts()
        .to_string()
    )

print("\nINSPECTION COMPLETED")
