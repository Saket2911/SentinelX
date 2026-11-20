from pathlib import Path
import csv

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)

ROOT = Path(r"F:\Projects\SentinelX")

INPUT = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "anomaly_experiment_v1"
    / "test_anomaly_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "Models" / "lstm_world_model_v1"
    / "anomaly_score_diagnostics_v1"
)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT_DIR}"
    )

OUTPUT_DIR.mkdir(parents=True)

labels = []
scores = []

with INPUT.open("r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        labels.append(row["actual_label"])
        scores.append(float(row["prediction_error"]))

labels = np.asarray(labels)
scores = np.asarray(scores, dtype=np.float64)

actual = (labels == "Bot").astype(int)

if set(labels) != {"Benign", "Bot"}:
    raise ValueError(f"Unexpected labels: {set(labels)}")

if len(np.unique(actual)) != 2:
    raise ValueError("Both classes are required.")

print("=" * 65)
print("SENTINELX - ANOMALY SCORE DISTRIBUTION")
print("=" * 65)

print("Total test sequences:", len(scores))
print("Benign:", int(np.sum(actual == 0)))
print("Bot:", int(np.sum(actual == 1)))


# -------------------------------------------------
# 1. Score distributions
# -------------------------------------------------

distribution_rows = []

for label in ["Benign", "Bot"]:

    values = scores[labels == label]

    percentiles = np.percentile(
        values, [25, 50, 75, 90, 95, 99]
    )

    row = {
        "label": label,
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "minimum": float(np.min(values)),
        "p25": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p90": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "maximum": float(np.max(values))
    }

    distribution_rows.append(row)

    print(f"\n{label} prediction-error distribution")

    for key, value in row.items():
        print(f"{key}: {value}")


# -------------------------------------------------
# 2. Ranking performance
# -------------------------------------------------

high_error_roc = roc_auc_score(actual, scores)
high_error_pr = average_precision_score(actual, scores)

# Reverse direction: lower prediction error means Bot.
low_error_roc = roc_auc_score(actual, -scores)
low_error_pr = average_precision_score(actual, -scores)

print("\n" + "=" * 65)
print("RANKING PERFORMANCE")
print("=" * 65)

print("\nHigher error = more suspicious")
print(f"ROC-AUC: {high_error_roc:.6f}")
print(f"PR-AUC:  {high_error_pr:.6f}")

print("\nLower error = more suspicious")
print(f"ROC-AUC: {low_error_roc:.6f}")
print(f"PR-AUC:  {low_error_pr:.6f}")


# -------------------------------------------------
# 3. Simple reference baseline
# -------------------------------------------------

# Always predict the majority class: Bot.
always_bot = np.ones(len(actual), dtype=int)

print("\n" + "=" * 65)
print("MAJORITY-CLASS REFERENCE")
print("=" * 65)

print(
    "Always-Bot accuracy:",
    f"{accuracy_score(actual, always_bot):.6f}"
)

print(
    "Always-Bot F1:",
    f"{f1_score(actual, always_bot):.6f}"
)


# -------------------------------------------------
# 4. Save report
# -------------------------------------------------

report = f"""
SENTINELX - ANOMALY SCORE DIAGNOSTICS

TEST SEQUENCES: {len(scores)}

LABEL DISTRIBUTIONS:
{distribution_rows}

HIGHER ERROR = MORE SUSPICIOUS:
ROC-AUC: {high_error_roc}
PR-AUC: {high_error_pr}

LOWER ERROR = MORE SUSPICIOUS:
ROC-AUC: {low_error_roc}
PR-AUC: {low_error_pr}

ALWAYS-BOT REFERENCE:
Accuracy: {accuracy_score(actual, always_bot)}
F1: {f1_score(actual, always_bot)}

NOTE:
These are post-hoc diagnostics on the existing test
evaluation. They must not be used to tune thresholds
or select a model. Any future model selection and
threshold calibration must use training/validation data.
"""

report_path = OUTPUT_DIR / "score_distribution_report.txt"

report_path.write_text(report, encoding="utf-8")

csv_path = OUTPUT_DIR / "score_distribution.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(distribution_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(distribution_rows)

print("\nReports saved:")
print(report_path)
print(csv_path)
