
from pathlib import Path
import pandas as pd

ROOT = Path(r"F:\Projects\SentinelX")
DATA = ROOT / "Datasets"

FILES = {
    "TRAIN": DATA / "sentinelx_train_sequence_metadata_v1.csv",
    "VALIDATION": DATA / "sentinelx_validation_sequence_metadata_v1.csv",
    "TEST": DATA / "sentinelx_test_sequence_metadata_v1.csv",
}

for partition, path in FILES.items():

    print("\n" + "=" * 80)
    print(f"{partition} — ATTACK TIMELINE INSPECTION")
    print("=" * 80)

    df = pd.read_csv(path)

    df["Target Timestamp"] = pd.to_datetime(
        df["Target Timestamp"],
        errors="coerce"
    )

    print("\nOverall label counts:")
    print(df["Target Label"].value_counts().to_string())

    print("\nSegment-level label distribution:")

    grouped = df.groupby("Segment ID")

    for segment_id, group in grouped:

        counts = group["Target Label"].value_counts()

        start = group["Target Timestamp"].min()
        end = group["Target Timestamp"].max()

        print("\n" + "-" * 60)
        print("Segment:", segment_id)
        print("Samples:", len(group))
        print("Start:", start)
        print("End:", end)
        print("Labels:", counts.to_dict())

        # Print label transitions in timestamp order
        ordered = group.sort_values("Target Timestamp")

        ordered_labels = ordered["Target Label"].to_numpy()
        ordered_times = ordered["Target Timestamp"].to_numpy()

        transitions = []

        for i in range(1, len(ordered_labels)):

            if ordered_labels[i] != ordered_labels[i - 1]:

                transitions.append({
                    "From": ordered_labels[i - 1],
                    "To": ordered_labels[i],
                    "Timestamp": ordered_times[i]
                })

        print("Label transitions:", len(transitions))

        for transition in transitions[:15]:
            print(
                f"  {transition['From']} -> "
                f"{transition['To']} at "
                f"{transition['Timestamp']}"
            )

        if len(transitions) > 15:
            print("  Additional transitions omitted.")

print("\n" + "=" * 80)
print("ATTACK TIMELINE INSPECTION COMPLETED")
print("No dataset or model files were modified.")

