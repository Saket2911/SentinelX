import numpy as np
import pandas as pd
from pathlib import Path

root = Path(r"F:\Projects\SentinelX")
data = root / "Datasets"

files = [
    "sentinelx_test_X.npy",
    "sentinelx_test_y.npy",
    "sentinelx_test_target_labels.npy",
    "sentinelx_test_target_timestamps.npy",
    "sentinelx_test_target_segment_ids.npy",
]

for filename in files:
    path = data / filename
    print(f"\n{filename}")
    print("Exists:", path.exists())

    if path.exists():
        arr = np.load(path, mmap_mode="r")
        print("Shape:", arr.shape)
        print("Dtype:", arr.dtype)

labels = np.asarray(
    np.load(data / "sentinelx_test_target_labels.npy")
).astype(str)

timestamps = pd.to_datetime(
    np.load(data / "sentinelx_test_target_timestamps.npy")
)

print("\nLabel distribution:")
print(pd.Series(labels).value_counts())

print("\nFirst 15 target labels:")
for i in range(15):
    print(i, timestamps[i], labels[i])

print("\nLast 15 target labels:")
for i in range(len(labels) - 15, len(labels)):
    print(i, timestamps[i], labels[i])

print("\nTransition directions:")
for i in range(1, len(labels)):
    if labels[i] != labels[i - 1]:
        print(
            timestamps[i],
            labels[i - 1],
            "->",
            labels[i]
        )
