from pathlib import Path
import json
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path("/mnt/f/Projects/SentinelX")
MODEL = ROOT / "Models/lstm_world_model_v3_gpu_run2"

REPLAY = MODEL / "historical_replay_v1"
EXPLAIN = MODEL / "explainable_triage_v2"
TRIAGE = MODEL / "triage_engine_v1"

def inspect_csv(path, name):
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    if not path.exists():
        print("MISSING:", path)
        return

    df = pd.read_csv(path, nrows=5)
    print("Path:", path)
    print("Columns:", df.columns.tolist())
    print("First rows:")
    print(df.to_string(index=False))

def main():
    print("=" * 75)
    print("SENTINELX STEP 32A — DASHBOARD DATA INSPECTION")
    print("=" * 75)

    print("\nSTREAMLIT INSTALLED:",
          importlib.util.find_spec("streamlit") is not None)

    manifest_path = REPLAY / "replay_manifest.json"
    print("\nREPLAY MANIFEST")

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        print(json.dumps(manifest, indent=2))
    else:
        print("Missing:", manifest_path)

    npz_path = REPLAY / "replay_arrays.npz"
    print("\nREPLAY ARRAYS")

    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            for key in data.files:
                arr = data[key]
                print(
                    key,
                    "| shape:", arr.shape,
                    "| dtype:", arr.dtype
                )
    else:
        print("Missing:", npz_path)

    inspect_csv(REPLAY / "test_replay.csv", "HISTORICAL REPLAY CSV")
    inspect_csv(
        EXPLAIN / "explainable_triage.csv",
        "EXPLAINABLE TRIAGE CSV"
    )
    inspect_csv(
        EXPLAIN / "most_frequent_error_contributors.csv",
        "ERROR CONTRIBUTORS CSV"
    )
    inspect_csv(
        TRIAGE / "test_triage.csv",
        "TEST TRIAGE CSV"
    )

    print("\nFEATURE MAP")
    feature_path = ROOT / "Datasets/sentinelx_model_features_v3.json"

    if feature_path.exists():
        features = json.loads(feature_path.read_text())
        print("Feature map type:", type(features).__name__)
        print("Feature count:", len(features))
        print("First 15 entries:", features[:15] if isinstance(features, list)
              else list(features)[:15])
    else:
        print("Missing:", feature_path)

    print("\n" + "=" * 75)
    print("INSPECTION COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
