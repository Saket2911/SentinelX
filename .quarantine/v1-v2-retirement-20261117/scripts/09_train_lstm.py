
from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
import torch
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinelx.data.loader import load_dataset
from sentinelx.data.splitter import split_dataset
from sentinelx.features.sequences import (
    build_network_states,
    make_sequences,
)
from sentinelx.features.state_preprocessor import StatePreprocessor
from sentinelx.models.lstm_world_model import LSTMWorldModel
from sentinelx.training.train_lstm import train_lstm


DATASET_PATH = (
    PROJECT_ROOT
    / "Datasets"
    / "Wednesday-14-02-2018_prepared.csv"
)

MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
PREPROCESSOR_DIR = PROJECT_ROOT / "artifacts" / "preprocessors"
REPORT_DIR = PROJECT_ROOT / "reports"

CHECKPOINT_PATH = MODEL_DIR / "lstm_world_model.pt"
PREPROCESSOR_PATH = PREPROCESSOR_DIR / "state_preprocessor.joblib"
HISTORY_PATH = REPORT_DIR / "lstm_training_history.csv"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. Training stopped."
        )

    print("GPU training enabled.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    # Reproducibility.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading dataset...")
    df = load_dataset(DATASET_PATH)

    print("\n2. Creating chronological splits...")
    train_df, validation_df, test_df = split_dataset(df)

    print("\n3. Building timestamped network states...")
    train_states = build_network_states(train_df)
    validation_states = build_network_states(validation_df)
    test_states = build_network_states(test_df)

    print("Training states:", len(train_states))
    print("Validation states:", len(validation_states))
    print("Test states:", len(test_states))

    print("\n4. Fitting state preprocessing on training data only...")
    preprocessor = StatePreprocessor()

    train_states = preprocessor.fit_transform(train_states)
    validation_states = preprocessor.transform(validation_states)
    test_states = preprocessor.transform(test_states)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("Preprocessor saved:", PREPROCESSOR_PATH)

    print("\n5. Creating next-state sequences...")
    X_train, y_train, _, _, feature_columns = make_sequences(
        train_states, sequence_length=10
    )

    X_validation, y_validation, _, _, validation_features = make_sequences(
        validation_states, sequence_length=10
    )

    # Prepare test sequences for final evaluation later.
    # Do not use test loss to select epochs or tune the model.
    X_test, y_test, test_timestamps, test_labels, test_features = (
        make_sequences(test_states, sequence_length=10)
    )

    if feature_columns != validation_features:
        raise ValueError("Training and validation feature columns differ.")

    if feature_columns != test_features:
        raise ValueError("Training and test feature columns differ.")

    if X_train.shape[2] != len(feature_columns):
        raise ValueError("Sequence feature count does not match metadata.")

    print("Training X:", X_train.shape)
    print("Training y:", y_train.shape)
    print("Validation X:", X_validation.shape)
    print("Validation y:", y_validation.shape)
    print("Test X (reserved):", X_test.shape)
    print("Test y (reserved):", y_test.shape)

    print("\n6. Creating LSTM World Model...")
    model = LSTMWorldModel(
        input_size=X_train.shape[2],
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    )

    print("\n7. Training with validation monitoring...")
    model, history, device = train_lstm(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        checkpoint_path=CHECKPOINT_PATH,
        epochs=30,
        batch_size=64,
        learning_rate=0.001,
        patience=5,
    )

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    print("\nTraining completed.")
    print("Best model:", CHECKPOINT_PATH)
    print("Training history:", HISTORY_PATH)
    print("Preprocessor:", PREPROCESSOR_PATH)
    print("Test set has not been evaluated or used for model selection.")


if __name__ == "__main__":
    main()
