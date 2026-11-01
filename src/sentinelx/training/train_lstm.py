
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def create_data_loader(X, y, batch_size=256, shuffle=False):
    """Convert sequence arrays into PyTorch mini-batches."""

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if X.ndim != 3:
        raise ValueError(f"Expected 3D X array, got {X.shape}")

    if y.ndim != 2:
        raise ValueError(f"Expected 2D y array, got {y.shape}")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    if len(X) == 0:
        raise ValueError("Cannot train or evaluate on an empty dataset.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X or y contains NaN or infinite values.")

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def evaluate_loss(model, data_loader, criterion, device):
    """Calculate sample-weighted mean MSE without updating weights."""

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            batch_count = X_batch.size(0)
            total_loss += loss.item() * batch_count
            total_samples += batch_count

    return total_loss / total_samples


def train_lstm(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    checkpoint_path,
    epochs=30,
    batch_size=256,
    learning_rate=0.001,
    patience=5,
):
    """Train the next-network-state prediction model."""

    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive.")

    train_loader = create_data_loader(
        X_train, y_train, batch_size=batch_size, shuffle=True
    )
    validation_loader = create_data_loader(
        X_validation, y_validation,
        batch_size=batch_size, shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    best_validation_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Training device: {device}")
    print(f"Training sequences: {len(X_train):,}")
    print(f"Validation sequences: {len(X_validation):,}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            loss.backward()

            # Reduce the risk of exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            batch_count = X_batch.size(0)
            total_train_loss += loss.item() * batch_count
            total_samples += batch_count

        train_loss = total_train_loss / total_samples

        validation_loss = evaluate_loss(
            model, validation_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": validation_loss,
        })

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Validation MSE: {validation_loss:.6f}"
        )

        if np.isfinite(validation_loss) and (
            validation_loss < best_validation_loss
        ):
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "input_size": model.input_size,
                    "best_validation_mse": best_validation_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping: validation loss did not improve.")
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid checkpoint. "
            "Check the data and losses."
        )

    model.load_state_dict(best_state)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation MSE: {best_validation_loss:.6f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    return model, history, device
