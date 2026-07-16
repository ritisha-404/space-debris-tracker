"""
train.py

Trains DebrisLSTM on synthetic orbital trajectories and saves a checkpoint to
models/lstm_debris.pt.

Usage:
    python backend/train.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import generate_synthetic_debris_set, build_training_set
from model import DebrisLSTM

SEQ_LEN = 10
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_debris.pt")


def normalize(X, y):
    """Simple mean/std normalization, fit on X, applied to both X and y."""
    mean = X.reshape(-1, 3).mean(axis=0)
    std = X.reshape(-1, 3).std(axis=0) + 1e-6
    X_norm = (X - mean) / std
    y_norm = (y - mean) / std
    return X_norm, y_norm, mean, std


def main():
    print("Generating synthetic training data...")
    dataset = generate_synthetic_debris_set(num_objects=30, num_steps=200)
    X, y = build_training_set(dataset, seq_len=SEQ_LEN)
    X, y, mean, std = normalize(X, y)
    print(f"Training set: X={X.shape}, y={y.shape}")

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=BATCH_SIZE, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DebrisLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Training on {device} for {EPOCHS} epochs...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch:3d}/{EPOCHS} - MSE loss: {avg_loss:.5f}")

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)

    # Save normalization stats alongside the checkpoint so the API can use them
    norm_path = CHECKPOINT_PATH.replace(".pt", "_norm.npz")
    np.savez(norm_path, mean=mean, std=std)

    print(f"Saved model checkpoint to {CHECKPOINT_PATH}")
    print(f"Saved normalization stats to {norm_path}")


if __name__ == "__main__":
    main()
