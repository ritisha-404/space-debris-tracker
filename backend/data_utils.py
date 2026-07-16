"""
data_utils.py

Synthetic orbital trajectory generation + windowing helpers for LSTM training.

"""

import numpy as np


def _elliptical_orbit(num_steps, radius, eccentricity, inclination_deg, noise_std, phase_offset=0.0):
    """Generate one noisy elliptical orbit trajectory in 3D.

    Returns an array of shape (num_steps, 3) of (x, y, z) positions.
    """
    theta = np.linspace(0, 4 * np.pi, num_steps) + phase_offset
    a = radius
    b = radius * (1 - eccentricity)

    x = a * np.cos(theta)
    y = b * np.sin(theta)

    incl = np.radians(inclination_deg)
    z = y * np.sin(incl)
    y = y * np.cos(incl)

    noise = np.random.normal(0, noise_std, size=(num_steps, 3))
    positions = np.stack([x, y, z], axis=1) + noise
    return positions.astype(np.float32)


def generate_synthetic_debris_set(num_objects=25, num_steps=200, seed=42):
    """Generate a set of synthetic debris trajectories.

    Returns a dict: {object_id: np.ndarray of shape (num_steps, 3)}
    """
    rng = np.random.default_rng(seed)
    dataset = {}
    for i in range(num_objects):
        radius = rng.uniform(7000, 8000)       # km, roughly LEO altitude + Earth radius
        eccentricity = rng.uniform(0.0, 0.15)
        inclination = rng.uniform(0, 98)        # degrees
        noise_std = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)

        traj = _elliptical_orbit(
            num_steps=num_steps,
            radius=radius,
            eccentricity=eccentricity,
            inclination_deg=inclination,
            noise_std=noise_std,
            phase_offset=phase,
        )
        dataset[f"DEBRIS-{i:03d}"] = traj
    return dataset


def make_windows(trajectory, seq_len=10):
    """Slice a single (T, 3) trajectory into (X, y) LSTM training windows.

    X has shape (num_windows, seq_len, 3)
    y has shape (num_windows, 3)  -- the position immediately following each window
    """
    X, y = [], []
    for start in range(len(trajectory) - seq_len):
        X.append(trajectory[start:start + seq_len])
        y.append(trajectory[start + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_training_set(dataset, seq_len=10):
    """Combine windows from every object in `dataset` into one training set."""
    all_X, all_y = [], []
    for traj in dataset.values():
        X, y = make_windows(traj, seq_len=seq_len)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


if __name__ == "__main__":
    data = generate_synthetic_debris_set(num_objects=5, num_steps=50)
    X, y = build_training_set(data, seq_len=10)
    print(f"Generated {len(data)} trajectories")
    print(f"Training windows: X={X.shape}, y={y.shape}")
