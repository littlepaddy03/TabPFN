"""Test script for the agricultural data encoder.

This script tests the functionality of the AgriDataEncoder by creating synthetic
agricultural data and verifying that the encoder can process it correctly.
"""

from __future__ import annotations

import numpy as np
import torch

from tabpfn.encoders.agri_interface import (
    encode_agri_batch_for_tabpfn,
    prepare_agri_data_for_tabpfn,
)


def create_synthetic_agri_data(
    n_samples: int = 100,
    n_time_steps: int = 120,
    n_features: int = 10,
    missing_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic agricultural data for testing.

    Args:
        n_samples: Number of samples
        n_time_steps: Number of time steps
        n_features: Number of features per time step
        missing_ratio: Ratio of missing values

    Returns:
        Tuple of (X, y) where X is of shape (n_samples, n_time_steps+1, n_features)
        and y is of shape (n_samples,)
    """
    # Create temporal data with seasonality
    X = np.zeros((n_samples, n_time_steps + 1, n_features))

    # Fill temporal part with seasonal patterns
    for i in range(n_samples):
        for j in range(n_features):
            # Create seasonal pattern with some noise
            season = np.sin(np.linspace(0, 4 * np.pi, n_time_steps)) + np.random.normal(
                0, 0.2, n_time_steps
            )
            X[i, :n_time_steps, j] = season

    # Fill static part (last row) with random values
    X[:, -1, :] = np.random.normal(0, 1, (n_samples, n_features))

    # Introduce missing values
    mask = np.random.random(X.shape) < missing_ratio
    X[mask] = np.nan

    # Create target variable (yield) based on both temporal and static features
    # with some non-linear relationships
    temporal_importance = np.random.normal(0, 1, n_features)
    static_importance = np.random.normal(0, 1, n_features)

    # Extract key temporal features (e.g., critical growth periods)
    critical_period_start = n_time_steps // 3
    critical_period_end = 2 * n_time_steps // 3
    critical_period_avg = np.nanmean(
        X[:, critical_period_start:critical_period_end, :], axis=1
    )

    # Combine temporal and static features to create yield
    temporal_component = np.sum(critical_period_avg * temporal_importance, axis=1)
    static_component = np.sum(X[:, -1, :] * static_importance, axis=1)

    # Add interaction term and noise
    interaction = np.sum(critical_period_avg * X[:, -1, :], axis=1) * 0.2
    noise = np.random.normal(0, 0.5, n_samples)

    y = temporal_component + static_component + interaction + noise

    return X, y


def test_agri_data_encoder():
    """Test the agricultural data encoder with synthetic data."""
    X, y = create_synthetic_agri_data()

    preprocessor, encoder, X_processed, y_processed = prepare_agri_data_for_tabpfn(X, y)

    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    with torch.no_grad():
        encoder(X_tensor)

    encode_agri_batch_for_tabpfn(X_processed, encoder)


if __name__ == "__main__":
    test_agri_data_encoder()
