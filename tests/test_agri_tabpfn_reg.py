"""Test script for AgriTabPFNRegressor.

This script tests the functionality of the AgriTabPFNRegressor by creating synthetic
agricultural data and verifying that the model can process and make predictions on it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn.agri_tabpfn import AgriTabPFNRegressor


def create_synthetic_agri_data(
    n_samples: int = 100,
    n_time_steps: int = 120,
    n_features: int = 10,
    missing_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic agricultural data for testing.

    Args:
        n_samples: Number of samples
        n_time_steps: Number of time steps
        n_features: Number of features per time step
        missing_ratio: Ratio of missing values
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X, y) where X is of shape (n_samples, n_time_steps+1, n_features)
        and y is of shape (n_samples,)
    """
    # Set random seed
    np.random.seed(random_state)

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


def test_agri_tabpfn_regressor():
    """Test the AgriTabPFNRegressor with synthetic data."""
    X, y = create_synthetic_agri_data(n_samples=200, n_time_steps=30, n_features=5)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Initialize the model
    model = AgriTabPFNRegressor(
        n_estimators=2,  # Use fewer estimators for faster testing
        temporal_hidden_dim=64,
        static_hidden_dims=[32],
        fusion_strategy="gated",
        random_state=42,
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mean_squared_error(y_test, y_pred)
    r2_score(y_test, y_pred)

    # Test quantile predictions
    quantiles = [0.1, 0.5, 0.9]
    model.predict(X_test, output_type="quantiles", quantiles=quantiles)

    for _i, _q in enumerate(quantiles):
        pass

    # Test embedding extraction
    model.get_embeddings(X_test)


if __name__ == "__main__":
    test_agri_tabpfn_regressor()
