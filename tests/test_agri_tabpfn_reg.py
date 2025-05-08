"""Test script for AgriTabPFNRegressor.

This script tests the functionality of the AgriTabPFNRegressor by creating synthetic
3D agricultural data and verifying that the model can process and make predictions on it.
"""
import os
# Add HF mirror if needed, keep if already there
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # Comment out if not needed

import logging # Import logging
import numpy as np
import pytest # Import pytest
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.base import check_is_fitted # Import check_is_fitted
from typing import Optional, List, Tuple # Import necessary types

from tabpfn.agri_tabpfn import AgriTabPFNRegressor

# Setup logger for this module
logger = logging.getLogger(__name__)


# Keep or refine the synthetic data generation function
# Make it generate 3D data consistently
def create_synthetic_agri_data_3d(
    n_samples=200,
    n_time_steps=30,
    n_features=5,
    missing_ratio=0.1,
    static_row_index=-1
    ) -> tuple[np.ndarray, np.ndarray]: # Added type hint
    """
    Creates synthetic 3D agricultural data.
    X shape: (n_samples, n_time_steps + 1, n_features)
    y shape: (n_samples,)
    """
    n_rows = n_time_steps + 1
    if not (-n_rows <= static_row_index < n_rows):
         raise ValueError(f"static_row_index {static_row_index} out of bounds for {n_rows} rows.")
    resolved_static_index = static_row_index % n_rows
    temporal_indices = [i for i in range(n_rows) if i != resolved_static_index]

    # Generate random features - initialize full array
    X = np.random.rand(n_samples, n_rows, n_features) * 5 # Scale temporal differently

    # Assign temporal and static parts conceptually for yield calculation
    # Make static features potentially different scale/distribution
    X[:, resolved_static_index, :] = np.random.randn(n_samples, n_features) * 10 + 20

    # --- Introduce missing values (NaNs) ---
    if missing_ratio > 0:
        nan_mask = np.random.random(X.shape) < missing_ratio
        # Avoid NaNs in the static row for simpler yield calculation logic here
        nan_mask[:, resolved_static_index, :] = False
        X[nan_mask] = np.nan

    # --- Calculate yield (y) based on features (example logic) ---
    temporal_data_view = X[:, temporal_indices, :] # View for calculation
    static_data_view = X[:, resolved_static_index, :] # View for calculation

    # Define importance for yield
    temporal_importance = np.random.rand(n_features) * 0.6
    static_importance = np.random.rand(n_features) * 1.2

    # Simulate a "critical period" within the temporal data
    # Ensure indices are valid for potentially 0 temporal steps
    if n_time_steps > 0:
        critical_period_len = max(1, n_time_steps // 3)
        critical_period_start = max(0, n_time_steps // 3)
        critical_period_end = min(n_time_steps, critical_period_start + critical_period_len)

        critical_period_data = temporal_data_view[:, critical_period_start:critical_period_end, :]
        # Average over time axis (axis=1), handling NaNs
        critical_period_avg = np.nanmean(critical_period_data, axis=1)
        # Handle cases where whole features might be NaN in critical period
        critical_period_avg = np.nan_to_num(critical_period_avg, nan=0.0)
        temporal_component = np.sum(critical_period_avg * temporal_importance, axis=1)
    else:
        temporal_component = np.zeros(n_samples) # No temporal contribution


    # Contribution from static features (handle potential NaNs if introduced, though avoided above)
    static_component = np.nansum(static_data_view * static_importance, axis=1) # Use nansum just in case

    # Interaction term (example)
    if n_time_steps > 0:
         # Impute NaNs in static view before multiplication for interaction term
         static_data_imputed_for_interaction = np.nan_to_num(static_data_view, nan=0.0)
         interaction = np.nansum(critical_period_avg * static_data_imputed_for_interaction, axis=1) * 0.1
    else:
         interaction = np.zeros(n_samples)

    noise = np.random.normal(0, 0.2, n_samples)

    y = temporal_component + static_component + interaction + noise
    # Ensure no NaNs in final y (e.g., if all components were somehow NaN)
    # Impute final y NaNs with the mean of non-NaN y values
    if np.isnan(y).any():
        y_mean = np.nanmean(y)
        y = np.nan_to_num(y, nan=y_mean)


    return X, y

# --- Test Function ---
# Use pytestmark for potential slow tests if needed
# @pytest.mark.slow
def test_agri_tabpfn_regressor_interface():
    """Test the AgriTabPFNRegressor interface with 3D synthetic data."""
    print("\n--- Testing AgriTabPFNRegressor Interface ---")
    print("Creating 3D synthetic agricultural data...")
    # Use the 3D data generator
    X, y = create_synthetic_agri_data_3d(n_samples=80, n_time_steps=15, n_features=4, missing_ratio=0.1, static_row_index=-1)
    print(f"Generated Data Shapes: X={X.shape}, y={y.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Split Shapes: Train X={X_train.shape}, y={y_train.shape} | Test X={X_test.shape}, y={y_test.shape}")

    # Initialize the model with some agri-specific params
    print("\nInitializing AgriTabPFNRegressor...")
    model = AgriTabPFNRegressor(
        # Agri-specific
        temporal_hidden_dim=32,
        static_hidden_dims=[16],
        fusion_strategy="gated",
        encoded_embedding_dim=64,
        max_temporal_length=10, # Test truncation
        handle_missing_agri_values='mean',
        standardize_agri_features=True,
        static_row_index=-1,
        # Underlying TabPFN
        n_estimators=2, # Fewer for faster testing
        device='cpu', # Force CPU for easier test runs
        random_state=42,
        # Set fit_mode explicitly if needed, default is 'fit_preprocessors'
        # fit_mode='low_memory',
    )
    print(f"Model Initialized: {model.get_params()}") # Print params

    # Fit the model
    print("\nFitting the model...")
    model.fit(X_train, y_train)
    print("Fit completed.")

    # --- Check Fitted Attributes ---
    print("\nChecking fitted attributes...")
    # Use sklearn's check_is_fitted, which relies on __sklearn_is_fitted__
    check_is_fitted(model)
    # Check internal flag as well
    assert model._is_fitted
    # Check that essential components seem initialized/fitted
    assert model.agri_preprocessor_ is not None and model.agri_preprocessor_.is_fitted_
    assert model.agri_encoder_ is not None
    assert model.tabpfn_regressor_ is not None
    # Check attributes derived from data
    assert hasattr(model, "y_train_mean_")
    assert hasattr(model, "y_train_std_")
    assert model.n_features_rows_in_ == X_train.shape[1]
    assert model.n_features_cols_in_ == X_train.shape[2]
    assert model.device_ is not None
    # Check attributes copied from underlying model
    assert model.executor_ is not None
    assert model.config_ is not None
    print("Fitted attributes checked.")

    # --- Test Predictions ---
    print("\nMaking predictions (mean)...")
    y_pred_mean = model.predict(X_test, output_type="mean")
    assert isinstance(y_pred_mean, np.ndarray)
    assert y_pred_mean.shape == y_test.shape, f"Mean pred shape mismatch: {y_pred_mean.shape} vs {y_test.shape}"
    assert not np.isnan(y_pred_mean).any(), "NaNs found in mean predictions"
    mse_mean = mean_squared_error(y_test, y_pred_mean)
    r2_mean = r2_score(y_test, y_pred_mean)
    print(f"Mean Prediction MSE: {mse_mean:.4f}, R2: {r2_mean:.4f}")

    print("Making predictions (median)...")
    y_pred_median = model.predict(X_test, output_type="median")
    assert isinstance(y_pred_median, np.ndarray)
    assert y_pred_median.shape == y_test.shape, f"Median pred shape mismatch: {y_pred_median.shape} vs {y_test.shape}"
    assert not np.isnan(y_pred_median).any(), "NaNs found in median predictions"
    mse_median = mean_squared_error(y_test, y_pred_median)
    r2_median = r2_score(y_test, y_pred_median)
    print(f"Median Prediction MSE: {mse_median:.4f}, R2: {r2_median:.4f}")

    print("Making predictions (quantiles)...")
    quantiles = [0.2, 0.5, 0.8]
    quantile_preds = model.predict(X_test, output_type="quantiles", quantiles=quantiles)
    assert isinstance(quantile_preds, list)
    assert len(quantile_preds) == len(quantiles)
    for i, q_pred in enumerate(quantile_preds):
        assert isinstance(q_pred, np.ndarray)
        assert q_pred.shape == y_test.shape, f"Quantile {quantiles[i]} pred shape mismatch: {q_pred.shape} vs {y_test.shape}"
        assert not np.isnan(q_pred).any(), f"NaNs found in quantile {quantiles[i]} predictions"
        print(f"  Quantile {quantiles[i]} prediction shape: {q_pred.shape}")
        # Check if 0.5 quantile is close to median prediction
        if quantiles[i] == 0.5:
             # Use a slightly looser tolerance as icdf vs median calculation might differ slightly
             assert np.allclose(q_pred, y_pred_median, atol=1e-3), "0.5 Quantile differs significantly from Median"

    # --- Test Embedding Extraction ---
    print("\nTesting embedding extraction...")
    # Test getting test embeddings
    embeddings = model.get_embeddings(X_test, data_source="test")
    assert isinstance(embeddings, np.ndarray)
    # Shape: (n_estimators, n_samples, embedding_dim)
    # Embedding dim comes from the *underlying* TabPFN model's config
    assert model.tabpfn_regressor_ is not None and model.tabpfn_regressor_.config_ is not None
    expected_embedding_dim = model.tabpfn_regressor_.config_.emsize # Get from fitted underlying model config
    assert embeddings.shape == (model.n_estimators, X_test.shape[0], expected_embedding_dim), \
        f"Embeddings shape mismatch: {embeddings.shape} vs {(model.n_estimators, X_test.shape[0], expected_embedding_dim)}"
    assert not np.isnan(embeddings).any(), "NaNs found in embeddings"
    print(f"Embeddings shape: {embeddings.shape}")

    print("\n--- AgriTabPFNRegressor interface tests completed successfully! ---")

# Keep the main execution block
if __name__ == "__main__":
    test_agri_tabpfn_regressor_interface()
