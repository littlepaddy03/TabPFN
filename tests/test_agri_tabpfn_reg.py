"""Test script for AgriTabPFNRegressor.

This script tests the functionality of the AgriTabPFNRegressor by creating synthetic
agricultural data and verifying that the model can process and make predictions on it.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn.agri_tabpfn import AgriTabPFNRegressor
from tabpfn.encoders.agri_encoders import AgriDataPreprocessor


def create_synthetic_agri_data(n_samples=200, n_time_steps=30, n_features=5, missing_ratio=0.1):
    """
    Creates synthetic agricultural data with temporal and static features.
    X shape: (n_samples, n_time_steps + 1, n_features)
    The last time step X[:, -1, :] is reserved for static features.
    y shape: (n_samples,)
    """
    # Generate random temporal features
    X_temporal = np.random.rand(n_samples, n_time_steps, n_features)

    # Generate random static features (conceptually, could be soil type, farm ID, etc.)
    # For simplicity, we'll just use one "time step" slot for them
    X_static = np.random.rand(n_samples, 1, n_features)

    # Combine into a single X array
    # The last "time step" (index n_time_steps) will hold static features
    X = np.concatenate((X_temporal, X_static), axis=1)
    assert X.shape == (n_samples, n_time_steps + 1, n_features)


    # Define importance of features for yield prediction
    temporal_importance = np.random.rand(n_features)
    static_importance = np.random.rand(n_features)

    # Simulate a "critical period" for temporal features
    critical_period_start = n_time_steps // 3
    critical_period_end = 2 * n_time_steps // 3

    # Introduce missing values (NaNs)
    # Do not introduce NaNs into the static features part for simplicity in this example
    # if we want them to directly contribute without nan-handling in their specific logic
    temporal_mask = np.random.random(X_temporal.shape) < missing_ratio
    X_temporal[temporal_mask] = np.nan
    X[:, :n_time_steps, :] = X_temporal # Put back the temporal part with NaNs

    # --- Calculate yield (y) based on features ---

    # 1. Contribution from temporal features (average over critical period)
    critical_period_data = X[:, critical_period_start:critical_period_end, :]
    critical_period_avg = np.nanmean(critical_period_data, axis=1) # Shape: (n_samples, n_features)
    # Handle cases where a sample might have all NaNs in the critical period for a feature
    critical_period_avg = np.nan_to_num(critical_period_avg, nan=0.0) # Replace NaN averages with 0

    temporal_component = np.sum(critical_period_avg * temporal_importance, axis=1) # Shape: (n_samples,)

    # 2. Contribution from static features
    # X[:, -1, :] accesses the static features, shape (n_samples, n_features)
    static_features_slice = X[:, -1, :]
    # If static features themselves could be NaN (not in this version, but for robustness):
    # static_features_slice = np.nan_to_num(static_features_slice, nan=0.0)
    
    # Use np.nansum for static_component if X_static could have NaNs that are multiplied
    # In the current setup, X_static doesn't have NaNs introduced directly,
    # but if missing_ratio applied to all X, this would be crucial.
    # For safety and general applicability if X_static could contain NaNs:
    static_component = np.nansum(static_features_slice * static_importance, axis=1) # Shape: (n_samples,)

    # 3. Interaction term (example: critical period average interacts with a static feature)
    # Use np.nansum for interaction if X_static could have NaNs
    interaction = np.nansum(critical_period_avg * static_features_slice, axis=1) * 0.2 # Shape: (n_samples,)
    
    # Add some noise
    noise = np.random.normal(0, 0.5, n_samples)

    y = temporal_component + static_component + interaction + noise

    # Ensure no NaNs in the final y
    if np.isnan(y).any():
        print("Debug: NaNs found in y. Components:")
        print(f"  temporal_component NaNs: {np.isnan(temporal_component).any()} (sum: {np.nansum(temporal_component)})")
        print(f"  static_component NaNs: {np.isnan(static_component).any()} (sum: {np.nansum(static_component)})")
        print(f"  interaction NaNs: {np.isnan(interaction).any()} (sum: {np.nansum(interaction)})")
        print(f"  noise NaNs: {np.isnan(noise).any()} (sum: {np.nansum(noise)})")
        # Further debug: check inputs to problematic components
        if np.isnan(static_component).any():
            print(f"  static_features_slice NaNs: {np.isnan(static_features_slice).any()}")
            problematic_static_mult = static_features_slice * static_importance
            print(f"  static_features_slice * static_importance NaNs: {np.isnan(problematic_static_mult).any()}")
        if np.isnan(interaction).any():
             problematic_interaction_mult = critical_period_avg * static_features_slice
             print(f"  critical_period_avg * static_features_slice NaNs: {np.isnan(problematic_interaction_mult).any()}")


    assert not np.isnan(y).any(), "Generated y contains NaN values!"

    return X, y

def test_agri_tabpfn_regressor():
    """Test the AgriTabPFNRegressor with synthetic data."""
    print("Creating synthetic agricultural data...")
    X, y = create_synthetic_agri_data(n_samples=200, n_time_steps=30, n_features=5)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    # Initialize the model
    print("\nInitializing AgriTabPFNRegressor...")
    model = AgriTabPFNRegressor(
        n_estimators=2,  # Use fewer estimators for faster testing
        temporal_hidden_dim=64,
        static_hidden_dims=[32],
        fusion_strategy="gated",
        random_state=42,
    )
    
    # Fit the model
    print("Fitting the model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nEvaluation results:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Test quantile predictions
    print("\nTesting quantile predictions...")
    quantiles = [0.1, 0.5, 0.9]
    quantile_preds = model.predict(X_test, output_type="quantiles", quantiles=quantiles)
    
    print(f"Shape of quantile predictions:")
    for i, q in enumerate(quantiles):
        print(f"  Quantile {q}: {quantile_preds[i].shape}")
    
    # Test embedding extraction
    print("\nTesting embedding extraction...")
    embeddings = model.get_embeddings(X_test)
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    test_agri_tabpfn_regressor()
