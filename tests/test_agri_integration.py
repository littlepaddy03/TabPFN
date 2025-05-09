# tests/test_agri_integration.py
# Copyright (c) Prior Labs GmbH 2025.

"""
Integration tests for AgriTabPFNRegressor, focusing on the interaction
between the 3D data handling (preprocessing, encoding) and the underlying
TabPFN model.
"""

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split

from tabpfn.encoders.agri_encoders import AgriDataPreprocessor, AgriDataEncoder
from tabpfn.agri_tabpfn import AgriTabPFNRegressor
from typing import Literal 

# --- Helper Functions ---

def generate_structured_3d_data(
    n_samples: int = 50,
    n_temporal: int = 10,
    n_features: int = 2, # Reduced default features for simpler signal
    static_idx: int = -1,
    temporal_pattern_value: float = 0.0, # Direct value to set/add
    static_pattern_value: float = 0.0,   # Direct value to set/add
    base_feature_value: float = 0.5, # Base value for features
    base_noise_scale: float = 0.01, # Very low noise for X features
    target_noise_scale: float = 0.001, # Very low noise for y
    target_type: Literal["temporal_only", "static_only", "combined"] = "combined"
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates 3D data. Target y can be made to depend primarily on temporal,
    static, or combined features. Patterns are now direct values.
    """
    n_rows = n_temporal + 1
    # Initialize X with a base value plus small noise
    X = np.full((n_samples, n_rows, n_features), base_feature_value)
    X += np.random.randn(n_samples, n_rows, n_features) * base_noise_scale

    resolved_static_idx = static_idx % n_rows
    temporal_indices = [i for i in range(n_rows) if i != resolved_static_idx]

    # Add temporal pattern to the first feature of temporal steps
    if n_temporal > 0 and n_features > 0:
        X[:, temporal_indices, 0] += temporal_pattern_value

    # Add static pattern to the first static feature
    if n_features > 0:
        X[:, resolved_static_idx, 0] += static_pattern_value

    # Calculate components for y based on the *first feature*
    # These components are now directly related to the pattern values
    y_temporal_component = np.zeros(n_samples)
    if n_temporal > 0 and n_features > 0:
        # Mean of the FIRST feature across all temporal steps
        y_temporal_component = np.mean(X[:, temporal_indices, 0], axis=1)

    y_static_component = np.zeros(n_samples)
    if n_features > 0:
        y_static_component = X[:, resolved_static_idx, 0]

    # Construct y based on target_type
    if target_type == "temporal_only":
        y = y_temporal_component
    elif target_type == "static_only":
        y = y_static_component
    elif target_type == "combined":
        y = y_temporal_component + y_static_component # Equal contribution
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    y += np.random.randn(n_samples) * target_noise_scale # Add tiny noise to the final target

    return X, y

# --- Integration Tests ---

def test_encoder_decoder_shapes():
    """Test if the preprocessor and encoder produce expected shapes."""
    n_samples, n_rows_input, n_cols = 20, 8, 5 # n_rows_input includes the static row
    n_temporal_input = n_rows_input - 1
    max_len_processed = 10 # Target temporal length after processing (padding)
    static_idx = -1
    
    X, _ = generate_structured_3d_data(n_samples, n_temporal_input, n_cols, static_idx)

    # Preprocessor
    preprocessor = AgriDataPreprocessor(static_row_index=static_idx, max_temporal_length=max_len_processed)
    X_proc, mask = preprocessor.fit_transform(X)
    
    assert X_proc.shape == (n_samples, max_len_processed + 1, n_cols)
    if max_len_processed > 0:
        assert mask is not None and mask.shape == (n_samples, max_len_processed)
    else:
        assert mask is None or mask.shape == (n_samples, 0)


    # Encoder
    encoded_dim = 64
    temporal_input_dim_for_encoder = n_cols if preprocessor.max_temporal_length_ is not None and preprocessor.max_temporal_length_ > 0 else 0
    static_input_dim_for_encoder = n_cols

    encoder = AgriDataEncoder(
        temporal_input_dim=temporal_input_dim_for_encoder,
        static_input_dim=static_input_dim_for_encoder,
        output_dim=encoded_dim,
        temporal_hidden_dim=32,
        static_hidden_dims=[16]
    )
    X_proc_t = torch.from_numpy(X_proc).float()
    mask_t = torch.from_numpy(mask).bool() if mask is not None and mask.size > 0 else None
    encoder.eval()
    with torch.no_grad():
        X_encoded_t = encoder(X_proc_t, temporal_mask=mask_t, static_row_index=static_idx)
    X_encoded = X_encoded_t.numpy()

    assert X_encoded.shape == (n_samples, encoded_dim), f"Encoded shape mismatch: {X_encoded.shape}"
    assert not np.isnan(X_encoded).any(), "NaNs found in encoded output"

@pytest.mark.parametrize("fusion_strategy", ["concat", "attention", "gated"])
def test_end_to_end_prediction_sensitivity(fusion_strategy: Literal["concat", "attention", "gated"]):
    """Verify that changes in specific input patterns lead to changes in prediction."""
    n_samples, n_temporal, n_features = 120, 15, 2 # Increased samples, more temporal steps
    static_idx = -1
    train_test_split_ratio = 0.5

    # Define signal scales
    base_pattern_val = 0.5
    high_pattern_val = 10.0 # Significantly higher signal value
    v_low_noise_X = 0.001
    v_low_noise_y = 0.0001

    # 1. Generate training data with base patterns
    X_train_base, y_train_base = generate_structured_3d_data(
        n_samples, n_temporal, n_features, static_idx,
        temporal_pattern_value=base_pattern_val,
        static_pattern_value=base_pattern_val,
        base_noise_scale=v_low_noise_X,
        target_noise_scale=v_low_noise_y,
        target_type="combined"
    )

    # 2. Generate test datasets
    # Test Base: same distribution as training
    X_test_base, y_test_base_expected_mean = generate_structured_3d_data(
        n_samples, n_temporal, n_features, static_idx,
        temporal_pattern_value=base_pattern_val, static_pattern_value=base_pattern_val,
        base_noise_scale=v_low_noise_X, target_noise_scale=0, target_type="combined"
    )
    mean_y_base_expected = np.mean(y_test_base_expected_mean)


    # Test High Temporal: X has high temporal signal, y target is based *only* on this high temporal signal
    X_test_high_temp, y_test_high_temp_expected_mean = generate_structured_3d_data(
        n_samples, n_temporal, n_features, static_idx,
        temporal_pattern_value=high_pattern_val, # High temporal X
        static_pattern_value=0.0,               # Base static X (or zero for clarity)
        base_noise_scale=v_low_noise_X, target_noise_scale=0, target_type="temporal_only"
    )
    mean_y_high_temp_expected = np.mean(y_test_high_temp_expected_mean)

    # Test High Static: X has high static signal, y target is based *only* on this high static signal
    X_test_high_static, y_test_high_static_expected_mean = generate_structured_3d_data(
        n_samples, n_temporal, n_features, static_idx,
        temporal_pattern_value=0.0,                # Base temporal X (or zero for clarity)
        static_pattern_value=high_pattern_val,   # High static X
        base_noise_scale=v_low_noise_X, target_noise_scale=0, target_type="static_only"
    )
    mean_y_high_static_expected = np.mean(y_test_high_static_expected_mean)
    
    # Split training data
    X_train_fit, _, y_train_fit, _ = train_test_split(
        X_train_base, y_train_base, test_size=train_test_split_ratio, random_state=42
    )
    if X_train_fit.shape[0] < 20: # Ensure enough training samples after split
        X_train_fit, y_train_fit = X_train_base, y_train_base


    model = AgriTabPFNRegressor(
        n_estimators=16, # Increased estimators
        fusion_strategy=fusion_strategy,
        encoded_embedding_dim=128, # Increased capacity
        temporal_hidden_dim=64,   # Increased capacity
        static_hidden_dims=[32, 16], # Increased capacity
        device='cpu',
        random_state=42,
        ignore_pretraining_limits=True,
        memory_saving_mode=False,
        fit_mode='fit_preprocessors' # Ensure preprocessors are fit
    )
    model.fit(X_train_fit, y_train_fit)

    # Predict
    pred_base = model.predict(X_test_base)
    pred_high_temp = model.predict(X_test_high_temp)
    pred_high_static = model.predict(X_test_high_static)

    mean_pred_base = np.mean(pred_base)
    mean_pred_high_temp = np.mean(pred_high_temp)
    mean_pred_high_static = np.mean(pred_high_static)

    print(f"\nFusion Strategy: {fusion_strategy}")
    print(f"  Mean Predicted Base:         {mean_pred_base:.3f} (Expected y ~{mean_y_base_expected:.3f})")
    print(f"  Mean Predicted High Temporal:{mean_pred_high_temp:.3f} (Expected y ~{mean_y_high_temp_expected:.3f})")
    print(f"  Mean Predicted High Static:  {mean_pred_high_static:.3f} (Expected y ~{mean_y_high_static_expected:.3f})")

    # Differences in predictions
    diff_pred_temp_vs_base = mean_pred_high_temp - mean_pred_base
    diff_pred_static_vs_base = mean_pred_high_static - mean_pred_base

    # Expected differences in y signals (based on how y was constructed for the specific test cases)
    # For X_test_high_temp, its y was (high_pattern_val + base_feature_val)
    # For X_test_base, its y was (base_pattern_val + base_feature_val) + (base_pattern_val + base_feature_val)
    # This comparison is tricky because the model is trained on combined signals.
    # We are checking if the prediction on X_test_high_temp is significantly larger than on X_test_base.
    # The "expected y" for X_test_high_temp is mean_y_high_temp_expected.
    # The "expected y" for X_test_base is mean_y_base_expected.
    # The difference we expect the model to approximate is:
    # mean_y_high_temp_expected - mean_y_base_expected (if model learned perfectly and base static was zero for y_high_temp)
    # A simpler check: the prediction for high signal should be substantially greater than prediction for base signal.

    # The actual y for X_test_high_temp was constructed with ONLY the temporal component active.
    # The actual y for X_test_base was constructed with BOTH components active at base_pattern_val.
    # So, mean_y_high_temp_expected = (base_feature_val + high_pattern_val)
    # And mean_y_base_expected = (base_feature_val + base_pattern_val) + (base_feature_val + base_pattern_val)
    # This makes direct comparison of diff_pred with diff_y_expected complex.

    # Let's simplify: the y for X_test_high_temp should be around high_pattern_val (plus base_feature_val).
    # The y for X_test_base should be around 2 * base_pattern_val (plus 2 * base_feature_val).
    # The model is trained on y_train_base which is ~ 2 * base_pattern_val.
    # When it sees X_test_high_temp, the temporal part is much stronger.

    # We expect pred_high_temp to be substantially larger than pred_base.
    # The difference in the *input signal values* is (high_pattern_val - base_pattern_val).
    # Let's require the prediction difference to be at least a fraction of this input signal difference.
    input_signal_diff = high_pattern_val - base_pattern_val
    min_effect_ratio = 0.3 # Model should capture at least 30% of the direct signal change

    threshold_temp = max(0.1, input_signal_diff * min_effect_ratio)
    threshold_static = max(0.1, input_signal_diff * min_effect_ratio)


    print(f"  Diff Pred Temporal vs Base:   {diff_pred_temp_vs_base:.3f} (Input signal change ~{input_signal_diff:.3f}, Threshold > {threshold_temp:.3f})")
    print(f"  Diff Pred Static vs Base:     {diff_pred_static_vs_base:.3f} (Input signal change ~{input_signal_diff:.3f}, Threshold > {threshold_static:.3f})")

    assert diff_pred_temp_vs_base > threshold_temp, \
        f"Prediction mean did not increase sufficiently for temporal signal ({fusion_strategy}). Diff: {diff_pred_temp_vs_base:.3f}, Expected > {threshold_temp:.3f}"
    
    assert diff_pred_static_vs_base > threshold_static, \
        f"Prediction mean did not increase sufficiently for static signal ({fusion_strategy}). Diff: {diff_pred_static_vs_base:.3f}, Expected > {threshold_static:.3f}"

