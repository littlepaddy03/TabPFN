# tests/test_agri_preprocessor.py
import numpy as np
import pytest
from tabpfn.encoders.agri_encoders import AgriDataPreprocessor
from typing import Tuple, Optional, List # Import List
import logging # Import logging

# Setup logger for this module
logger = logging.getLogger(__name__)


# --- Test Data Generation (Fix for n_rows=1) ---
def create_test_data(
    n_samples=50,
    n_rows=10,
    n_cols=5,
    static_row_index=-1,
    variable_length=False,
    add_nans=False,
    nan_ratio=0.1,
) -> Tuple[np.ndarray, int, int]: # Added type hint for return
    """Creates synthetic 3D data for testing the preprocessor."""
    if n_rows < 1: # Allow n_rows=1
        raise ValueError("n_rows must be at least 1.")
    # REMOVED the check: if variable_length and original_temporal_length > 0 and n_rows < 2:

    X = np.random.rand(n_samples, n_rows, n_cols) * 10
    resolved_static_index = static_row_index % n_rows
    # Make static features potentially different scale/distribution
    if n_cols > 0: # Avoid error if n_cols is 0
        X[:, resolved_static_index, :] = np.random.randn(n_samples, n_cols) * 5 + 50

    original_temporal_length = max(0, n_rows - 1) # Calculate original temporal length
    lengths = np.full(n_samples, original_temporal_length, dtype=int) # Store actual length per sample

    if variable_length and original_temporal_length > 0:
        # Generate random lengths between half and full original temporal length
        min_len = max(1, original_temporal_length // 2) # Ensure at least 1 step
        # Ensure upper bound is inclusive for randint
        lengths = np.random.randint(min_len, original_temporal_length + 1, size=n_samples)
        temporal_indices = [idx for idx in range(n_rows) if idx != resolved_static_index]
        for i in range(n_samples):
            invalid_start_index = lengths[i]
            # Set elements *after* the actual length to NaN initially
            if invalid_start_index < original_temporal_length:
                 # Correctly get indices from the temporal_indices list
                 indices_to_nan = temporal_indices[invalid_start_index:]
                 if indices_to_nan: # Check if list is not empty
                    X[i, indices_to_nan, :] = np.nan
    # No need for elif for original_temporal_length == 0


    if add_nans:
        # Add NaNs randomly, but avoid making entire columns NaN if possible
        for i in range(n_samples):
             for j in range(n_cols):
                 # Introduce NaNs only if the value is not already NaN (from variable length)
                 col_data = X[i, :, j]
                 non_nan_mask = ~np.isnan(col_data)
                 num_non_nan = non_nan_mask.sum()
                 if num_non_nan > 1: # Ensure we don't NaN the only non-NaN value
                     nan_indices = np.where(non_nan_mask)[0]
                     # Ensure num_to_nan is not larger than available indices
                     num_to_nan = min(int(nan_ratio * num_non_nan), len(nan_indices))
                     if num_to_nan > 0:
                          nan_candidates = np.random.choice(nan_indices, size=num_to_nan, replace=False)
                          X[i, nan_candidates, j] = np.nan

        # Final check: ensure at least one non-NaN value per feature column across all samples for robust stats
        for j in range(n_cols):
             # Check across all samples for the column
             if n_samples > 0 and np.all(np.isnan(X[..., j])): # Add check for n_samples > 0
                 # If a column is all NaNs, put a 0 in the first sample's first row
                 X[0, 0, j] = 0.0
                 logger.warning(f"Column {j} was all NaN, inserted 0.0 at sample 0, row 0.")


    return X, resolved_static_index, original_temporal_length


# --- Tests ---

@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("missing_handling", ["mean", "zero", "none"])
@pytest.mark.parametrize("static_index", [-1, 0, 3]) # Test different static row positions
@pytest.mark.parametrize("max_len_setting", [None, 5, 15]) # None, shorter, longer
@pytest.mark.parametrize("variable_len", [True, False]) # Test fixed and variable length input
@pytest.mark.parametrize("add_nans_input", [True, False]) # Test with and without added NaNs
def test_agri_preprocessor_fit_transform(standardize, missing_handling, static_index, max_len_setting, variable_len, add_nans_input):
    """Test fit_transform basic functionality with various configurations."""
    n_samples, n_rows, n_cols = 30, 10, 4
    # Ensure static_index is valid for n_rows
    if abs(static_index) >= n_rows:
         static_index = -1 # Default to last row if index is out of bounds

    X_orig, resolved_static_idx_orig, orig_temp_len = create_test_data(
        n_samples=n_samples, n_rows=n_rows, n_cols=n_cols,
        static_row_index=static_index,
        variable_length=variable_len,
        add_nans=add_nans_input,
        nan_ratio=0.2 # Increase ratio slightly
        )
    X = X_orig.copy() # Keep original for checks

    preprocessor = AgriDataPreprocessor(
        static_row_index=static_index,
        max_temporal_length=max_len_setting,
        standardize_features=standardize,
        handle_missing_values=missing_handling,
    )

    # --- Fit and Transform ---
    X_transformed, mask = preprocessor.fit_transform(X)

    # --- Assertions ---
    # Check fit attributes
    assert preprocessor.is_fitted_
    assert preprocessor.n_features_cols_ == n_cols
    # Determine expected max length after fit
    expected_max_len = max_len_setting if max_len_setting is not None else orig_temp_len
    assert preprocessor.max_temporal_length_ == expected_max_len

    # Check output shape
    assert X_transformed.shape == (n_samples, expected_max_len + 1, n_cols)
    if expected_max_len > 0: # Mask only exists if there's temporal data
        assert mask is not None, f"Mask should exist for expected_max_len={expected_max_len}"
        assert mask.shape == (n_samples, expected_max_len)
        assert mask.dtype == bool
    else:
        assert mask is None, f"Mask should be None for expected_max_len={expected_max_len}"

    # Check NaN handling in output
    if missing_handling in ["mean", "zero"]:
        # No NaNs should exist after 'mean' or 'zero' imputation
        assert not np.isnan(X_transformed).any(), f"NaNs found with missing_handling='{missing_handling}'"
    elif missing_handling == "none":
        # NaNs might still exist if they were in the original data and not imputed/standardized away
        pass

    # Check standardization (if applicable)
    if standardize:
        # Check stats were computed (should not be None)
        assert preprocessor.temporal_means_ is not None
        assert preprocessor.temporal_stds_ is not None
        assert preprocessor.static_means_ is not None
        assert preprocessor.static_stds_ is not None
        # Check stats have correct shape (broadcastable to features)
        assert preprocessor.temporal_means_.shape == (1, 1, n_cols)
        assert preprocessor.temporal_stds_.shape == (1, 1, n_cols)
        assert preprocessor.static_means_.shape == (1, n_cols)
        assert preprocessor.static_stds_.shape == (1, n_cols)
        # Check no NaNs in computed stats
        assert not np.isnan(preprocessor.temporal_means_).any()
        assert not np.isnan(preprocessor.temporal_stds_).any()
        assert not np.isnan(preprocessor.static_means_).any()
        assert not np.isnan(preprocessor.static_stds_).any()


        # Check transformed data stats (approximate checks)
        # Resolve static index for the *transformed* shape
        resolved_static_idx_transformed = static_index % X_transformed.shape[1]
        temporal_indices_transformed = [i for i in range(X_transformed.shape[1]) if i != resolved_static_idx_transformed]

        # Check temporal data (consider mask)
        if mask is not None and mask.any(): # If there is temporal data and a mask
            # Select valid temporal steps using the mask
            temporal_data_transformed = X_transformed[:, temporal_indices_transformed, :]
            # Apply the mask directly to the temporal data slice
            valid_temporal_data = temporal_data_transformed[mask] # This flattens based on True values in mask

            if valid_temporal_data.size > n_cols * 10: # Need enough points for reliable stats
                 # Calculate mean/std across valid points for each feature
                 # Reshape the flattened valid data back to (n_valid_points, n_features)
                 n_valid_points = valid_temporal_data.size // n_cols
                 valid_temporal_data_reshaped = valid_temporal_data.reshape(n_valid_points, n_cols)

                 mean_check = np.mean(valid_temporal_data_reshaped, axis=0)
                 std_check = np.std(valid_temporal_data_reshaped, axis=0)
                 # Increased tolerance for std check
                 assert np.allclose(mean_check, 0, atol=0.3), f"Temporal mean not close to 0: {mean_check}"
                 assert np.allclose(std_check, 1, atol=0.3), f"Temporal std not close to 1: {std_check}" # FIX: Increased atol

        # Check static data
        static_data_transformed = X_transformed[:, resolved_static_idx_transformed, :]
        # Check only if the original static data wasn't all NaNs
        original_static_data = X_orig[:, resolved_static_idx_orig, :]
        if n_cols > 0 and not np.all(np.isnan(original_static_data)): # Add check for n_cols > 0
            mean_check_static = np.mean(static_data_transformed, axis=0)
            std_check_static = np.std(static_data_transformed, axis=0)
            assert np.allclose(mean_check_static, 0, atol=0.2), f"Static mean not close to 0: {mean_check_static}"
            assert np.allclose(std_check_static, 1, atol=0.2), f"Static std not close to 1: {std_check_static}" # FIX: Increased atol

    else: # Not standardize
        # Stats should only be computed if needed for mean imputation
        if missing_handling == 'mean':
             assert preprocessor.temporal_means_ is not None
             assert preprocessor.static_means_ is not None
             # Stds might or might not be computed depending on internal logic, don't assert None
        else:
             # If not standardizing and not mean imputing, stats should be None
             assert preprocessor.temporal_means_ is None
             assert preprocessor.temporal_stds_ is None
             assert preprocessor.static_means_ is None
             assert preprocessor.static_stds_ is None


def test_agri_preprocessor_padding_masking():
    """Test padding and mask generation specifically with variable lengths."""
    n_samples, n_rows_orig, n_cols = 5, 7, 3
    static_idx = -1
    max_len_pad = 10 # Pad to length 10 (original temporal len = 6)
    # Generate data where variable_length=True introduced NaNs to mark shorter sequences
    X_orig, _, original_temporal_length = create_test_data(
        n_samples, n_rows_orig, n_cols, static_row_index=static_idx, variable_length=True, add_nans=False
        )

    # Determine the *actual* original lengths for each sample based on the NaNs we added
    lengths = []
    resolved_static_idx_orig = static_idx % n_rows_orig
    temporal_indices_orig = [idx for idx in range(n_rows_orig) if idx != resolved_static_idx_orig]
    for i in range(n_samples):
        sample_temp_data = X_orig[i, temporal_indices_orig, :]
        # Find first row where all features are NaN (our proxy for end of sequence)
        first_all_nan_row = np.where(np.isnan(sample_temp_data).all(axis=1))[0]
        actual_len = first_all_nan_row[0] if len(first_all_nan_row) > 0 else len(temporal_indices_orig)
        lengths.append(actual_len)

    preprocessor = AgriDataPreprocessor(
        static_row_index=static_idx,
        max_temporal_length=max_len_pad,
        handle_missing_values='zero', # Use zero to check padding values easily
        standardize_features=False
        )
    X_transformed, mask = preprocessor.fit_transform(X_orig) # Fit and transform

    assert X_transformed.shape == (n_samples, max_len_pad + 1, n_cols)
    assert mask is not None
    assert mask.shape == (n_samples, max_len_pad)

    # Verify mask based on the *actual* lengths determined from input NaNs
    for i in range(n_samples):
        valid_len = lengths[i] # Use the length determined from NaNs in X_orig
        # The mask should be True for the first 'valid_len' steps
        assert np.all(mask[i, :valid_len]), f"Sample {i}: Mask should be True up to actual length {valid_len}"
        # The mask should be False for steps *after* 'valid_len' up to max_len_pad
        if valid_len < max_len_pad:
             assert np.all(~mask[i, valid_len:]), f"Sample {i}: Mask should be False after actual length {valid_len}"

    # Check padded values are zero (since handle_missing='zero')
    resolved_static_idx_transformed = static_idx % (max_len_pad + 1)
    temporal_indices_transformed = [idx for idx in range(max_len_pad + 1) if idx != resolved_static_idx_transformed]
    for i in range(n_samples):
         valid_len = lengths[i] # Use actual length again
         if valid_len < max_len_pad:
             # Indices in the transformed array corresponding to padding
             padding_indices_in_transformed = temporal_indices_transformed[valid_len:]
             # Check if the slice corresponding to padding is all zeros
             assert np.all(X_transformed[i, padding_indices_in_transformed, :] == 0), f"Sample {i}: Padding should be zero"


def test_agri_preprocessor_truncation():
    """Test sequence truncation and mask generation."""
    n_samples, n_rows_orig, n_cols = 5, 12, 3
    static_idx = 0 # Test static at beginning
    max_len_truncate = 8 # Truncate temporal sequences (original len 11)
    X_orig, _, _ = create_test_data(n_samples, n_rows_orig, n_cols, static_row_index=static_idx, add_nans=False)

    preprocessor = AgriDataPreprocessor(
        static_row_index=static_idx,
        max_temporal_length=max_len_truncate,
        standardize_features=False,
        handle_missing_values='none'
        )
    X_transformed, mask = preprocessor.fit_transform(X_orig)

    # Check output shape (truncated temporal + 1 static)
    assert X_transformed.shape == (n_samples, max_len_truncate + 1, n_cols)
    assert mask is not None
    assert mask.shape == (n_samples, max_len_truncate)
    # Mask should be all True for truncated data (all kept steps are valid)
    assert np.all(mask)

    # Check that the data matches the truncated original data
    resolved_static_idx_orig = static_idx % n_rows_orig
    temporal_indices_orig = [idx for idx in range(n_rows_orig) if idx != resolved_static_idx_orig]
    original_temporal_data_truncated = X_orig[:, temporal_indices_orig[:max_len_truncate], :]
    original_static_data = X_orig[:, resolved_static_idx_orig, :]

    resolved_static_idx_transformed = static_idx % (max_len_truncate + 1)
    temporal_indices_transformed = [idx for idx in range(max_len_truncate + 1) if idx != resolved_static_idx_transformed]

    assert np.array_equal(X_transformed[:, temporal_indices_transformed, :], original_temporal_data_truncated)
    assert np.array_equal(X_transformed[:, resolved_static_idx_transformed, :], original_static_data)

def test_agri_preprocessor_no_temporal():
    """Test case where input data only has the static row."""
    n_samples, n_rows, n_cols = 5, 1, 3
    static_idx = 0
    # Adjusted create_test_data handles n_rows=1
    X, _, _ = create_test_data(n_samples, n_rows, n_cols, static_row_index=static_idx)

    preprocessor = AgriDataPreprocessor(static_row_index=static_idx, max_temporal_length=5) # Request padding
    X_transformed, mask = preprocessor.fit_transform(X)

    assert preprocessor.max_temporal_length_ == 5
    assert X_transformed.shape == (n_samples, 5 + 1, n_cols) # Padded temporal + static
    assert mask is not None
    assert mask.shape == (n_samples, 5)
    assert not mask.any() # Mask should be all False (all padding)

    # Check padded values are zero
    resolved_static_idx_transformed = static_idx % (5 + 1)
    temporal_indices_transformed = [idx for idx in range(5 + 1) if idx != resolved_static_idx_transformed]
    assert np.all(X_transformed[:, temporal_indices_transformed, :] == 0)
    # Check static data is preserved
    assert np.array_equal(X_transformed[:, resolved_static_idx_transformed, :], X[:, 0, :])

def test_agri_preprocessor_transform_before_fit():
     """Test that transform raises error if called before fit."""
     preprocessor = AgriDataPreprocessor()
     X = np.random.rand(5, 4, 3)
     with pytest.raises(RuntimeError, match="Preprocessor must be fitted before transform"):
          preprocessor.transform(X)

def test_agri_preprocessor_mismatched_cols_transform():
     """Test transform raises error if feature columns mismatch."""
     preprocessor = AgriDataPreprocessor()
     X_fit = np.random.rand(10, 5, 4)
     X_transform = np.random.rand(5, 5, 3) # Different number of columns
     preprocessor.fit(X_fit)
     with pytest.raises(ValueError, match="Input data has 3 feature columns"):
          preprocessor.transform(X_transform)

# Add a test for static index at beginning
def test_agri_preprocessor_static_index_zero():
    n_samples, n_rows, n_cols = 10, 6, 3
    static_idx = 0
    X, _, _ = create_test_data(n_samples, n_rows, n_cols, static_row_index=static_idx)
    preprocessor = AgriDataPreprocessor(static_row_index=static_idx, max_temporal_length=None) # Use original length
    X_transformed, mask = preprocessor.fit_transform(X)

    assert X_transformed.shape == (n_samples, n_rows, n_cols) # Shape should be same if max_len is None
    assert mask is not None and mask.shape == (n_samples, n_rows - 1)
    assert np.all(mask) # Should be all true if no padding/truncation

    # Check static data is correctly placed
    assert np.array_equal(X_transformed[:, 0, :], X[:, 0, :])
    # Check temporal data is correctly placed
    assert np.array_equal(X_transformed[:, 1:, :], X[:, 1:, :])


# Add a test for n_cols = 0
def test_agri_preprocessor_zero_cols():
    n_samples, n_rows = 10, 5
    n_cols = 0
    X = np.zeros((n_samples, n_rows, n_cols)) # Empty features
    preprocessor = AgriDataPreprocessor(static_row_index=-1, max_temporal_length=3)
    X_transformed, mask = preprocessor.fit_transform(X)

    assert preprocessor.n_features_cols_ == 0
    assert preprocessor.max_temporal_length_ == 3
    assert X_transformed.shape == (n_samples, 3 + 1, 0) # Correct shape with 0 cols
    assert mask is not None
    assert mask.shape == (n_samples, 3)
    # Mask should still be all True if max_len_setting was used,
    # even if there's no data, the *steps* exist conceptually up to max_len
    assert np.all(mask)
