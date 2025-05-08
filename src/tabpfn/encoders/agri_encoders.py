"""Specialized encoders for agricultural 3D data.

This module provides encoders for processing 3D agricultural data such as time-series
weather data, NDVI measurements, and static crop and soil information for use with
TabPFN models.
"""

from __future__ import annotations

import logging # Add logging
from typing import Any, Literal, Optional, Tuple, Union, Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tabpfn.model.encoders import (
    InputEncoder,
    normalize_data, # Keep this utility if used
    torch_nanmean,
    torch_nanstd,
)

# Setup logger for this module
logger = logging.getLogger(__name__)


class TimeFeaturesEncoder(nn.Module):
    """Encoder for time-series agricultural features such as weather and NDVI data.

    This encoder handles temporal relationships in the data using a combination of
    1D convolutions and self-attention mechanisms. It can process data with varying
    time periods and handle missing values via masking. # Updated docstring

    Attributes:
        input_dim: Dimensionality of each time step's feature vector.
        hidden_dim: Size of hidden representations.
        output_dim: Dimensionality of the output embeddings.
        kernel_size: Size of the convolutional kernel for local pattern extraction.
        padding: Padding for convolutional layers.
        use_attention: Whether to use self-attention for capturing temporal relationships.
        dropout: Dropout rate for regularization.
        num_heads: Number of attention heads for the multi-head attention layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_attention: bool = True,
        dropout: float = 0.1,
        num_heads: int = 4, # Existing parameter
    ):
        """Initialize the time features encoder.

        Args:
            input_dim: Dimensionality of each time step's feature vector.
            hidden_dim: Size of hidden representations.
            output_dim: Dimensionality of the output embeddings.
            kernel_size: Size of the convolutional kernel for local pattern extraction.
            padding: Padding for convolutional layers.
            use_attention: Whether to use self-attention for capturing temporal relationships.
            dropout: Dropout rate for regularization.
            num_heads: Number of attention heads (must divide hidden_dim evenly).
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
             raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
             raise ValueError(f"output_dim must be positive, got {output_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim # Store original request
        self.output_dim = output_dim
        self.use_attention = use_attention
        self._adjusted_hidden_dim = hidden_dim # Store potentially adjusted dim

        # Ensure hidden_dim is divisible by num_heads for attention
        if use_attention and hidden_dim % num_heads != 0:
            # Ensure integer division
            adjusted_hidden_dim = (hidden_dim // num_heads) * num_heads
            if adjusted_hidden_dim == 0 and hidden_dim > 0:
                # Handle case where hidden_dim is smaller than num_heads by setting dim = num_heads
                adjusted_hidden_dim = num_heads
                logger.warning(
                    f"TimeFeaturesEncoder: hidden_dim ({hidden_dim}) is less than num_heads ({num_heads}). "
                    f"Adjusting hidden_dim to {adjusted_hidden_dim}."
                )
            elif adjusted_hidden_dim != hidden_dim:
                 logger.warning(
                    f"TimeFeaturesEncoder: hidden_dim ({hidden_dim}) is not divisible by num_heads ({num_heads}). "
                    f"Adjusting hidden_dim to {adjusted_hidden_dim}."
                )
            self._adjusted_hidden_dim = adjusted_hidden_dim # Use adjusted dim internally

        # Check if adjusted hidden dim is valid before creating layers
        if self._adjusted_hidden_dim <= 0:
             raise ValueError(f"Calculated _adjusted_hidden_dim ({self._adjusted_hidden_dim}) must be positive.")


        # Temporal feature extraction with 1D convolutions
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self._adjusted_hidden_dim, # Use adjusted dim
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=self._adjusted_hidden_dim, # Use adjusted dim
            out_channels=self._adjusted_hidden_dim, # Use adjusted dim
            kernel_size=kernel_size,
            padding=padding,
        )

        # Optional self-attention layer
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=self._adjusted_hidden_dim, # Use adjusted dim
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.attention = None # Explicitly set to None if not used

        # Final projection layer
        self.fc_out = nn.Linear(self._adjusted_hidden_dim, output_dim) # Project from adjusted dim

        self.dropout = nn.Dropout(dropout)
        # Layer norms use the adjusted hidden dimension
        self.layer_norm_conv = nn.LayerNorm(self._adjusted_hidden_dim)
        if use_attention:
            self.layer_norm_attn = nn.LayerNorm(self._adjusted_hidden_dim)


    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the time features encoder.

        Args:
            x: Input tensor of shape (batch_size, time_steps, features).
            mask: Optional mask tensor (batch_size, time_steps),
                  True for valid steps, False for padding. # Clarified mask meaning

        Returns:
            Encoded time features of shape (batch_size, output_dim).
        """
        if x.shape[1] == 0: # Handle empty sequence case
             return torch.zeros(x.shape[0], self.output_dim, device=x.device, dtype=x.dtype)

        # Handle potential NaNs - replace with zero before conv/attn
        # Masking should handle their contribution later
        x = torch.nan_to_num(x, nan=0.0)

        # Transpose for 1D convolution: (batch_size, features, time_steps)
        x = x.permute(0, 2, 1)

        # Apply convolutions with activation
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))

        # Transpose back: (batch_size, time_steps, hidden_dim)
        x = x.permute(0, 2, 1)

        # Apply layer normalization after convolutions
        x = self.layer_norm_conv(x)

        # Apply self-attention if enabled
        if self.use_attention and self.attention is not None:
            # Create attention mask from input mask (True where attention should be prevented)
            # MultiheadAttention expects key_padding_mask where True indicates a padded/masked position
            attn_mask = ~mask if mask is not None else None # Invert mask: True means ignore

            # Apply multi-head attention
            attn_output, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
            x = x + self.dropout(attn_output)  # Residual connection
            x = self.layer_norm_attn(x)  # Normalization after attention + residual

        # Global pooling: average only over valid time steps using the mask
        if mask is not None:
            # Expand mask to match tensor dimensions for broadcasting: (B, T, 1)
            mask_expanded = mask.unsqueeze(-1) #.expand_as(x) # No need to expand if using multiplication
            # Zero out padded steps before summing
            x_masked = x * mask_expanded # Broadcasting handles expansion
            # Sum valid steps and divide by the count of valid steps
            # Add epsilon to avoid division by zero if a sample has no valid steps
            valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1e-6) # Shape (B, 1)
            x_mean = x_masked.sum(dim=1) / valid_counts # Sum shape (B, H), divide by (B, 1) -> (B, H)
        else:
            # If no mask, simply take the mean over the time dimension
            x_mean = torch.mean(x, dim=1)

        # Final projection
        output = self.fc_out(x_mean)

        return output


class StaticFeaturesEncoder(nn.Module):
    """Encoder for static agricultural features such as soil properties and crop information.

    This encoder handles non-temporal features using a multi-layer perceptron architecture
    with normalization and dropout for regularization.

    Attributes:
        input_dim: Dimensionality of the static feature vector.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Dimensionality of the output embeddings.
        dropout: Dropout rate for regularization.
        normalize_features: Whether to normalize input features (primarily done externally now).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        normalize_features: bool = True, # Keep flag, maybe for internal LayerNorms
    ) -> None:
        """Initialize the static features encoder.

        Args:
            input_dim: Dimensionality of the static feature vector.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Dimensionality of the output embeddings.
            dropout: Dropout rate for regularization.
            normalize_features: Whether to normalize input features (primarily done externally).
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
             raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not hidden_dims: # Check if list is empty
             logger.warning("StaticFeaturesEncoder initialized with no hidden layers.")


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_features = normalize_features # Store flag

        # Build MLP layers
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                 raise ValueError(f"Hidden dimensions must be positive, got {hidden_dim}")
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Add LayerNorm for stability within the MLP
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the static features encoder.

        Args:
            x: Input tensor of shape (batch_size, features).

        Returns:
            Encoded static features of shape (batch_size, output_dim).
        """
        # Handle potential NaNs - replace with zero before MLP
        # Main imputation should happen in Preprocessor
        x = torch.nan_to_num(x, nan=0.0)

        # Normalization is now primarily done in AgriDataPreprocessor.
        # The self.normalize_features flag might control internal LayerNorms if added,
        # or could be removed if LayerNorm is always used.

        # Apply MLP
        output = self.mlp(x)

        return output


class AgriDataPreprocessor:
    """Preprocessor for 3D agricultural data.

    Handles standardization of input structure, missing values, feature scaling,
    and padding for variable-length time series. Separates temporal and static
    features based on configuration.
    """

    def __init__(
        self,
        *, # Make arguments keyword-only
        # Column names are optional, primarily for potential future DataFrame input
        temporal_features_cols: Optional[List[str]] = None,
        static_features_cols: Optional[List[str]] = None,
        # Configuration for identifying static/temporal rows
        static_row_index: int = -1, # Index of the row containing static features
        # Configuration for padding/truncation
        max_temporal_length: Optional[int] = None, # Max length for temporal padding
        # Configuration for processing
        standardize_features: bool = True,
        handle_missing_values: Literal["mean", "zero", "none"] = "mean",
    ) -> None:
        """Initialize the agricultural data preprocessor.

        Args:
            temporal_features_cols: Optional list of temporal feature column names.
            static_features_cols: Optional list of static feature column names.
            static_row_index: Index of the row holding static features. Defaults to -1 (last row).
            max_temporal_length: Maximum length to pad/truncate temporal sequences to.
                                 If None, determined from data during fit.
            standardize_features: Whether to standardize features (temporal and static separately)
                                  to zero mean and unit variance.
            handle_missing_values: Strategy for handling missing values ('mean', 'zero', 'none').
        """
        # Store column names if provided
        self.temporal_features_cols = temporal_features_cols
        self.static_features_cols = static_features_cols

        # Store configuration
        self.static_row_index = static_row_index
        self.max_temporal_length = max_temporal_length # User-provided setting
        self.standardize_features = standardize_features
        self.handle_missing_values = handle_missing_values

        # --- Fitted parameters (initialized to None) ---
        # Statistics for standardization/imputation
        self.temporal_means_: Optional[np.ndarray] = None
        self.temporal_stds_: Optional[np.ndarray] = None
        self.static_means_: Optional[np.ndarray] = None
        self.static_stds_: Optional[np.ndarray] = None
        # Determined properties of the fitted data
        self.max_temporal_length_: Optional[int] = None # Final length used (determined or user-provided)
        self.n_features_cols_: Optional[int] = None # Number of feature columns
        self.is_fitted_ = False

    def _split_temporal_static(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Splits the 3D input array into temporal and static parts based on static_row_index."""
        if X.ndim != 3:
             raise ValueError(f"_split_temporal_static expects 3D input, got shape {X.shape}")
        n_rows = X.shape[1]
        # Resolve negative index (e.g., -1 becomes n_rows - 1)
        resolved_static_index = self.static_row_index % n_rows

        if not (0 <= resolved_static_index < n_rows):
             raise ValueError(f"static_row_index {self.static_row_index} (resolved to {resolved_static_index}) "
                              f"is out of bounds for input with {n_rows} rows.")

        static_data = X[:, resolved_static_index, :]
        temporal_indices = [i for i in range(n_rows) if i != resolved_static_index]
        # Handle case where there might be no temporal rows if n_rows=1
        temporal_data = X[:, temporal_indices, :] if temporal_indices else np.zeros((X.shape[0], 0, X.shape[2]), dtype=X.dtype)


        return temporal_data, static_data


    def fit(self, X: np.ndarray) -> 'AgriDataPreprocessor':
        """Fit the preprocessor to the data.

        Calculates statistics for standardization and imputation, and determines
        the maximum temporal sequence length if not provided.

        Args:
            X: Input data of shape (n_samples, n_features_rows, n_features_cols).

        Returns:
            Self: The fitted preprocessor instance.
        """
        if not isinstance(X, np.ndarray):
             raise TypeError(f"Input X must be a NumPy array, got {type(X)}")
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input data (samples, rows, cols), got shape {X.shape}")

        self.n_features_cols_ = X.shape[2]
        if self.n_features_cols_ == 0:
             logger.warning("Fitting AgriDataPreprocessor with 0 feature columns.")

        temporal_data, static_data = self._split_temporal_static(X)
        actual_temporal_length = temporal_data.shape[1]

        # Determine final max_temporal_length_
        if self.max_temporal_length is None:
            self.max_temporal_length_ = actual_temporal_length
            logger.info(f"max_temporal_length not provided, determined max length from data: {self.max_temporal_length_}")
        else:
            self.max_temporal_length_ = self.max_temporal_length
            if actual_temporal_length < self.max_temporal_length_:
                logger.warning(
                    f"User-provided max_temporal_length ({self.max_temporal_length_}) is greater "
                    f"than the maximum length found in the training data ({actual_temporal_length}). "
                    f"Padding will occur during transform."
                )
            elif actual_temporal_length > self.max_temporal_length_:
                logger.warning(
                    f"User-provided max_temporal_length ({self.max_temporal_length_}) is smaller "
                    f"than the maximum length found in the training data ({actual_temporal_length}). "
                    f"Data will be truncated during transform."
                )

        # Calculate statistics if needed
        if self.standardize_features or self.handle_missing_values == "mean":
            # Calculate temporal stats (only if temporal data exists)
            if actual_temporal_length > 0 and self.n_features_cols_ > 0:
                self.temporal_means_ = np.nanmean(temporal_data, axis=(0, 1), keepdims=True)
                self.temporal_stds_ = np.nanstd(temporal_data, axis=(0, 1), keepdims=True)
                # Replace zero stds with 1.0
                self.temporal_stds_ = np.where(self.temporal_stds_ == 0, 1.0, self.temporal_stds_)
                # Handle potential all-NaN features
                if np.isnan(self.temporal_means_).any():
                    logger.warning("Found NaN in temporal means during fit (likely all-NaN feature). Replacing with 0.")
                    self.temporal_means_ = np.nan_to_num(self.temporal_means_, nan=0.0)
                if np.isnan(self.temporal_stds_).any():
                    logger.warning("Found NaN in temporal stds during fit (likely all-NaN feature). Replacing with 1.")
                    self.temporal_stds_ = np.nan_to_num(self.temporal_stds_, nan=1.0)
                    self.temporal_stds_ = np.where(self.temporal_stds_ == 0, 1.0, self.temporal_stds_) # Re-check after nan_to_num
            else:
                 # Set to appropriate shapes with default values if no temporal data/features
                 feat_shape = (1, 1, self.n_features_cols_)
                 self.temporal_means_ = np.zeros(feat_shape)
                 self.temporal_stds_ = np.ones(feat_shape)


            # Calculate static stats (only if features exist)
            if self.n_features_cols_ > 0:
                self.static_means_ = np.nanmean(static_data, axis=0, keepdims=True)
                self.static_stds_ = np.nanstd(static_data, axis=0, keepdims=True)
                # Replace zero stds with 1.0
                self.static_stds_ = np.where(self.static_stds_ == 0, 1.0, self.static_stds_)
                # Handle potential all-NaN features
                if np.isnan(self.static_means_).any():
                    logger.warning("Found NaN in static means during fit (likely all-NaN feature). Replacing with 0.")
                    self.static_means_ = np.nan_to_num(self.static_means_, nan=0.0)
                if np.isnan(self.static_stds_).any():
                    logger.warning("Found NaN in static stds during fit (likely all-NaN feature). Replacing with 1.")
                    self.static_stds_ = np.nan_to_num(self.static_stds_, nan=1.0)
                    self.static_stds_ = np.where(self.static_stds_ == 0, 1.0, self.static_stds_) # Re-check
            else:
                 # Set to appropriate shapes with default values if no features
                 feat_shape = (1, self.n_features_cols_)
                 self.static_means_ = np.zeros(feat_shape)
                 self.static_stds_ = np.ones(feat_shape)


        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform the data using fitted parameters.

        Applies imputation, standardization, padding/truncation, and generates a temporal mask.

        Args:
            X: Input data of shape (n_samples, n_features_rows, n_features_cols).

        Returns:
            A tuple containing:
            - Transformed data (np.ndarray): Shape (n_samples, max_temporal_length_ + 1, n_features_cols).
            - Temporal mask (Optional[np.ndarray]): Shape (n_samples, max_temporal_length_).
                                                    True for valid data, False for padding. None if no temporal data.
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        if not isinstance(X, np.ndarray):
             raise TypeError(f"Input X must be a NumPy array, got {type(X)}")
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input data (samples, rows, cols), got shape {X.shape}")
        if X.shape[2] != self.n_features_cols_:
             raise ValueError(f"Input data has {X.shape[2]} feature columns, "
                              f"but preprocessor was fitted with {self.n_features_cols_} columns.")

        # Create a copy to avoid modifying the original array
        X_transformed = X.copy()

        temporal_data, static_data = self._split_temporal_static(X_transformed)
        n_samples = X.shape[0]
        original_temporal_length = temporal_data.shape[1]

        # --- Handle Missing Values ---
        if self.handle_missing_values == "mean":
            if self.temporal_means_ is None or self.static_means_ is None:
                raise RuntimeError("Cannot use 'mean' imputation if preprocessor was not fitted or stats are missing.")
            # Impute temporal data (only if it exists)
            if original_temporal_length > 0:
                temporal_nan_mask = np.isnan(temporal_data)
                # Use np.broadcast_to for safe broadcasting
                temporal_fill_values = np.broadcast_to(self.temporal_means_, temporal_data.shape)
                temporal_data[temporal_nan_mask] = temporal_fill_values[temporal_nan_mask]
            # Impute static data
            static_nan_mask = np.isnan(static_data)
            static_fill_values = np.broadcast_to(self.static_means_, static_data.shape)
            static_data[static_nan_mask] = static_fill_values[static_nan_mask]

        elif self.handle_missing_values == "zero":
            if original_temporal_length > 0:
                temporal_data = np.nan_to_num(temporal_data, nan=0.0)
            static_data = np.nan_to_num(static_data, nan=0.0)
        elif self.handle_missing_values != "none":
            raise ValueError(f"Unknown handle_missing_values strategy: {self.handle_missing_values}")
        # If 'none', NaNs remain

        # --- Standardize Features ---
        if self.standardize_features:
            if self.temporal_means_ is None or self.temporal_stds_ is None or \
               self.static_means_ is None or self.static_stds_ is None:
                raise RuntimeError("Cannot standardize features if preprocessor was not fitted or stats are missing.")
            # Standardize temporal data (only if it exists)
            if original_temporal_length > 0:
                temporal_data = (temporal_data - self.temporal_means_) / self.temporal_stds_
            # Standardize static data
            static_data = (static_data - self.static_means_) / self.static_stds_

        # --- Padding / Truncation and Mask Generation ---
        temporal_mask = None
        # Ensure max_temporal_length_ is not None before proceeding
        if self.max_temporal_length_ is None:
             raise RuntimeError("max_temporal_length_ not set during fit. Cannot transform.")

        target_length = self.max_temporal_length_

        # Handle case with no temporal data
        if original_temporal_length == 0:
             if target_length > 0:
                 # Need to create empty temporal data and full padding mask
                 temporal_data = np.zeros((n_samples, target_length, self.n_features_cols_), dtype=X.dtype)
                 temporal_mask = np.zeros((n_samples, target_length), dtype=bool)
             else:
                 # No temporal data expected and none present
                 temporal_data = np.zeros((n_samples, 0, self.n_features_cols_), dtype=X.dtype)
                 temporal_mask = None # Or np.zeros((n_samples, 0), dtype=bool)? None seems cleaner.
        # Handle cases with temporal data
        elif original_temporal_length < target_length:
            # Pad
            padding_shape = (n_samples, target_length - original_temporal_length, self.n_features_cols_)
            padding = np.zeros(padding_shape, dtype=temporal_data.dtype) # Pad with zeros
            temporal_data = np.concatenate([temporal_data, padding], axis=1)
            # Create mask: True for original data, False for padding
            temporal_mask = np.zeros((n_samples, target_length), dtype=bool)
            temporal_mask[:, :original_temporal_length] = True
        elif original_temporal_length > target_length:
            # Truncate
            temporal_data = temporal_data[:, :target_length, :]
            # Create mask: all True up to the truncated length
            temporal_mask = np.ones((n_samples, target_length), dtype=bool)
        else: # original_temporal_length == target_length
            # No padding/truncation needed, mask is all True
            temporal_mask = np.ones((n_samples, target_length), dtype=bool)


        # --- Reassemble the data ---
        # Final shape includes the determined temporal length + 1 static row
        final_temporal_length = temporal_data.shape[1]
        final_rows = final_temporal_length + 1
        if self.n_features_cols_ is None:
             raise RuntimeError("n_features_cols_ not set during fit.")

        X_final = np.zeros((n_samples, final_rows, self.n_features_cols_), dtype=X.dtype)

        # Place temporal data and static data correctly
        resolved_static_index_final = self.static_row_index % final_rows
        temporal_indices_final = [i for i in range(final_rows) if i != resolved_static_index_final]

        if final_temporal_length > 0: # Only place if temporal data exists
             X_final[:, temporal_indices_final, :] = temporal_data
        X_final[:, resolved_static_index_final, :] = static_data

        # Final safety check for NaNs introduced by standardization/imputation edge cases
        if (self.standardize_features or self.handle_missing_values == 'mean') and np.isnan(X_final).any():
            logger.warning("NaNs detected after final assembly. Replacing with 0.")
            X_final = np.nan_to_num(X_final, nan=0.0)

        # Return None mask if final_temporal_length is 0
        if final_temporal_length == 0:
            temporal_mask = None

        return X_final, temporal_mask

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit the preprocessor and transform the data.

        Args:
            X: Input data of shape (n_samples, n_features_rows, n_features_cols).

        Returns:
            A tuple containing:
            - Transformed data (np.ndarray).
            - Temporal mask (Optional[np.ndarray]).
        """
        return self.fit(X).transform(X)


class AgriDataEncoder(InputEncoder):
    """Main encoder for 3D agricultural data combining temporal and static features.

    Processes preprocessed 3D data (including temporal mask) using specialized
    encoders for temporal and static parts, then fuses them according to the
    specified strategy.
    """

    def __init__(
        self,
        temporal_input_dim: int,
        static_input_dim: int,
        output_dim: int, # This is the final output dimension after fusion
        temporal_hidden_dim: int = 128,
        static_hidden_dims: Optional[List[int]] = None, # Default handled below
        fusion_strategy: Literal["concat", "attention", "gated"] = "gated",
        temporal_encoder_kwargs: Optional[Dict[str, Any]] = None,
        static_encoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the agricultural data encoder.

        Args:
            temporal_input_dim: Dimensionality of features at each time step.
            static_input_dim: Dimensionality of static features.
            output_dim: Dimensionality of the final combined embedding.
            temporal_hidden_dim: Size of hidden temporal representations. Defaults to 128.
            static_hidden_dims: List of hidden layer dimensions for static features. Defaults to [64].
            fusion_strategy: Strategy for combining embeddings ('concat', 'attention', 'gated'). Defaults to 'gated'.
            temporal_encoder_kwargs: Additional keyword arguments for TimeFeaturesEncoder.
            static_encoder_kwargs: Additional keyword arguments for StaticFeaturesEncoder.
        """
        super().__init__()

        # Set defaults
        if static_hidden_dims is None:
            static_hidden_dims = [64]
        temporal_encoder_kwargs = temporal_encoder_kwargs or {}
        static_encoder_kwargs = static_encoder_kwargs or {}

        # --- Determine intermediate dimensions based on fusion strategy ---
        self.output_dim = output_dim # Store the final desired output dimension
        fusion_input_dim = output_dim # Default for attention/gated

        if fusion_strategy == "concat":
             # For concat, each encoder outputs half the final dimension
             if output_dim % 2 != 0:
                  # Adjust output_dim to be even if necessary
                  adjusted_output_dim = output_dim + 1
                  logger.warning(f"output_dim ({output_dim}) is odd for 'concat' fusion. "
                                 f"Adjusting final output_dim to {adjusted_output_dim}.")
                  self.output_dim = adjusted_output_dim # Update the final output dim
             # Each encoder produces half of the (potentially adjusted) output dim
             fusion_input_dim = self.output_dim // 2
        # For 'attention' and 'gated', each encoder produces embeddings of size 'output_dim'
        # which are then fused and potentially projected back to 'output_dim'.
        # So, fusion_input_dim should match the desired output_dim for these strategies.
        elif fusion_strategy in ["attention", "gated"]:
             fusion_input_dim = self.output_dim


        # --- Initialize Encoders ---
        # Ensure dimensions are positive before creating encoders
        if temporal_input_dim <= 0 and static_input_dim <= 0:
             raise ValueError("Both temporal_input_dim and static_input_dim are non-positive.")

        # Only initialize temporal encoder if there are temporal dimensions
        self.temporal_encoder = None
        if temporal_input_dim > 0:
            self.temporal_encoder = TimeFeaturesEncoder(
                input_dim=temporal_input_dim,
                hidden_dim=temporal_hidden_dim,
                output_dim=fusion_input_dim, # Temporal encoder outputs this size
                **temporal_encoder_kwargs,
            )
        else:
             logger.info("temporal_input_dim is 0, not creating TimeFeaturesEncoder.")


        # Only initialize static encoder if there are static dimensions
        self.static_encoder = None
        if static_input_dim > 0:
            self.static_encoder = StaticFeaturesEncoder(
                input_dim=static_input_dim,
                hidden_dims=static_hidden_dims,
                output_dim=fusion_input_dim, # Static encoder outputs this size
                **static_encoder_kwargs,
            )
        else:
             logger.info("static_input_dim is 0, not creating StaticFeaturesEncoder.")


        self.fusion_strategy = fusion_strategy

        # --- Initialize Fusion Mechanism ---
        if fusion_strategy == "concat":
            # Input to fusion layer is concatenation of both outputs
            self.fusion_layer = nn.Linear(fusion_input_dim * 2, self.output_dim)
        elif fusion_strategy == "attention":
            # Attention operates on embeddings of size fusion_input_dim
            if fusion_input_dim <= 0:
                 raise ValueError(f"fusion_input_dim ({fusion_input_dim}) must be positive for attention.")
            num_heads = temporal_encoder_kwargs.get('num_heads', 4) # Reuse heads or default
            if fusion_input_dim % num_heads != 0:
                num_heads = 1
                logger.warning(f"Fusion attention: fusion_input_dim {fusion_input_dim} not divisible by "
                               f"num_heads {temporal_encoder_kwargs.get('num_heads', 4)}. Using 1 head.")
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=fusion_input_dim, num_heads=num_heads, dropout=0.1, batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(fusion_input_dim)
            # Projection is only needed if fusion_input_dim differs from final output_dim (shouldn't for attn/gated)
            self.fusion_proj = nn.Identity() # Output of attention fusion is already correct size

        elif fusion_strategy == "gated":
            # Gate input is concatenation, output is fusion_input_dim
            self.gate = nn.Linear(fusion_input_dim * 2, fusion_input_dim)
            # Projection is identity as fused output is already correct size
            self.fusion_proj = nn.Identity()
        else:
            raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        static_row_index: int = -1,
        single_eval_pos: Optional[int] = None # Keep for compatibility
        ) -> torch.Tensor:
        """Forward pass through the agricultural data encoder.

        Args:
            x: Input tensor (preprocessed) of shape (batch_size, n_rows, n_features_cols).
            temporal_mask: Boolean mask for temporal data (batch_size, n_temporal_steps), True for valid.
            static_row_index: Index used to extract static features.
            single_eval_pos: Position for single evaluation (kept for compatibility).

        Returns:
            Encoded agricultural data of shape (batch_size, output_dim).
        """
        n_rows = x.shape[1]
        resolved_static_index = static_row_index % n_rows

        # Extract static and temporal data
        static_data = x[:, resolved_static_index, :]
        temporal_indices = [i for i in range(n_rows) if i != resolved_static_index]
        temporal_data = x[:, temporal_indices, :] if temporal_indices else torch.zeros(x.shape[0], 0, x.shape[2], device=x.device, dtype=x.dtype)

        # --- Process Temporal Data (if encoder exists) ---
        if self.temporal_encoder is not None:
            if temporal_data.shape[1] == 0: # No temporal steps
                 # Need a zero tensor of the correct embedding size
                 temporal_embedding = torch.zeros(x.shape[0], self.temporal_encoder.output_dim, device=x.device, dtype=x.dtype)
            else:
                 # Ensure mask matches temporal data shape if provided
                 if temporal_mask is not None and temporal_mask.shape[1] != temporal_data.shape[1]:
                      raise ValueError(f"Temporal mask shape {temporal_mask.shape} incompatible with "
                                       f"temporal data shape {temporal_data.shape}")
                 temporal_embedding = self.temporal_encoder(temporal_data, mask=temporal_mask)
        else:
            # If no temporal encoder, use zeros (or handle differently if needed)
            # This case implies fusion_input_dim might need adjustment based on strategy
            if self.fusion_strategy == 'concat':
                 fusion_input_dim = self.output_dim // 2
            else:
                 fusion_input_dim = self.output_dim
            temporal_embedding = torch.zeros(x.shape[0], fusion_input_dim, device=x.device, dtype=x.dtype)


        # --- Process Static Data (if encoder exists) ---
        if self.static_encoder is not None:
            static_embedding = self.static_encoder(static_data)
        else:
            # If no static encoder, use zeros
            if self.fusion_strategy == 'concat':
                 fusion_input_dim = self.output_dim // 2
            else:
                 fusion_input_dim = self.output_dim
            static_embedding = torch.zeros(x.shape[0], fusion_input_dim, device=x.device, dtype=x.dtype)


        # --- Fuse Embeddings ---
        if self.fusion_strategy == "concat":
            # Concatenate along the feature dimension
            combined = torch.cat([temporal_embedding, static_embedding], dim=1)
            output = self.fusion_layer(combined)

        elif self.fusion_strategy == "attention":
            # Stack embeddings to form a sequence [temporal, static]
            seq = torch.stack([temporal_embedding, static_embedding], dim=1) # Shape (B, 2, fusion_input_dim)
            # Apply attention
            attn_output, _ = self.fusion_attention(seq, seq, seq)
            # Add & Norm, then pool (mean over the sequence dim)
            seq = self.fusion_norm(seq + attn_output)
            output = self.fusion_proj(torch.mean(seq, dim=1)) # Shape (B, output_dim)

        elif self.fusion_strategy == "gated":
            # Compute gate based on concatenation
            combined = torch.cat([temporal_embedding, static_embedding], dim=1)
            gate_value = torch.sigmoid(self.gate(combined)) # Shape (B, fusion_input_dim)
            # Apply gate: weighted sum of temporal and static embeddings
            fused = gate_value * temporal_embedding + (1 - gate_value) * static_embedding
            output = self.fusion_proj(fused) # Shape (B, output_dim)

        return output
