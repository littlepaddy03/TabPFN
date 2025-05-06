"""Integration interface between 3D agricultural data and TabPFN.

This module provides adapter functions to connect the agricultural data encoders
with the TabPFN model architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tabpfn.encoders.agri_encoders import AgriDataEncoder, AgriDataPreprocessor

if TYPE_CHECKING:
    import numpy as np


def prepare_agri_data_for_tabpfn(
    X: np.ndarray,
    y: np.ndarray,
    temporal_features: list[str] | None = None,
    static_features: list[str] | None = None,
    preprocessor: AgriDataPreprocessor = None,
) -> tuple[AgriDataPreprocessor, AgriDataEncoder, np.ndarray, np.ndarray]:
    """Prepare 3D agricultural data for use with TabPFN.

    This function handles the preprocessing and encoding setup for 3D agricultural
    data before passing it to the TabPFN model.

    Args:
        X: Input data of shape (n_samples, n_features_rows, n_features_cols)
        y: Target values of shape (n_samples,)
        temporal_features: List of temporal feature names
        static_features: List of static feature names
        preprocessor: Optional preprocessor to use

    Returns:
        A tuple containing:
        - Fitted preprocessor
        - Configured encoder
        - Preprocessed data
        - Target values
    """
    # Validate input data
    if X.ndim != 3:
        raise ValueError(f"Expected 3D input data, got shape {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"Expected 1D target data, got shape {y.shape}")

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples. Got {len(X)} and {len(y)}"
        )

    # Initialize lists of feature names if not provided
    if temporal_features is None:
        temporal_features = [f"temp_{i}" for i in range(X.shape[2])]

    if static_features is None:
        static_features = [f"static_{i}" for i in range(X.shape[2])]

    # Initialize and fit preprocessor if not provided
    if preprocessor is None:
        preprocessor = AgriDataPreprocessor(
            temporal_features=temporal_features,
            static_features=static_features,
        )

    X_preprocessed = preprocessor.fit_transform(X)

    # Configure encoder dimensions based on data
    temporal_input_dim = X.shape[2]  # Number of features per time step
    static_input_dim = X.shape[2]  # Number of static features

    # Create appropriately sized encoder
    encoder = AgriDataEncoder(
        temporal_input_dim=temporal_input_dim,
        temporal_hidden_dim=min(128, temporal_input_dim * 2),  # Scale with input size
        static_input_dim=static_input_dim,
        static_hidden_dims=[min(64, static_input_dim * 2)],  # Scale with input size
        output_dim=128,  # Match TabPFN's default embedding size
        fusion_strategy="gated",  # Use gated fusion by default
    )

    return preprocessor, encoder, X_preprocessed, y


def encode_agri_batch_for_tabpfn(
    X: np.ndarray,
    encoder: AgriDataEncoder,
    device: torch.device = None,
) -> torch.Tensor:
    """Encode a batch of 3D agricultural data for TabPFN.

    Args:
        X: Input data of shape (n_samples, n_features_rows, n_features_cols)
        encoder: The agricultural data encoder
        device: The device to use for encoding

    Returns:
        Encoded tensor compatible with TabPFN
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensor and move to device
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    # Get encoder output
    with torch.no_grad():
        encoder = encoder.to(device)
        return encoder(X_tensor)
