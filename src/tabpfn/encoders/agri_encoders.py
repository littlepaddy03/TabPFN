"""Specialized encoders for agricultural 3D data.

This module provides encoders for processing 3D agricultural data such as time-series
weather data, NDVI measurements, and static crop and soil information for use with
TabPFN models.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tabpfn.model.encoders import (
    InputEncoder,
    normalize_data,
    torch_nanmean,
)


class TimeFeaturesEncoder(nn.Module):
    """Encoder for time-series agricultural features such as weather and NDVI data.

    This encoder handles temporal relationships in the data using a combination of
    1D convolutions and self-attention mechanisms. It can process data with varying
    time periods and handle missing values.

    Attributes:
        input_dim: Dimensionality of each time step's feature vector
        hidden_dim: Size of hidden representations
        output_dim: Dimensionality of the output embeddings
        kernel_size: Size of the convolutional kernel for local pattern extraction
        padding: Padding for convolutional layers
        use_attention: Whether to use self-attention for capturing temporal relationships
        dropout: Dropout rate for regularization
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
    ) -> None:
        """Initialize the time features encoder.

        Args:
            input_dim: Dimensionality of each time step's feature vector
            hidden_dim: Size of hidden representations
            output_dim: Dimensionality of the output embeddings
            kernel_size: Size of the convolutional kernel for local pattern extraction
            padding: Padding for convolutional layers
            use_attention: Whether to use self-attention for capturing temporal relationships
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention

        # Temporal feature extraction with 1D convolutions
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

        # Optional self-attention layer for capturing long-range dependencies
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )

        # Final projection layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the time features encoder.

        Args:
            x: Input tensor of shape (batch_size, time_steps, features)
            mask: Optional mask tensor for handling missing values

        Returns:
            Encoded time features of shape (batch_size, output_dim)
        """
        # Handle missing values
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # Transpose for 1D convolution: (batch_size, features, time_steps)
        x = x.permute(0, 2, 1)

        # Apply convolutions with activation and normalization
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))

        # Transpose back: (batch_size, time_steps, hidden_dim)
        x = x.permute(0, 2, 1)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Apply self-attention if enabled
        if self.use_attention:
            # Create attention mask from input mask if provided
            attn_mask = None
            if mask is not None:
                attn_mask = ~mask  # Invert mask for attention (True values are masked)

            # Apply multi-head attention
            attn_output, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
            x = x + self.dropout(attn_output)  # Residual connection
            x = self.layer_norm(x)  # Another normalization after attention

        # Global pooling to get fixed-size representation
        # We compute means along the time dimension, handling potential NaNs
        x_mean = torch_nanmean(x, axis=1)

        # Final projection
        return self.fc_out(x_mean)


class StaticFeaturesEncoder(nn.Module):
    """Encoder for static agricultural features such as soil properties and crop information.

    This encoder handles non-temporal features using a multi-layer perceptron architecture
    with normalization and dropout for regularization.

    Attributes:
        input_dim: Dimensionality of the static feature vector
        hidden_dims: List of hidden layer dimensions
        output_dim: Dimensionality of the output embeddings
        dropout: Dropout rate for regularization
        normalize_features: Whether to normalize input features
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
        normalize_features: bool = True,
    ) -> None:
        """Initialize the static features encoder.

        Args:
            input_dim: Dimensionality of the static feature vector
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimensionality of the output embeddings
            dropout: Dropout rate for regularization
            normalize_features: Whether to normalize input features
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_features = normalize_features

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the static features encoder.

        Args:
            x: Input tensor of shape (batch_size, features)

        Returns:
            Encoded static features of shape (batch_size, output_dim)
        """
        # Optional feature normalization
        if self.normalize_features:
            # Normalize each feature to have zero mean and unit variance
            # Handle potential NaN values
            x = normalize_data(x, clip=True)

        # Apply MLP
        return self.mlp(x)


class AgriDataEncoder(InputEncoder):
    """Main encoder for 3D agricultural data combining temporal and static features.

    This encoder handles the full 3D structure of agricultural data by processing
    temporal features (weather, NDVI) and static features (soil, crop info) separately
    and then combining them into a unified representation compatible with TabPFN.

    Attributes:
        temporal_input_dim: Dimensionality of features at each time step
        temporal_hidden_dim: Size of hidden temporal representations
        static_input_dim: Dimensionality of static features
        static_hidden_dims: List of hidden layer dimensions for static features
        output_dim: Dimensionality of the final combined embedding
        fusion_strategy: Strategy for combining temporal and static representations
    """

    def __init__(
        self,
        temporal_input_dim: int,
        temporal_hidden_dim: int,
        static_input_dim: int,
        static_hidden_dims: list[int],
        output_dim: int,
        fusion_strategy: Literal["concat", "attention", "gated"] = "gated",
        temporal_encoder_kwargs: dict[str, Any] | None = None,
        static_encoder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the agricultural data encoder.

        Args:
            temporal_input_dim: Dimensionality of features at each time step
            temporal_hidden_dim: Size of hidden temporal representations
            static_input_dim: Dimensionality of static features
            static_hidden_dims: List of hidden layer dimensions for static features
            output_dim: Dimensionality of the final combined embedding
            fusion_strategy: Strategy for combining temporal and static representations
                - "concat": Simple concatenation followed by projection
                - "attention": Use attention mechanism for fusion
                - "gated": Gated fusion mechanism with learnable parameters
            temporal_encoder_kwargs: Additional arguments for temporal encoder
            static_encoder_kwargs: Additional arguments for static encoder
        """
        super().__init__()

        # Initialize default kwargs if not provided
        if temporal_encoder_kwargs is None:
            temporal_encoder_kwargs = {}
        if static_encoder_kwargs is None:
            static_encoder_kwargs = {}

        # Calculate intermediate dimensions for fusion
        if fusion_strategy == "concat":
            # For concatenation, we'll need a projection layer
            fusion_input_dim = output_dim // 2
        else:
            # For attention or gated fusion, both embeddings should match output_dim
            fusion_input_dim = output_dim

        # Initialize encoders
        self.temporal_encoder = TimeFeaturesEncoder(
            input_dim=temporal_input_dim,
            hidden_dim=temporal_hidden_dim,
            output_dim=fusion_input_dim,
            **temporal_encoder_kwargs,
        )

        self.static_encoder = StaticFeaturesEncoder(
            input_dim=static_input_dim,
            hidden_dims=static_hidden_dims,
            output_dim=fusion_input_dim,
            **static_encoder_kwargs,
        )

        self.fusion_strategy = fusion_strategy

        # Initialize fusion mechanism based on strategy
        if fusion_strategy == "concat":
            self.fusion_layer = nn.Linear(fusion_input_dim * 2, output_dim)
        elif fusion_strategy == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=fusion_input_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True,
            )
            self.fusion_norm = nn.LayerNorm(fusion_input_dim)
            self.fusion_proj = nn.Linear(fusion_input_dim, output_dim)
        elif fusion_strategy == "gated":
            self.gate = nn.Linear(fusion_input_dim * 2, fusion_input_dim)
            self.fusion_proj = nn.Linear(fusion_input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported fusion strategy: {fusion_strategy}")

    def forward(
        self, x: torch.Tensor, single_eval_pos: int | None = None
    ) -> torch.Tensor:
        """Forward pass through the agricultural data encoder.

        Args:
            x: Input tensor of shape (batch_size, n_features_rows, n_features_cols)
                where n_features_rows typically represents time steps
                and n_features_cols represents different variables
            single_eval_pos: Position for single evaluation (kept for compatibility with TabPFN)

        Returns:
            Encoded agricultural data of shape (batch_size, output_dim)
        """
        # Split the input into temporal and static parts
        # Assuming the first n-1 rows are temporal and the last row is static
        # This is a simplification and should be adapted based on actual data structure
        temporal_data = x[:, :-1, :]  # All but the last row
        static_data = x[:, -1, :]  # Just the last row

        # Create mask for temporal data (identify missing values)
        temporal_mask = ~torch.isnan(temporal_data).any(dim=-1)

        # Process temporal and static data through respective encoders
        temporal_embedding = self.temporal_encoder(temporal_data, mask=temporal_mask)
        static_embedding = self.static_encoder(static_data)

        # Combine embeddings based on fusion strategy
        if self.fusion_strategy == "concat":
            # Simple concatenation followed by projection
            combined = torch.cat([temporal_embedding, static_embedding], dim=1)
            output = self.fusion_layer(combined)

        elif self.fusion_strategy == "attention":
            # Use attention mechanism for fusion
            # Reshape for attention (add sequence dimension)
            temp_emb = temporal_embedding.unsqueeze(1)  # [batch, 1, dim]
            stat_emb = static_embedding.unsqueeze(1)  # [batch, 1, dim]

            # Concatenate to form sequence
            seq = torch.cat([temp_emb, stat_emb], dim=1)  # [batch, 2, dim]

            # Self-attention over the sequence
            attn_output, _ = self.fusion_attention(seq, seq, seq)

            # Residual connection and normalization
            seq = seq + attn_output
            seq = self.fusion_norm(seq)

            # Mean pooling and projection
            output = self.fusion_proj(torch.mean(seq, dim=1))

        elif self.fusion_strategy == "gated":
            # Gated fusion mechanism
            combined = torch.cat([temporal_embedding, static_embedding], dim=1)
            gate_value = torch.sigmoid(self.gate(combined))

            # Apply gate to combine embeddings
            fused = (
                gate_value * temporal_embedding + (1 - gate_value) * static_embedding
            )
            output = self.fusion_proj(fused)

        return output


class AgriDataPreprocessor:
    """Preprocessor for 3D agricultural data.

    This class handles the preprocessing of 3D agricultural data into a format
    that can be used by the AgriDataEncoder. It standardizes features, handles
    missing values, and prepares the data for the encoder.

    Attributes:
        temporal_features: List of temporal feature names
        static_features: List of static feature names
        standardize_features: Whether to standardize features
        handle_missing_values: Strategy for handling missing values
    """

    def __init__(
        self,
        temporal_features: list[str],
        static_features: list[str],
        standardize_features: bool = True,
        handle_missing_values: Literal["mean", "zero", "none"] = "mean",
    ) -> None:
        """Initialize the agricultural data preprocessor.

        Args:
            temporal_features: List of temporal feature names
            static_features: List of static feature names
            standardize_features: Whether to standardize features
            handle_missing_values: Strategy for handling missing values
                - "mean": Replace missing values with mean
                - "zero": Replace missing values with zero
                - "none": Leave missing values as NaN
        """
        self.temporal_features = temporal_features
        self.static_features = static_features
        self.standardize_features = standardize_features
        self.handle_missing_values = handle_missing_values

        # Placeholders for fitted parameters
        self.temporal_means = None
        self.temporal_stds = None
        self.static_means = None
        self.static_stds = None
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> AgriDataPreprocessor:
        """Fit the preprocessor to the data.

        Args:
            X: Input data of shape (n_samples, n_features_rows, n_features_cols)

        Returns:
            Self for method chaining
        """
        # Extract temporal and static parts
        temporal_data = X[:, :-1, :]  # All but the last row
        static_data = X[:, -1, :]  # Just the last row

        if self.standardize_features:
            # Compute means and stds for temporal features, handling NaNs
            self.temporal_means = np.nanmean(temporal_data, axis=(0, 1), keepdims=True)
            self.temporal_stds = np.nanstd(temporal_data, axis=(0, 1), keepdims=True)
            self.temporal_stds[self.temporal_stds == 0] = 1.0  # Avoid division by zero

            # Compute means and stds for static features, handling NaNs
            self.static_means = np.nanmean(static_data, axis=0, keepdims=True)
            self.static_stds = np.nanstd(static_data, axis=0, keepdims=True)
            self.static_stds[self.static_stds == 0] = 1.0  # Avoid division by zero

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data.

        Args:
            X: Input data of shape (n_samples, n_features_rows, n_features_cols)

        Returns:
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Create a copy to avoid modifying the original
        X_transformed = X.copy()

        # Extract temporal and static parts
        temporal_data = X_transformed[:, :-1, :]
        static_data = X_transformed[:, -1, :]

        # Handle missing values
        if self.handle_missing_values == "mean":
            # Replace NaNs with means
            temporal_mask = np.isnan(temporal_data)
            if temporal_mask.any():
                temporal_data[temporal_mask] = np.take(
                    self.temporal_means.reshape(-1),
                    np.nonzero(temporal_mask.reshape(temporal_mask.shape[0], -1))[-1]
                    % self.temporal_means.size,
                )

            static_mask = np.isnan(static_data)
            if static_mask.any():
                static_data[static_mask] = np.take(
                    self.static_means.reshape(-1),
                    np.nonzero(static_mask)[-1] % self.static_means.size,
                )

        elif self.handle_missing_values == "zero":
            # Replace NaNs with zeros
            temporal_data = np.nan_to_num(temporal_data, nan=0.0)
            static_data = np.nan_to_num(static_data, nan=0.0)

        # Standardize features if enabled
        if self.standardize_features:
            temporal_data = (temporal_data - self.temporal_means) / self.temporal_stds
            static_data = (static_data - self.static_means) / self.static_stds

        # Reassemble the transformed data
        X_transformed[:, :-1, :] = temporal_data
        X_transformed[:, -1, :] = static_data

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform the data.

        Args:
            X: Input data of shape (n_samples, n_features_rows, n_features_cols)

        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
