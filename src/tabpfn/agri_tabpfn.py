"""TabPFN interface for 3D agricultural data.

This module provides a specialized TabPFN interface for agricultural data, which
typically has a 3D structure with time series weather data, NDVI measurements,
and static soil and crop information.
"""

from __future__ import annotations

import logging # Import logging
import typing
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, List, Dict, Tuple
from typing_extensions import Self, overload

import numpy as np
import torch
from sklearn import config_context
from sklearn.base import BaseEstimator, RegressorMixin, check_is_fitted

from tabpfn.base import (
    check_cpu_warning,
    determine_precision,
)
from tabpfn.config import ModelInterfaceConfig
from tabpfn.constants import XType, YType # Keep these type aliases if used
from tabpfn.encoders.agri_encoders import (
    AgriDataEncoder,
    AgriDataPreprocessor,
)
# Removed agri_interface imports
from tabpfn.regressor import TabPFNRegressor
from tabpfn.utils import (
    infer_device_and_type,
    infer_random_state,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch.types import _dtype
    from tabpfn.inference import InferenceEngine
    from tabpfn.model.config import InferenceConfig
    # Import underlying model type if needed for type hints
    from tabpfn.model.transformer import PerFeatureTransformer
    # Import bar distribution type if needed
    from tabpfn.model.bar_distribution import FullSupportBarDistribution


# Setup logger for this module
logger = logging.getLogger(__name__)

class AgriTabPFNRegressor(RegressorMixin, BaseEstimator):
    """TabPFN regressor adapted for 3D agricultural data.

    Handles 3D input (samples, rows, columns) by preprocessing and encoding
    it into a 2D format suitable for the underlying TabPFN model. Assumes
    temporal features occupy initial rows and static features are in a
    designated row (default: last).

    Example usage:
        ```python
        import numpy as np
        from tabpfn.agri_tabpfn import AgriTabPFNRegressor

        # Synthetic 3D data: 10 samples, 5 time steps + 1 static row, 4 features
        X_3d = np.random.rand(10, 6, 4)
        y = np.random.rand(10) * 10

        model = AgriTabPFNRegressor(device='cpu') # Specify device if needed
        model.fit(X_3d, y)
        predictions = model.predict(X_3d)
        print(predictions)
        ```
    """

    # --- Attributes set during __init__ ---
    # Agri-specific config
    temporal_hidden_dim: int
    static_hidden_dims: List[int]
    fusion_strategy: Literal["concat", "attention", "gated"]
    encoded_embedding_dim: int
    static_row_index: int
    max_temporal_length: Optional[int]
    standardize_agri_features: bool
    handle_missing_agri_values: Literal["mean", "zero", "none"]
    agri_encoder_kwargs: Dict[str, Any]
    # Underlying TabPFN config (passed down)
    n_estimators: int
    softmax_temperature: float
    average_before_softmax: bool
    model_path: Union[str, Path, Literal["auto"]]
    device: Union[str, torch.device, Literal["auto"]]
    ignore_pretraining_limits: bool
    inference_precision: Union[_dtype, Literal["autocast", "auto"]]
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"]
    memory_saving_mode: Union[bool, Literal["auto"], float, int]
    random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]]
    n_jobs: int
    inference_config: Optional[Union[Dict, ModelInterfaceConfig]]

    # --- Attributes set during fit ---
    # Configuration objects from underlying TabPFN
    config_: Optional[InferenceConfig] = None
    interface_config_: Optional[ModelInterfaceConfig] = None
    # Underlying inference engine
    executor_: Optional[InferenceEngine] = None

    # Device and precision attributes determined during fit
    device_: Optional[torch.device] = None
    use_autocast_: bool = False
    forced_inference_dtype_: Optional[_dtype] = None

    # Input shape attributes determined during fit
    n_features_rows_in_: int = 0
    n_features_cols_in_: int = 0
    n_outputs_: int = 1 # TabPFN currently supports single output regression

    # Fitted components specific to AgriTabPFN
    agri_preprocessor_: Optional[AgriDataPreprocessor] = None
    agri_encoder_: Optional[AgriDataEncoder] = None
    # Underlying TabPFN model instance (needed for predict)
    tabpfn_regressor_: Optional[TabPFNRegressor] = None
    # Add type hint for underlying model if possible
    model_: Optional["PerFeatureTransformer"] = None
    # Add type hints for underlying criterion components if possible
    bardist_: Optional["FullSupportBarDistribution"] = None
    renormalized_criterion_: Optional["FullSupportBarDistribution"] = None


    # Target scaling attributes determined during fit
    y_train_mean_: float = 0.0
    y_train_std_: float = 1.0

    # Sklearn compatibility attribute
    _is_fitted: bool = False


    def __init__(  # noqa: PLR0913
        self,
        *,
        # --- Agri-specific parameters ---
        temporal_hidden_dim: int = 128,
        static_hidden_dims: Optional[List[int]] = None, # Default handled below
        fusion_strategy: Literal["concat", "attention", "gated"] = "gated",
        encoded_embedding_dim: int = 128, # Matches default emsize of TabPFN
        static_row_index: int = -1,
        max_temporal_length: Optional[int] = None,
        standardize_agri_features: bool = True,
        handle_missing_agri_values: Literal["mean", "zero", "none"] = "mean",
        agri_encoder_kwargs: Optional[Dict[str, Any]] = None,

        # --- Standard TabPFN parameters passed down ---
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = False,
        model_path: str | Path | Literal["auto"] = "auto",
        device: str | torch.device | Literal["auto"] = "auto",
        ignore_pretraining_limits: bool = False,
        inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
        fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
        ] = "fit_preprocessors",
        memory_saving_mode: bool | Literal["auto"] | float | int = "auto",
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
        n_jobs: int = -1,
        inference_config: Optional[Union[Dict, ModelInterfaceConfig]] = None,
    ) -> None:
        """Initialize the agricultural TabPFN regressor.

        Args:
            temporal_hidden_dim: Size of hidden temporal representations in the encoder.
            static_hidden_dims: List of hidden layer dimensions for static features. Defaults to [64].
            fusion_strategy: Strategy for combining embeddings ('concat', 'attention', 'gated').
            encoded_embedding_dim: Dimensionality of the 2D features after encoding, fed into TabPFN.
            static_row_index: Row index containing static features. Defaults to -1 (last row).
            max_temporal_length: Max length for temporal sequences (padding/truncation). Determined from data if None.
            standardize_agri_features: Whether to standardize 3D input features.
            handle_missing_agri_values: How to handle NaNs in 3D input ('mean', 'zero', 'none').
            agri_encoder_kwargs: Additional keyword arguments for the AgriDataEncoder.
            n_estimators: Number of estimators for the underlying TabPFN ensemble.
            softmax_temperature: Softmax temperature for underlying TabPFN predictions.
            average_before_softmax: Whether to average logits before softmax in underlying TabPFN.
            model_path: Path or 'auto' for the underlying TabPFN model weights.
            device: Device ('auto', 'cpu', 'cuda', etc.).
            ignore_pretraining_limits: Ignore TabPFN size/feature limits.
            inference_precision: Precision for inference ('auto', 'autocast', torch.dtype).
            fit_mode: Caching mode for the underlying TabPFN ('low_memory', 'fit_preprocessors', 'fit_with_cache').
            memory_saving_mode: Memory saving strategy ('auto', bool, float).
            random_state: Random state for reproducibility.
            n_jobs: Number of parallel jobs for preprocessing (passed to underlying TabPFN).
            inference_config: Advanced interface configuration for underlying TabPFN.
        """
        super().__init__()

        # Store Agri-specific params
        self.temporal_hidden_dim = temporal_hidden_dim
        self.static_hidden_dims = static_hidden_dims if static_hidden_dims is not None else [64]
        self.fusion_strategy = fusion_strategy
        self.encoded_embedding_dim = encoded_embedding_dim
        self.static_row_index = static_row_index
        self.max_temporal_length = max_temporal_length
        self.standardize_agri_features = standardize_agri_features
        self.handle_missing_agri_values = handle_missing_agri_values
        self.agri_encoder_kwargs = agri_encoder_kwargs or {}

        # Store standard TabPFN params
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.model_path = model_path
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.fit_mode = fit_mode
        self.memory_saving_mode = memory_saving_mode
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.inference_config = inference_config

        # Initialize the underlying TabPFN regressor instance immediately
        # We pass all relevant standard TabPFN parameters here.
        # Note: categorical_features_indices is set to None because the agri encoder
        # handles feature transformation into a numeric 2D space before TabPFN sees it.
        self.tabpfn_regressor_ = TabPFNRegressor(
            n_estimators=self.n_estimators,
            categorical_features_indices=None, # Categorical handling done before TabPFN
            softmax_temperature=self.softmax_temperature,
            average_before_softmax=self.average_before_softmax,
            model_path=self.model_path,
            device=self.device,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            inference_precision=self.inference_precision,
            fit_mode=self.fit_mode,
            memory_saving_mode=self.memory_saving_mode,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            inference_config=self.inference_config,
        )
        # Initialize internal fitted state
        self._is_fitted = False

    def _validate_agri_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate that the input data has the correct 3D structure."""
        if not isinstance(X, np.ndarray):
             raise TypeError(f"Input X must be a NumPy array, got {type(X)}")
        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input data (samples, rows, columns), got shape {X.shape}"
            )
        if X.shape[1] < 2: # Need at least one temporal and one static row conceptually
             raise ValueError(
                 f"Input X must have at least 2 rows, got {X.shape[1]} rows."
                 )

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError(f"Input y must be a NumPy array, got {type(y)}")
            if y.ndim != 1:
                raise ValueError(
                    f"Expected 1D target data (samples,), got shape {y.shape}"
                )
            if len(X) != len(y):
                raise ValueError(
                    f"X and y must have the same number of samples (dimension 0). "
                    f"Got X: {len(X)} and y: {len(y)}."
                )

    @config_context(transform_output="default")
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the agricultural TabPFN regressor to the data.

        Args:
            X: Input features of shape (n_samples, n_features_rows, n_features_cols).
            y: Target values of shape (n_samples,).

        Returns:
            Self: The fitted estimator.
        """
        self._validate_agri_input(X, y)
        _, rng = infer_random_state(self.random_state) # Get rng

        # Store input dimensions determined from data
        self.n_features_rows_in_ = X.shape[1]
        self.n_features_cols_in_ = X.shape[2]

        # Determine device and precision
        self.device_ = infer_device_and_type(self.device)
        (self.use_autocast_, self.forced_inference_dtype_, _) = determine_precision(
            self.inference_precision, self.device_
        )

        # --- Preprocessing and Encoding ---
        # 1. Initialize and Fit AgriDataPreprocessor
        logger.debug("Initializing and fitting AgriDataPreprocessor...")
        self.agri_preprocessor_ = AgriDataPreprocessor(
            static_row_index=self.static_row_index,
            max_temporal_length=self.max_temporal_length,
            standardize_features=self.standardize_agri_features,
            handle_missing_values=self.handle_missing_agri_values,
        )
        # Fit preprocessor *before* potential y filtering
        self.agri_preprocessor_.fit(X)
        logger.debug(f"Preprocessor fitted. Determined max temporal length: {self.agri_preprocessor_.max_temporal_length_}")

        # Handle NaNs in y: Remove corresponding samples from X and y
        nan_mask_y = np.isnan(y)
        if nan_mask_y.any():
            n_removed = nan_mask_y.sum()
            if n_removed == len(y):
                 raise ValueError("All target values are NaN. Cannot fit the model.")
            logger.warning(
                f"Found {n_removed} NaN values in the target variable y. "
                f"Removing corresponding samples from X and y before fitting."
            )
            X = X[~nan_mask_y] # Filter X based on valid y
            y = y[~nan_mask_y] # Filter y itself
            if X.shape[0] == 0: # Check if filtering removed all samples
                 raise ValueError("No valid samples remaining after removing NaN targets.")


        # 2. Transform X using the fitted preprocessor
        logger.debug("Transforming training data X with AgriDataPreprocessor...")
        X_processed, temporal_mask = self.agri_preprocessor_.transform(X)
        logger.debug(f"X_processed shape: {X_processed.shape}, temporal_mask shape: {temporal_mask.shape if temporal_mask is not None else 'None'}")

        # 3. Initialize AgriDataEncoder
        logger.debug("Initializing AgriDataEncoder...")
        processed_rows, processed_cols = X_processed.shape[1], X_processed.shape[2]
        # Use fitted max temporal length from preprocessor
        n_temporal_steps = self.agri_preprocessor_.max_temporal_length_ or 0
        temporal_input_dim = processed_cols if n_temporal_steps > 0 else 0
        static_input_dim = processed_cols

        self.agri_encoder_ = AgriDataEncoder(
             temporal_input_dim=temporal_input_dim,
             static_input_dim=static_input_dim,
             output_dim=self.encoded_embedding_dim,
             temporal_hidden_dim=self.temporal_hidden_dim,
             static_hidden_dims=self.static_hidden_dims,
             fusion_strategy=self.fusion_strategy,
             temporal_encoder_kwargs=self.agri_encoder_kwargs.get('temporal_encoder_kwargs'),
             static_encoder_kwargs=self.agri_encoder_kwargs.get('static_encoder_kwargs'),
        )
        self.agri_encoder_.to(self.device_)
        logger.debug("AgriDataEncoder initialized and moved to device.")

        # 4. Encode X_processed -> X_encoded (2D)
        logger.debug("Encoding preprocessed training data...")
        X_processed_tensor = torch.tensor(X_processed, dtype=torch.float32, device=self.device_)
        temporal_mask_tensor = torch.tensor(temporal_mask, dtype=torch.bool, device=self.device_) if temporal_mask is not None else None

        self.agri_encoder_.eval() # Set to eval mode
        with torch.no_grad():
            X_encoded_tensor = self.agri_encoder_(
                X_processed_tensor,
                temporal_mask=temporal_mask_tensor,
                static_row_index=self.static_row_index # Pass index
            )
        X_encoded = X_encoded_tensor.cpu().numpy()
        logger.info(f"Encoded training data shape (X_encoded): {X_encoded.shape}") # Log encoded shape

        # 5. Standardize y (use the potentially filtered y)
        logger.debug("Standardizing target variable y...")
        self.y_train_mean_ = np.mean(y)
        self.y_train_std_ = np.std(y) + 1e-8 # Add epsilon
        y_standardized = (y - self.y_train_mean_) / self.y_train_std_
        logger.debug(f"y standardized: mean={self.y_train_mean_:.4f}, std={self.y_train_std_:.4f}")


        # --- Fit Underlying TabPFN ---
        # 6. Fit the internal TabPFNRegressor instance
        logger.info(f"Fitting underlying TabPFNRegressor with encoded data of shape {X_encoded.shape}...")
        # The internal regressor will handle its own model loading, config setup etc.
        # It also sets its own fitted attributes.
        self.tabpfn_regressor_.fit(X_encoded, y_standardized)

        # 7. Store necessary attributes from the fitted underlying regressor
        # These are essential for the predict method and sklearn compatibility checks
        self.config_ = self.tabpfn_regressor_.config_
        self.interface_config_ = self.tabpfn_regressor_.interface_config_
        self.executor_ = self.tabpfn_regressor_.executor_
        self.model_ = self.tabpfn_regressor_.model_ # Store reference to TabPFN's transformer
        self.bardist_ = self.tabpfn_regressor_.bardist_
        self.renormalized_criterion_ = self.tabpfn_regressor_.renormalized_criterion_


        self._is_fitted = True # Mark AgriTabPFN as fitted
        logger.info("AgriTabPFNRegressor fitting completed.")
        return self

    # Overload definitions remain the same
    @overload
    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: Literal["mean", "median", "mode"] = "mean",
        quantiles: Optional[List[float]] = None,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: Literal["quantiles"],
        quantiles: Optional[List[float]] = None,
    ) -> List[np.ndarray]: ...

    @config_context(transform_output="default")
    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: Literal["mean", "median", "mode", "quantiles"] = "mean",
        quantiles: Optional[List[float]] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Predict using the fitted agricultural TabPFN regressor.

        Args:
            X: Input features of shape (n_samples, n_features_rows, n_features_cols).
               Must match the structure used during fitting.
            output_type: Type of prediction ('mean', 'median', 'mode', 'quantiles').
            quantiles: List of quantiles [0.0, 1.0] if output_type is 'quantiles'.

        Returns:
            Predicted values (np.ndarray) or list of quantile predictions (List[np.ndarray]).
        """
        check_is_fitted(self) # Uses the __sklearn_is_fitted__ method
        self._validate_agri_input(X) # Basic shape validation

        if X.shape[1] != self.n_features_rows_in_ or X.shape[2] != self.n_features_cols_in_:
            raise ValueError(
                f"Input data dimensions ({X.shape[1]}, {X.shape[2]}) do not match "
                f"training data dimensions ({self.n_features_rows_in_}, {self.n_features_cols_in_})."
            )
        if self.agri_preprocessor_ is None or self.agri_encoder_ is None:
             raise RuntimeError("Preprocessor or Encoder not initialized. Ensure model is fitted.")


        # --- Preprocess and Encode Test Data ---
        # 1. Preprocess X_test
        logger.debug("Transforming test data X with AgriDataPreprocessor...")
        X_processed, temporal_mask = self.agri_preprocessor_.transform(X)
        logger.debug(f"Test X_processed shape: {X_processed.shape}, temporal_mask shape: {temporal_mask.shape if temporal_mask is not None else 'None'}")


        # 2. Encode X_processed -> X_encoded (2D)
        logger.debug("Encoding preprocessed test data...")
        X_processed_tensor = torch.tensor(X_processed, dtype=torch.float32, device=self.device_)
        temporal_mask_tensor = torch.tensor(temporal_mask, dtype=torch.bool, device=self.device_) if temporal_mask is not None else None

        self.agri_encoder_.to(self.device_)
        self.agri_encoder_.eval()
        with torch.no_grad():
             X_encoded_tensor = self.agri_encoder_(
                 X_processed_tensor,
                 temporal_mask=temporal_mask_tensor,
                 static_row_index=self.static_row_index
                 )
        X_encoded = X_encoded_tensor.cpu().numpy()
        logger.debug(f"Encoded test data shape (X_encoded): {X_encoded.shape}")

        # --- Predict using Underlying TabPFN ---
        # 3. Predict using the fitted internal TabPFNRegressor
        # The internal regressor handles its own device placement, autocast, etc.
        logger.debug(f"Predicting with underlying TabPFNRegressor (output_type='{output_type}')...")
        predictions_standardized = self.tabpfn_regressor_.predict(
            X_encoded,
            output_type=output_type, # type: ignore
            quantiles=quantiles,
        )

        # --- Rescale Predictions ---
        # 4. Rescale the standardized predictions back to the original scale
        logger.debug("Rescaling predictions...")
        if output_type == "quantiles":
            if not isinstance(predictions_standardized, list):
                 raise TypeError(f"Expected list for quantile predictions, got {type(predictions_standardized)}")
            # Rescale each numpy array in the list
            rescaled_predictions = [
                q_pred * self.y_train_std_ + self.y_train_mean_
                for q_pred in predictions_standardized
            ]
            logger.debug(f"Rescaled quantile predictions shapes: {[p.shape for p in rescaled_predictions]}")
            return rescaled_predictions
        else:
             if not isinstance(predictions_standardized, np.ndarray):
                 raise TypeError(f"Expected ndarray for {output_type} predictions, got {type(predictions_standardized)}")
             # Rescale the single numpy array
             rescaled_predictions = predictions_standardized * self.y_train_std_ + self.y_train_mean_
             logger.debug(f"Rescaled {output_type} predictions shape: {rescaled_predictions.shape}")
             return rescaled_predictions


    def get_embeddings(
        self,
        X: np.ndarray,
        data_source: Literal["train", "test"] = "test",
    ) -> np.ndarray:
        """Get the embeddings for the input data after 3D preprocessing and encoding.

        Args:
            X: Input features of shape (n_samples, n_features_rows, n_features_cols).
            data_source: Whether to extract 'train' or 'test' embeddings
                         from the underlying TabPFN model's perspective.

        Returns:
            np.ndarray: The computed embeddings from the underlying TabPFN model,
                        shape (n_estimators, n_samples, embedding_dim).
        """
        check_is_fitted(self)
        self._validate_agri_input(X)

        if self.agri_preprocessor_ is None or self.agri_encoder_ is None:
             raise RuntimeError("Preprocessor or Encoder not initialized. Ensure model is fitted.")

        # --- Preprocess and Encode Input Data ---
        logger.debug(f"Preprocessing and encoding data for get_embeddings (data_source='{data_source}')...")
        X_processed, temporal_mask = self.agri_preprocessor_.transform(X)
        X_processed_tensor = torch.tensor(X_processed, dtype=torch.float32, device=self.device_)
        temporal_mask_tensor = torch.tensor(temporal_mask, dtype=torch.bool, device=self.device_) if temporal_mask is not None else None

        self.agri_encoder_.to(self.device_)
        self.agri_encoder_.eval()
        with torch.no_grad():
            X_encoded_tensor = self.agri_encoder_(
                X_processed_tensor,
                temporal_mask=temporal_mask_tensor,
                static_row_index=self.static_row_index
            )
        X_encoded = X_encoded_tensor.cpu().numpy()
        logger.debug(f"Encoded data shape for embeddings: {X_encoded.shape}")


        # --- Get Embeddings from Underlying TabPFN ---
        # Call the get_embeddings method of the fitted internal regressor
        logger.debug(f"Getting embeddings from underlying TabPFNRegressor (data_source='{data_source}')...")
        embeddings = self.tabpfn_regressor_.get_embeddings(X_encoded, data_source=data_source)
        logger.debug(f"Retrieved embeddings shape: {embeddings.shape}")
        return embeddings

    # Implement __sklearn_is_fitted__ for check_is_fitted compatibility
    def __sklearn_is_fitted__(self) -> bool:
        """Check if the estimator is fitted."""
        # Check if essential agri-specific components are fitted/initialized
        agri_components_fitted = all(
            hasattr(self, attr) and getattr(self, attr) is not None
            for attr in ['agri_preprocessor_', 'agri_encoder_', 'tabpfn_regressor_']
        )
        # Check if the underlying TabPFNRegressor instance has been fitted
        # by checking for an attribute typically set during its fit, like 'executor_'
        # Also check the _is_fitted flag set by AgriTabPFN itself.
        underlying_fitted = (
            self.tabpfn_regressor_ is not None and
            # Use check_is_fitted on the underlying regressor for robustness
            # This handles cases where the underlying regressor might change its
            # internal fitted attributes in the future.
            # We need a try-except block as check_is_fitted raises NotFittedError
            # if the underlying model is not fitted.
            # Also check the explicit flag set by AgriTabPFN.
            getattr(self, '_is_fitted', False)
        )
        underlying_really_fitted = False # Assume not fitted unless check passes
        if underlying_fitted and self.tabpfn_regressor_ is not None:
            try:
                 check_is_fitted(self.tabpfn_regressor_)
                 underlying_really_fitted = True
            except Exception: # Catch NotFittedError from check_is_fitted
                 underlying_really_fitted = False


        return agri_components_fitted and underlying_really_fitted

    # Optional: Add _more_tags if needed for specific sklearn checks,
    # but usually inheriting from Mixins is enough.
    # def _more_tags(self):
    #     return {"requires_y": True}