"""TabPFN interface for 3D agricultural data.

This module provides a specialized TabPFN interface for agricultural data, which
typically has a 3D structure with time series weather data, NDVI measurements,
and static soil and crop information.

Example usage:
    ```python
    from tabpfn import AgriTabPFNRegressor
    
    # Initialize the model
    model = AgriTabPFNRegressor()
    
    # Fit the model with 3D agricultural data
    # X shape: (n_samples, n_time_steps+static, n_features)
    # y shape: (n_samples,)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X_test)
    ```
"""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

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
    create_inference_engine,
    determine_precision,
    initialize_tabpfn_model,
)
from tabpfn.config import ModelInterfaceConfig
from tabpfn.constants import XType, YType
from tabpfn.encoders.agri_encoders import (
    AgriDataEncoder, 
    AgriDataPreprocessor,
    TimeFeaturesEncoder,
    StaticFeaturesEncoder,
)
from tabpfn.encoders.agri_interface import (
    prepare_agri_data_for_tabpfn,
    encode_agri_batch_for_tabpfn,
)
from tabpfn.preprocessing import (
    EnsembleConfig,
    PreprocessorConfig,
    RegressorEnsembleConfig,
    default_regressor_preprocessor_configs,
)
from tabpfn.regressor import TabPFNRegressor
from tabpfn.utils import (
    infer_device_and_type,
    infer_random_state,
    update_encoder_outlier_params,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from torch.types import _dtype
    from tabpfn.inference import InferenceEngine
    from tabpfn.model.config import InferenceConfig


class AgriTabPFNRegressor(RegressorMixin, BaseEstimator):
    """TabPFN regressor adapted for 3D agricultural data.
    
    This class extends TabPFN's capabilities to work with 3D agricultural data,
    which includes time series weather data, NDVI measurements, and static soil
    and crop information.
    
    The model preserves TabPFN's powerful foundation model architecture while
    adding specialized encoding for agricultural data structure.
    """
    
    config_: InferenceConfig
    """The configuration of the loaded model to be used for inference."""
    
    interface_config_: ModelInterfaceConfig
    """Additional configuration of the interface for expert users."""
    
    device_: torch.device
    """The device determined to be used."""
    
    n_features_rows_in_: int
    """The number of feature rows in the input data (time steps + static rows)."""
    
    n_features_cols_in_: int
    """The number of feature columns in the input data (variables per row)."""
    
    n_outputs_: int = 1
    """The number of outputs the model supports. Only 1 for now."""
    
    use_autocast_: bool
    """Whether torch's autocast should be used."""
    
    forced_inference_dtype_: _dtype | None
    """The forced inference dtype for the model based on `inference_precision`."""
    
    executor_: InferenceEngine
    """The inference engine used to make predictions."""
    
    tabpfn_regressor_: TabPFNRegressor
    """The underlying TabPFN regressor used for predictions after encoding."""
    
    agri_preprocessor_: AgriDataPreprocessor
    """The preprocessor for 3D agricultural data."""
    
    agri_encoder_: AgriDataEncoder
    """The encoder for 3D agricultural data."""
    
    y_train_mean_: float
    """The mean of the target variable during training."""
    
    y_train_std_: float
    """The standard deviation of the target variable during training."""
    
    def __init__(  # noqa: PLR0913
        self,
        *,
        n_estimators: int = 8,
        temporal_hidden_dim: int = 128,
        static_hidden_dims: List[int] = None,
        fusion_strategy: Literal["concat", "attention", "gated"] = "gated",
        output_dim: int = 128,
        encoder_kwargs: Dict[str, Any] = None,
        preprocessor_kwargs: Dict[str, Any] = None,
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
        inference_config: dict | ModelInterfaceConfig | None = None,
    ) -> None:
        """Initialize the agricultural TabPFN regressor.
        
        Args:
            n_estimators:
                The number of estimators in the TabPFN ensemble.
            
            temporal_hidden_dim:
                Size of hidden temporal representations in the encoder.
            
            static_hidden_dims:
                List of hidden layer dimensions for static features in the encoder.
                If None, defaults to [64].
            
            fusion_strategy:
                Strategy for combining temporal and static representations:
                - "concat": Simple concatenation followed by projection
                - "attention": Use attention mechanism for fusion
                - "gated": Gated fusion mechanism with learnable parameters
            
            output_dim:
                Dimensionality of the final combined embedding.
            
            encoder_kwargs:
                Additional arguments for the agricultural data encoder.
            
            preprocessor_kwargs:
                Additional arguments for the agricultural data preprocessor.
            
            softmax_temperature:
                The temperature for the softmax function. Lower values make 
                predictions more confident.
            
            average_before_softmax:
                Whether to average predictions before applying softmax.
            
            model_path:
                The path to the TabPFN model file.
                If "auto", downloads the model upon first use.
            
            device:
                The device to use for inference.
                If "auto", uses CUDA if available, otherwise CPU.
            
            ignore_pretraining_limits:
                Whether to ignore TabPFN's pretraining limits.
            
            inference_precision:
                The precision to use for inference.
            
            fit_mode:
                Mode for caching preprocessed data during inference.
            
            memory_saving_mode:
                Enable GPU/CPU memory saving mode.
            
            random_state:
                Controls the randomness of the model.
            
            n_jobs:
                The number of workers for parallelizable tasks.
            
            inference_config:
                Additional advanced arguments for the model interface.
        """
        super().__init__()
        
        # Initialize default values for optional parameters
        if static_hidden_dims is None:
            static_hidden_dims = [64]
        
        if encoder_kwargs is None:
            encoder_kwargs = {}
        
        if preprocessor_kwargs is None:
            preprocessor_kwargs = {}
        
        # Store all configuration parameters
        self.n_estimators = n_estimators
        self.temporal_hidden_dim = temporal_hidden_dim
        self.static_hidden_dims = static_hidden_dims
        self.fusion_strategy = fusion_strategy
        self.output_dim = output_dim
        self.encoder_kwargs = encoder_kwargs
        self.preprocessor_kwargs = preprocessor_kwargs
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
        
        # Initialize underlying TabPFN regressor
        self.tabpfn_regressor_ = TabPFNRegressor(
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            average_before_softmax=average_before_softmax,
            model_path=model_path,
            device=device,
            ignore_pretraining_limits=ignore_pretraining_limits,
            inference_precision=inference_precision,
            fit_mode=fit_mode,
            memory_saving_mode=memory_saving_mode,
            random_state=random_state,
            n_jobs=n_jobs,
            inference_config=inference_config,
        )
    
    def _validate_agri_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate that the input data has the correct 3D structure for agricultural data.
        
        Args:
            X: Input features of shape (n_samples, n_features_rows, n_features_cols)
            y: Optional target values of shape (n_samples,)
            
        Raises:
            ValueError: If the input data does not have the expected shape
        """
        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D input data for agricultural model, got shape {X.shape}"
            )
        
        if y is not None:
            if y.ndim != 1:
                raise ValueError(
                    f"Expected 1D target data, got shape {y.shape}"
                )
            
            if len(X) != len(y):
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got {len(X)} samples in X and {len(y)} samples in y."
                )
    
    @config_context(transform_output="default")  # type: ignore
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit the agricultural TabPFN regressor to the data.
        
        Args:
            X: Input features of shape (n_samples, n_features_rows, n_features_cols)
            y: Target values of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        # Validate 3D input structure
        self._validate_agri_input(X, y)
        
        # Store input dimensions
        self.n_features_rows_in_ = X.shape[1]
        self.n_features_cols_in_ = X.shape[2]
        
        # Determine device and precision
        self.device_ = infer_device_and_type(self.device)
        (self.use_autocast_, self.forced_inference_dtype_, byte_size) = (
            determine_precision(self.inference_precision, self.device_)
        )
        
        static_seed, rng = infer_random_state(self.random_state)
        
        # Prepare agricultural data: preprocess and create encoder
        self.agri_preprocessor_, self.agri_encoder_, X_processed, y_processed = (
            prepare_agri_data_for_tabpfn(
                X, 
                y,
                preprocessor=AgriDataPreprocessor(**self.preprocessor_kwargs),
            )
        )
        
        # Configure agricultural encoder with appropriate dimensions
        temporal_input_dim = X.shape[2]  # Number of features per time step
        static_input_dim = X.shape[2]    # Number of static features
        
        # If not already set in prepare_agri_data_for_tabpfn, create the encoder here
        if self.agri_encoder_ is None:
            self.agri_encoder_ = AgriDataEncoder(
                temporal_input_dim=temporal_input_dim,
                temporal_hidden_dim=self.temporal_hidden_dim,
                static_input_dim=static_input_dim,
                static_hidden_dims=self.static_hidden_dims,
                output_dim=self.output_dim,
                fusion_strategy=self.fusion_strategy,
                **self.encoder_kwargs,
            )
        
        # Encode the processed 3D data into 2D format for TabPFN
        X_encoded = encode_agri_batch_for_tabpfn(
            X_processed, 
            self.agri_encoder_,
            device=self.device_,
        ).cpu().numpy()
        
        # Standardize target values
        mean = np.mean(y_processed)
        std = np.std(y_processed)
        self.y_train_mean_ = mean.item()
        self.y_train_std_ = std.item() + 1e-20
        y_standardized = (y_processed - self.y_train_mean_) / self.y_train_std_
        
        # Fit the underlying TabPFN regressor with the encoded data
        self.tabpfn_regressor_.fit(X_encoded, y_standardized)
        
        # Store necessary attributes from the underlying regressor
        self.config_ = self.tabpfn_regressor_.config_
        self.interface_config_ = self.tabpfn_regressor_.interface_config_
        self.executor_ = self.tabpfn_regressor_.executor_
        
        return self
    
    @overload
    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: Literal["mean", "median", "mode"] = "mean",
        quantiles: List[float] | None = None,
    ) -> np.ndarray: ...
    
    @overload
    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: Literal["quantiles"],
        quantiles: List[float] | None = None,
    ) -> List[np.ndarray]: ...
    
    @config_context(transform_output="default")  # type: ignore
    def predict(
        self,
        X: np.ndarray,
        *,
        output_type: Literal["mean", "median", "mode", "quantiles"] = "mean",
        quantiles: List[float] | None = None,
    ) -> np.ndarray | List[np.ndarray]:
        """Predict using the agricultural TabPFN regressor.
        
        Args:
            X: Input features of shape (n_samples, n_features_rows, n_features_cols)
            output_type: Type of prediction to return:
                - "mean": Mean of the predicted distribution
                - "median": Median of the predicted distribution
                - "mode": Mode of the predicted distribution
                - "quantiles": Quantiles of the predicted distribution
            quantiles: Quantiles to return if output_type="quantiles"
                
        Returns:
            Predictions according to the specified output_type
        """
        check_is_fitted(self)
        
        # Validate 3D input structure
        self._validate_agri_input(X)
        
        # Ensure dimensions match training data
        if X.shape[1] != self.n_features_rows_in_ or X.shape[2] != self.n_features_cols_in_:
            raise ValueError(
                f"Input data dimensions ({X.shape[1]}, {X.shape[2]}) do not match "
                f"training data dimensions ({self.n_features_rows_in_}, {self.n_features_cols_in_})"
            )
        
        # Preprocess the input data
        X_processed = self.agri_preprocessor_.transform(X)
        
        # Encode the processed data for TabPFN
        X_encoded = encode_agri_batch_for_tabpfn(
            X_processed, 
            self.agri_encoder_, 
            device=self.device_,
        ).cpu().numpy()
        
        # Make predictions with the underlying TabPFN regressor
        predictions = self.tabpfn_regressor_.predict(
            X_encoded,
            output_type=output_type,
            quantiles=quantiles,
        )
        
        # If returning quantiles, rescale each quantile prediction
        if output_type == "quantiles":
            return [
                q_pred * self.y_train_std_ + self.y_train_mean_ 
                for q_pred in predictions
            ]
        
        # Otherwise, rescale and return the predictions
        return predictions * self.y_train_std_ + self.y_train_mean_
    
    def get_embeddings(
        self,
        X: np.ndarray,
        data_source: Literal["train", "test"] = "test",
    ) -> np.ndarray:
        """Get the embeddings for the input data.
        
        Args:
            X: Input features of shape (n_samples, n_features_rows, n_features_cols)
            data_source: Whether to extract train or test embeddings
            
        Returns:
            The computed embeddings for each fitted estimator
        """
        check_is_fitted(self)
        
        # Validate 3D input structure
        self._validate_agri_input(X)
        
        # Preprocess the input data
        X_processed = self.agri_preprocessor_.transform(X)
        
        # Encode the processed data for TabPFN
        X_encoded = encode_agri_batch_for_tabpfn(
            X_processed, 
            self.agri_encoder_, 
            device=self.device_,
        ).cpu().numpy()
        
        # Get embeddings from the underlying TabPFN regressor
        return self.tabpfn_regressor_.get_embeddings(X_encoded, data_source=data_source)
