# File: TabPFN/src/tabpfn/agri_utils/experiment_utils.py
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pylint: disable=line-too-long
"""
Utility functions for running agricultural yield prediction experiments.

This module provides functions for:
- Loading processed data from NPZ files.
- Dynamically importing and instantiating models.
- Preprocessing 3D list data into 2D arrays for baseline models.
- Calculating evaluation metrics.
- Saving model checkpoints and predictions.
"""
import importlib
import logging
import json
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple, Union, Callable
import sys

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

# --- Constants ---
METRICS_FILENAME = "metrics.json"
PREDICTIONS_FILENAME = "predictions_test.npz"
MODEL_CHECKPOINT_DIRNAME = "model_checkpoint"
CONFIG_USED_FILENAME = "config_used.yaml"
RUN_LOG_FILENAME = "run_log.txt"

SUPPORTED_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),
    "mae": mean_absolute_error,
    "r2": r2_score,
}

def load_processed_npz_data(
    npz_file_path: Union[str, Path]
) -> Tuple[List[np.ndarray], np.ndarray, List[Any]]:
    """Loads features, targets, and info from a processed .npz file."""
    npz_path = Path(npz_file_path)
    if not npz_path.exists():
        logger.error(f"NPZ file not found: {npz_path}")
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    logger.info(f"Loading processed data from: {npz_path}")
    try:
        data = np.load(npz_path, allow_pickle=True)
        features = list(data["features"])
        targets = data["targets"]
        info = list(data["info"])
        logger.info(f"Successfully loaded {len(features)} samples from {npz_path}.")
        return features, targets, info
    except KeyError as e:
        logger.error(f"Missing expected key in NPZ file {npz_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading NPZ file {npz_path}: {e}")
        raise


def dynamic_import_and_instantiate_model(
    model_class_path: str, **kwargs: Any
) -> Any:
    """Dynamically imports and instantiates a model class."""
    try:
        module_path, class_name = model_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        logger.debug(f"Instantiating {model_class_path} with kwargs: {kwargs}")
        model_instance = model_class(**kwargs)
        logger.info(f"Successfully instantiated model: {model_class_path}")
        return model_instance
    except ImportError as e:
        logger.error(f"Failed to import module for class path {model_class_path}: {e}")
        raise
    except AttributeError as e:
        logger.error(f"Failed to find class {class_name} in module {module_path}: {e}")
        raise
    except TypeError as e:
        logger.error(f"Type error during model instantiation for {model_class_path} with params {kwargs}: {e}")
        raise


def preprocess_for_baseline(
    features_list_3d: List[np.ndarray],
    static_row_index: int,
    temporal_aggregation_methods: List[str],
) -> np.ndarray:
    """Transforms 3D list data to 2D array for baseline models."""
    if not features_list_3d:
        return np.array([])

    processed_samples: List[np.ndarray] = []
    for sample_idx, sample_3d in enumerate(features_list_3d):
        if sample_3d.ndim != 2:
            raise ValueError(
                f"Sample {sample_idx}: Expected 2D array, got {sample_3d.ndim}D"
            )

        actual_static_row_idx = static_row_index
        if actual_static_row_idx < 0:
            actual_static_row_idx = sample_3d.shape[0] + actual_static_row_idx

        if not (0 <= actual_static_row_idx < sample_3d.shape[0]):
            raise ValueError(
                f"Sample {sample_idx}: static_row_index {static_row_index} "
                f"is out of bounds for sample with shape {sample_3d.shape}"
            )

        static_features = sample_3d[actual_static_row_idx, :].astype(np.float64)
        temporal_features_block = np.delete(sample_3d, actual_static_row_idx, axis=0)
        temporal_features_block_numeric = temporal_features_block.astype(np.float64, copy=False)

        aggregated_temporal_features: List[np.ndarray] = []

        if temporal_features_block_numeric.size > 0:
            for method_name in temporal_aggregation_methods:
                agg_val: np.ndarray
                try:
                    if method_name == "mean":
                        agg_val = np.nanmean(temporal_features_block_numeric, axis=0)
                    elif method_name == "std":
                        # Calculate std robustly
                        var_val = np.nanvar(temporal_features_block_numeric, axis=0)
                        # Ensure var_val is an array, esp. if a column was all NaNs
                        var_val = np.atleast_1d(var_val)
                        # Suppress RuntimeWarning for sqrt of NaN if var_val contains NaNs
                        with np.errstate(invalid='ignore'):
                            agg_val = np.sqrt(var_val)
                    elif method_name == "min":
                        agg_val = np.nanmin(temporal_features_block_numeric, axis=0)
                    elif method_name == "max":
                        agg_val = np.nanmax(temporal_features_block_numeric, axis=0)
                    elif method_name == "median":
                        agg_val = np.nanmedian(temporal_features_block_numeric, axis=0)
                    elif method_name == "sum":
                        agg_val = np.nansum(temporal_features_block_numeric, axis=0)
                    elif method_name == "var":
                        agg_val = np.nanvar(temporal_features_block_numeric, axis=0)
                    else:
                        logger.warning(
                            f"Sample {sample_idx}: Unsupported temporal aggregation method: {method_name}. Skipping."
                        )
                        continue
                    
                    # Ensure agg_val is always a 1D array of the correct feature dimension
                    if np.isscalar(agg_val): # Can happen if input block is (N,1) and then reduced
                        agg_val = np.full(static_features.shape[0], agg_val)
                    elif agg_val.ndim == 0: # Handle 0-dim array (scalar) from some np functions
                         agg_val = np.full(static_features.shape[0], agg_val.item())


                    aggregated_temporal_features.append(agg_val)

                except Exception as e:
                    logger.error(f"Sample {sample_idx}: Error during temporal aggregation method '{method_name}': {e}")
                    aggregated_temporal_features.append(np.full(static_features.shape[0], np.nan))
        else:
            logger.warning(f"Sample {sample_idx}: has no temporal data rows for baseline preprocessing.")
            num_cols = static_features.shape[0]
            for _ in temporal_aggregation_methods:
                aggregated_temporal_features.append(np.full(num_cols, np.nan))

        # Ensure all aggregated features are 1D arrays before concatenation
        flat_aggregated_list = []
        for arr in aggregated_temporal_features:
            flat_aggregated_list.append(np.atleast_1d(arr))
        
        flat_aggregated_temporal = np.concatenate(flat_aggregated_list) if flat_aggregated_list else np.array([])
        
        processed_sample_row = np.concatenate([flat_aggregated_temporal, np.atleast_1d(static_features)])
        processed_samples.append(processed_sample_row)

    return np.array(processed_samples, dtype=np.float64)


def calculate_evaluation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, metric_names: List[str]
) -> Dict[str, float]:
    """Calculates specified evaluation metrics."""
    results: Dict[str, float] = {}
    for metric_name in metric_names:
        metric_func = SUPPORTED_METRICS.get(metric_name.lower())
        if metric_func:
            try:
                results[metric_name] = float(metric_func(y_true, y_pred))
            except Exception as e:
                logger.error(f"Error calculating metric {metric_name}: {e}")
                results[metric_name] = np.nan
        else:
            logger.warning(f"Unsupported metric: {metric_name}. Skipping.")
    return results


def save_model_checkpoint(model: Any, model_name: str, save_dir: Path) -> None:
    """Saves the trained model."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{model_name}_trained.joblib"

    if model_name == "AgriTabPFNRegressor":
        logger.warning(
            "Saving AgriTabPFNRegressor with joblib. This might not capture all "
            "internal states perfectly. Consider a custom save/load mechanism."
        )
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model {model_name} to {model_path}: {e}")
        logger.error("Model saving failed.")


def save_predictions_to_npz(
    save_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_sample_info: List[Any],
) -> None:
    """Saves predictions to an NPZ file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / PREDICTIONS_FILENAME
    try:
        np.savez_compressed(
            file_path,
            y_true=y_true,
            y_pred=y_pred,
            test_sample_info=np.array(test_sample_info, dtype=object)
        )
        logger.info(f"Test predictions saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving predictions to {file_path}: {e}")


def save_metrics_to_json(metrics_dict: Dict[str, Any], save_dir: Path) -> None:
    """Saves metrics to a JSON file."""
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / METRICS_FILENAME
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Metrics saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {file_path}: {e}")

def load_config_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Configuration file not found: {path}")
        raise FileNotFoundError(f"Configuration file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading YAML {path}: {e}")
        raise

def setup_logging(log_path: Path, level_str: str = "INFO") -> None:
    """Configures file-based logging for the experiment run."""
    log_level = getattr(logging, level_str.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    # Important: Clear existing handlers to avoid duplicate logs or file lock issues
    # especially if this function is called multiple times (e.g., in tests).
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            # Close handler before removing to release file locks
            if isinstance(handler, logging.FileHandler):
                handler.close()
            root_logger.removeHandler(handler)
        
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # sys needs to be imported
        ]
    )
    # This logger is the one for the current module (experiment_utils.py)
    logging.getLogger(__name__).info(
        f"Logging configured. Log file: {log_path}, Level: {level_str.upper()}"
    )


def create_experiment_output_dir(
    base_results_dir: Path,
    template_str: str,
    experiment_id: str,
    experiment_name: str,
    model_name: str
    ) -> Path:
    """Creates a unique output directory for the experiment run."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    try:
        subdir_name = template_str.format(
            experiment_id=experiment_id,
            experiment_name=experiment_name.replace(" ", "_").lower(),
            model_name=model_name.replace(" ", "_").lower(),
            timestamp=timestamp
        )
    except KeyError as e:
        logger.error(f"Invalid placeholder in experiment_subdir_template '{template_str}': {e}")
        logger.warning("Using fallback directory name structure.")
        subdir_name = f"{experiment_id}_{timestamp}"

    experiment_dir = base_results_dir / subdir_name
    try:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created experiment output directory: {experiment_dir}")
    except OSError as e:
        logger.error(f"Could not create experiment output directory {experiment_dir}: {e}")
        fallback_dir_name = f"exp_run_fallback_{timestamp}"
        experiment_dir = base_results_dir / fallback_dir_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Fell back to output directory: {experiment_dir}")
    return experiment_dir
