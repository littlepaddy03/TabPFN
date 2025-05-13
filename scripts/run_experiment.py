# File: TabPFN/scripts/run_experiment.py
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
# pylint: disable=line-too-long,broad-except
"""
Main script to run a single agricultural yield prediction experiment based on a YAML configuration file.
"""
import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from typing import Union, Dict, Any, List, Tuple

# Ensure src path is available for utility imports
SCRIPT_DIR_RUN = Path(__file__).resolve().parent
TABPFN_ROOT_RUN = SCRIPT_DIR_RUN.parent
SRC_DIR_RUN = TABPFN_ROOT_RUN / "src"

if SRC_DIR_RUN.exists() and str(SRC_DIR_RUN) not in sys.path:
    sys.path.insert(0, str(SRC_DIR_RUN))
if TABPFN_ROOT_RUN.exists() and str(TABPFN_ROOT_RUN) not in sys.path: # If src is not directly there
    sys.path.insert(0, str(TABPFN_ROOT_RUN))

try:
    from tabpfn.agri_utils.experiment_utils import (
        load_processed_npz_data,
        dynamic_import_and_instantiate_model,
        preprocess_for_baseline,
        calculate_evaluation_metrics,
        save_model_checkpoint,
        save_predictions_to_npz,
        save_metrics_to_json,
        load_config_yaml,
        setup_logging,
        create_experiment_output_dir,
        METRICS_FILENAME,
        PREDICTIONS_FILENAME,
        MODEL_CHECKPOINT_DIRNAME,
        CONFIG_USED_FILENAME,
        RUN_LOG_FILENAME
    )
    # Directly import core Agri components assuming fixed paths
    # based on the project structure.
    # IMPORTANT: User must ensure these paths are correct for their project.
    from tabpfn.agri_tabpfn import AgriDataPreprocessor
    from tabpfn.encoders.agri_encoders import AgriDataEncoder
    # AgriTabPFNRegressor will be imported dynamically via model_class_path from config
except ImportError as e:
    print(f"Critical Import Error in run_experiment.py: {e}. Check PYTHONPATH, __init__.py files, and file locations.", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger(__name__)

def run_single_experiment(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Runs a single experiment as defined by the configuration file.
    """
    experiment_output_dir = None 
    try:
        # 1. Load Configuration
        config = load_config_yaml(config_path)

        # 2. Setup Output Directory and Logging
        output_conf = config.get("output", {})
        base_results_dir = Path(output_conf.get("base_results_dir", "experiment_results"))
        exp_subdir_template = output_conf.get("experiment_subdir_template", "{experiment_id}_{model_name}_{timestamp}")
        
        meta_conf = config.get("experiment_metadata", {})
        exp_id = meta_conf.get("experiment_id", f"exp_{time.strftime('%Y%m%d%H%M%S')}")
        exp_name = meta_conf.get("experiment_name", "unnamed_experiment")
        model_conf = config.get("model", {})
        model_name_for_dir = model_conf.get("model_name", "unknown_model")

        experiment_output_dir = create_experiment_output_dir(
            base_results_dir, exp_subdir_template, exp_id, exp_name, model_name_for_dir
        )
        
        log_file_path = experiment_output_dir / RUN_LOG_FILENAME
        log_level = output_conf.get("log_level", "INFO")
        setup_logging(log_file_path, log_level)

        logger.info(f"Starting experiment: {exp_name} (ID: {exp_id})")
        logger.info(f"Configuration loaded from: {Path(config_path).resolve()}")
        logger.info(f"Results will be saved to: {experiment_output_dir.resolve()}")

        try:
            shutil.copy(Path(config_path), experiment_output_dir / CONFIG_USED_FILENAME)
            logger.info(f"Copied configuration to {experiment_output_dir / CONFIG_USED_FILENAME}")
        except Exception as e:
            logger.error(f"Could not copy config file to output directory: {e}")

        # 3. Set Random Seed
        random_seed = meta_conf.get("random_seed", None)
        if random_seed is not None:
            np.random.seed(random_seed) # type: ignore[call-overload]
            logger.info(f"Global random seed set to: {random_seed}")

        # 4. Load Data
        data_conf = config.get("data", {})
        train_npz_path_str = data_conf.get("train_npz_path", "")
        test_npz_path_str = data_conf.get("test_npz_path", "")

        config_dir = Path(config_path).parent
        train_npz_path = Path(train_npz_path_str)
        if not train_npz_path.is_absolute():
            train_npz_path = (config_dir / train_npz_path).resolve()
        
        test_npz_path = Path(test_npz_path_str)
        if not test_npz_path.is_absolute():
            test_npz_path = (config_dir / test_npz_path).resolve()

        if not train_npz_path_str or not test_npz_path_str:
            logger.error("Train or test NPZ data path not specified in configuration.")
            raise ValueError("Missing train/test NPZ paths in config.")
        if not train_npz_path.exists() or not test_npz_path.exists():
            logger.error(f"Train ({train_npz_path}) or Test ({test_npz_path}) NPZ data file not found.")
            raise FileNotFoundError("Train or Test NPZ data file not found after path resolution.")

        X_train_list, y_train, train_info = load_processed_npz_data(train_npz_path)
        X_test_list, y_test, test_info = load_processed_npz_data(test_npz_path)

        if not X_train_list or y_train.size == 0:
            logger.error("Training data is empty. Aborting.")
            raise ValueError("Empty training data loaded.")

        # 5. Instantiate Model
        model_name = model_conf.get("model_name", "")
        model_class_path = model_conf.get("model_class_path", "") # Path for the main model (e.g., AgriTabPFNRegressor)
        model_params_from_config = model_conf.get("model_params", {})

        if not model_name or not model_class_path:
            logger.error("Model name or class path not specified.")
            raise ValueError("Missing model_name or model_class_path in config.")

        instantiation_kwargs: Dict[str, Any] = {}

        if model_name == "AgriTabPFNRegressor":
            preprocessor_kwargs = model_params_from_config.get("agri_preprocessor_kwargs", {})
            encoder_kwargs = model_params_from_config.get("agri_encoder_kwargs", {})
            
            logger.info(f"Instantiating AgriDataPreprocessor with params: {preprocessor_kwargs}")
            # Directly use imported class
            agri_preprocessor_instance = AgriDataPreprocessor(**preprocessor_kwargs)
            
            logger.info(f"Instantiating AgriDataEncoder with params: {encoder_kwargs}")
            # Directly use imported class
            agri_encoder_instance = AgriDataEncoder(**encoder_kwargs)

            instantiation_kwargs["agri_preprocessor"] = agri_preprocessor_instance
            instantiation_kwargs["agri_encoder"] = agri_encoder_instance

            tabpfn_reg_kwargs = model_params_from_config.get("tabpfn_regressor_kwargs", {})
            instantiation_kwargs.update(tabpfn_reg_kwargs)

            if 'random_state' in model_params_from_config:
                instantiation_kwargs['random_state'] = model_params_from_config['random_state']
            elif random_seed is not None and 'random_state' not in instantiation_kwargs:
                instantiation_kwargs['random_state'] = random_seed
            
            if 'device' in model_params_from_config:
                 instantiation_kwargs['device'] = model_params_from_config['device']
            elif 'device' not in instantiation_kwargs:
                 instantiation_kwargs['device'] = 'cpu'
        else: 
            instantiation_kwargs.update(model_params_from_config)
            if "random_state" not in instantiation_kwargs and random_seed is not None:
                instantiation_kwargs["random_state"] = random_seed
        
        logger.info(f"Instantiating main model: {model_name} from {model_class_path} with final kwargs: {instantiation_kwargs}")
        model = dynamic_import_and_instantiate_model(model_class_path, **instantiation_kwargs)

        # 6. Data Preprocessing for Baseline Models
        X_train_processed: Union[List[np.ndarray], np.ndarray] = X_train_list
        X_test_processed: Union[List[np.ndarray], np.ndarray] = X_test_list

        if model_name != "AgriTabPFNRegressor": 
            baseline_prep_conf = config.get("baseline_preprocessing", {})
            if baseline_prep_conf.get("enabled", False):
                logger.info("Applying baseline preprocessing (3D list to 2D array).")
                static_idx = baseline_prep_conf.get("static_row_index", -1)
                agg_methods = baseline_prep_conf.get("temporal_aggregation", {}).get("methods", ["mean", "std"])
                
                X_train_processed = preprocess_for_baseline(X_train_list, static_idx, agg_methods)
                X_test_processed = preprocess_for_baseline(X_test_list, static_idx, agg_methods)
                logger.info(f"Shape of X_train after baseline preprocessing: {X_train_processed.shape}")
            else:
                logger.warning(
                    f"Model {model_name} is not AgriTabPFNRegressor, but baseline_preprocessing is not enabled. "
                    "Model will receive List[np.ndarray]. Ensure it can handle this format."
                )
        
        # 7. Train Model
        logger.info(f"Starting training for model: {model_name}")
        train_start_time = time.time()
        model.fit(X_train_processed, y_train)
        train_duration_seconds = time.time() - train_start_time
        logger.info(f"Training completed in {train_duration_seconds:.2f} seconds.")

        # 8. Make Predictions
        logger.info("Starting predictions on the test set.")
        predict_start_time = time.time()
        y_pred_test = model.predict(X_test_processed)
        predict_duration_seconds = time.time() - predict_start_time
        logger.info(f"Predictions completed in {predict_duration_seconds:.2f} seconds.")

        y_pred_train = None
        if config.get("evaluation", {}).get("evaluate_on_train_set", False):
            logger.info("Starting predictions on the train set.")
            try:
                y_pred_train = model.predict(X_train_processed)
            except Exception as e:
                logger.warning(f"Could not make predictions on train set: {e}")

        # 9. Evaluate Model
        eval_conf = config.get("evaluation", {})
        metric_names = eval_conf.get("metrics", ["rmse", "mae", "r2"])
        logger.info(f"Calculating evaluation metrics for test set: {metric_names}")
        
        test_metrics = calculate_evaluation_metrics(y_test, y_pred_test, metric_names)
        logger.info(f"Test Set Metrics: {test_metrics}")

        train_metrics = {}
        if y_pred_train is not None:
            logger.info(f"Calculating evaluation metrics for train set: {metric_names}")
            train_metrics = calculate_evaluation_metrics(y_train, y_pred_train, metric_names)
            logger.info(f"Train Set Metrics: {train_metrics}")

        # 10. Save Results
        timestamp_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        metrics_summary = {
            "experiment_id": exp_id,
            "experiment_name": exp_name,
            "model_name": model_name,
            "model_class_path": model_class_path,
            "model_params_from_config": model_params_from_config, 
            "model_instantiation_kwargs": instantiation_kwargs, 
            "timestamp_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(train_start_time - train_duration_seconds)),
            "timestamp_end_utc": timestamp_end,
            "training_duration_seconds": round(train_duration_seconds, 2),
            "inference_duration_seconds": round(predict_duration_seconds, 2),
            "evaluation_metrics_test": test_metrics,
            "data_info": {
                "train_npz_path": str(train_npz_path.resolve()),
                "test_npz_path": str(test_npz_path.resolve()),
                "num_train_samples": len(X_train_list),
                "num_test_samples": len(X_test_list),
                "random_seed_used": random_seed,
            }
        }
        if train_metrics:
            metrics_summary["evaluation_metrics_train"] = train_metrics

        save_metrics_to_json(metrics_summary, experiment_output_dir)

        if output_conf.get("save_predictions", True):
            save_predictions_to_npz(experiment_output_dir, y_test, y_pred_test, test_info)

        if output_conf.get("save_model", True):
            model_checkpoint_path = experiment_output_dir / MODEL_CHECKPOINT_DIRNAME
            save_model_checkpoint(model, model_name, model_checkpoint_path)

        logger.info(f"Experiment {exp_id} finished successfully.")
        return test_metrics
    
    except Exception as e:
        if logger.handlers: # Check if logger was successfully configured
            logger.critical(f"Experiment run failed critically during execution: {e}", exc_info=True)
        else: # Fallback to print if logger setup failed
            print(f"CRITICAL ERROR in run_single_experiment (logging may not be set up): {e}", file=sys.stderr)
        raise 
    finally:
        logging.shutdown()


def main_cli():
    """Command-line interface for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run a machine learning experiment using a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment configuration file."
    )
    args = parser.parse_args()

    try:
        run_single_experiment(args.config)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
