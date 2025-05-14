# Copyright 2024 Google LLC
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
"""
Main script to run a single experiment based on a YAML configuration file.

This script handles:
1. Loading experiment configuration from a YAML file.
2. Setting up logging.
3. Loading and splitting data (standard, agricultural, or pre-split NPZ).
4. Instantiating the specified model.
5. Training the model.
6. Making predictions.
7. Evaluating the model.
8. Saving results and trained model.
"""

import argparse
import logging
import os
import time
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.agri_tabpfn import AgriTabPFNRegressor
# AgriDataLoader import remains commented as it's not yet implemented/used.
# from tabpfn.agri_utils.data_loader import AgriDataLoader
from tabpfn.encoders.agri_encoders import AgriDataEncoder
from tabpfn.encoders.agri_interface import AgriDataPreprocessor

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads experiment configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the experiment configuration.
    """
    logger.info('Loading configuration from %s', config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info('Configuration loaded successfully.')
    return config


def load_and_split_data(
    data_config: Dict[str, Any], random_state: int
) -> Tuple[
    Union[pd.DataFrame, np.ndarray],
    Union[pd.DataFrame, np.ndarray],
    Union[pd.Series, np.ndarray],
    Union[pd.Series, np.ndarray],
]:
    """Loads and splits data based on the provided configuration.

    Supports loading from pre-split NPZ files (keys 'features', 'targets'),
    a single agricultural NPY file (keys 'X', 'y' in a dictionary),
    a CSV file, or a predefined dataset name.

    Args:
        data_config: Dictionary containing data loading parameters.
        random_state: Random state for reproducibility of train/test split
                      (only used if data is not pre-split).

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    train_npz_path = data_config.get('train_npz_path')
    test_npz_path = data_config.get('test_npz_path')
    is_agri_data = data_config.get('is_agri_data', False)
    file_path = data_config.get('file_path') # Used for single agri .npy or CSV
    dataset_name = data_config.get('name')
    target_column = data_config.get('target_column', 'target')
    test_size = data_config.get('test_size', 0.2)

    if train_npz_path and test_npz_path:
        logger.info(
            'Loading pre-split data from NPZ files: Train=%s, Test=%s',
            train_npz_path,
            test_npz_path,
        )
        if not os.path.exists(train_npz_path):
            raise FileNotFoundError(f'Train NPZ file not found: {train_npz_path}')
        if not os.path.exists(test_npz_path):
            raise FileNotFoundError(f'Test NPZ file not found: {test_npz_path}')

        train_data = np.load(train_npz_path, allow_pickle=True)
        test_data = np.load(test_npz_path, allow_pickle=True)

        expected_x_key = 'features'
        expected_y_key = 'targets'

        if expected_x_key not in train_data or expected_y_key not in train_data:
            raise KeyError(
                f"Keys '{expected_x_key}' and '{expected_y_key}' not found in"
                f' {train_npz_path}. Available keys: {list(train_data.keys())}'
            )
        if expected_x_key not in test_data or expected_y_key not in test_data:
            raise KeyError(
                f"Keys '{expected_x_key}' and '{expected_y_key}' not found in"
                f' {test_npz_path}. Available keys: {list(test_data.keys())}'
            )

        x_train, y_train = train_data[expected_x_key], train_data[expected_y_key]
        x_test, y_test = test_data[expected_x_key], test_data[expected_y_key]
        logger.info(
            'Loaded X_train shape: %s, y_train shape: %s from NPZ',
            x_train.shape,
            y_train.shape,
        )
        logger.info(
            'Loaded X_test shape: %s, y_test shape: %s from NPZ',
            x_test.shape,
            y_test.shape,
        )
        return x_train, x_test, y_train, y_test

    elif is_agri_data:
        logger.info(
            'Loading agricultural data from single NPY file for splitting: %s',
            file_path,
        )
        if not file_path:
            raise ValueError(
                "'file_path' must be specified for 'is_agri_data' if not using"
                ' pre-split NPZ files.'
            )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f'Agricultural data file not found: {file_path}. '
                'Ensure it is preprocessed (e.g., by'
                ' preprocess_agri_datasets.py).'
            )
        data = np.load(file_path, allow_pickle=True).item()
        if 'X' not in data or 'y' not in data:
            raise KeyError(
                f"Keys 'X' and 'y' not found in agri data file: {file_path}."
                f' Available keys: {list(data.keys())}'
            )

        x_all = data['X']
        y_all = data['y']
        logger.info(
            'Loaded agricultural data X shape: %s, y shape: %s before splitting.',
            x_all.shape,
            y_all.shape,
        )
    elif file_path: # Assumed to be a CSV file
        logger.info('Loading data from CSV for splitting: %s', file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'CSV data file not found: {file_path}')
        data_df = pd.read_csv(file_path)
        if target_column not in data_df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in {file_path}"
            )
        x_all = data_df.drop(columns=[target_column])
        y_all = data_df[target_column]
    elif dataset_name:
        logger.info('Loading predefined dataset for splitting: %s', dataset_name)
        if dataset_name == 'iris':
            from sklearn.datasets import load_iris
            iris = load_iris()
            x_all = pd.DataFrame(iris.data, columns=iris.feature_names)
            y_all = pd.Series(iris.target)
        elif dataset_name == 'boston':
            from sklearn.datasets import fetch_california_housing
            housing = fetch_california_housing()
            x_all = pd.DataFrame(housing.data, columns=housing.feature_names)
            y_all = pd.Series(housing.target)
        else:
            raise ValueError(f'Unsupported predefined dataset_name: {dataset_name}')
    else:
        raise ValueError(
            "Data config must provide valid keys: ('train_npz_path' and "
            "'test_npz_path' with 'features'/'targets' keys), or ('is_agri_data:"
            " True' and 'file_path' to an .npy dictionary with 'X'/'y' keys), "
            "or ('file_path' to a CSV), or ('dataset_name')."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=test_size, random_state=random_state
    )
    logger.info(
        'Data split into training and testing sets. X_train shape: %s, X_test'
        ' shape: %s',
        x_train.shape,
        x_test.shape,
    )
    return x_train, x_test, y_train, y_test


def run_single_experiment(config_path: str) -> Dict[str, Any]:
    """Runs a single experiment based on the configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing the experiment results.
    """
    config = load_config(config_path)
    exp_name = config.get('experiment_metadata', {}).get(
        'experiment_name', config.get('experiment_name', 'default_experiment')
    )
    output_base_dir = config.get('output', {}).get('base_results_dir', 'results')
    exp_subdir_template = config.get('output', {}).get(
        'experiment_subdir_template', '{experiment_name}'
    )
    exp_id = config.get('experiment_metadata', {}).get('experiment_id', exp_name)
    try:
        exp_subdir = exp_subdir_template.format(experiment_name=exp_name, experiment_id=exp_id)
    except KeyError:
        logger.warning(
            "Could not format experiment_subdir_template with available metadata."
            " Using experiment_name directly as subdir."
            )
        exp_subdir = exp_name

    output_dir = os.path.join(output_base_dir, exp_subdir)
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, f'{exp_name}.log')
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file_path:
            logging.getLogger().removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)

    logger.info('Starting experiment: %s (ID: %s)', exp_name, exp_id)
    logger.info('Output directory: %s', output_dir)

    random_state = config.get('experiment_metadata', {}).get('random_seed', 42)
    np.random.seed(random_state)

    if 'data' not in config:
        raise ValueError("Missing 'data' section in configuration.")
    x_train, x_test, y_train, y_test = load_and_split_data(
        config['data'], random_state
    )

    if 'model' not in config:
        raise ValueError("Missing 'model' section in configuration.")
    model_config = config['model'] # This is config['model'] dictionary
    model_name = model_config.get('name')
    if not model_name:
        model_class_path = model_config.get('model_class_path', '')
        if model_class_path:
            model_name = model_class_path.split('.')[-1]
        else:
            raise ValueError("Missing 'name' or 'model_class_path' in model configuration.")

    # general_model_args will hold parameters intended for the **kwargs of the model constructor
    # after specific component configurations have been extracted.
    general_model_args = model_config.get('params', model_config.get('model_params', {})).copy()
    
    # Ensure 'device' is handled. It's a common top-level param for AgriTabPFNRegressor.
    # If not in general_model_args (from 'params'), check if it's at the top of model_config.
    if 'device' not in general_model_args:
        general_model_args['device'] = model_config.get('device', 'cpu')


    logger.info(
        'Instantiating model: %s. Initial general_model_args from config: %s', model_name, general_model_args
    )

    if model_name == 'TabPFNClassifier':
        model = TabPFNClassifier(**general_model_args)
    elif model_name == 'TabPFNRegressor':
        model = TabPFNRegressor(**general_model_args)
    elif model_name == 'AgriTabPFNRegressor':
        # AgriTabPFNRegressor expects arguments like 'agri_preprocessor_kwargs'.
        # These dictionaries are sourced from the YAML (where they are also named with '_kwargs').
        
        # 1. Extract the preprocessor configuration dictionary
        # It could be directly under model_config (e.g., config.model.agri_preprocessor_kwargs)
        # OR nested under general_model_args (e.g., config.model.params.agri_preprocessor_kwargs)
        # We prioritize the one directly under model_config if both exist.
        # Then, we ensure it's removed from general_model_args to avoid passing it twice.
        
        # Keys for YAML configuration
        yaml_pp_key = 'agri_preprocessor_kwargs'
        yaml_enc_key = 'agri_encoder_kwargs'
        yaml_tab_key = 'tabpfn_regressor_kwargs'

        # Argument names for AgriTabPFNRegressor constructor
        constructor_pp_arg_name = 'agri_preprocessor_kwargs'
        constructor_enc_arg_name = 'agri_encoder_kwargs'
        constructor_tab_arg_name = 'tabpfn_regressor_kwargs'

        # Get preprocessor_kwargs_dict
        if yaml_pp_key in model_config:
            preprocessor_kwargs_dict = model_config[yaml_pp_key]
            general_model_args.pop(yaml_pp_key, None) # Remove if it was also in params
        else:
            preprocessor_kwargs_dict = general_model_args.pop(yaml_pp_key, {})

        # Get encoder_kwargs_dict
        if yaml_enc_key in model_config:
            encoder_kwargs_dict = model_config[yaml_enc_key]
            general_model_args.pop(yaml_enc_key, None)
        else:
            encoder_kwargs_dict = general_model_args.pop(yaml_enc_key, {})
        
        # Get tabpfn_regressor_kwargs_dict
        if yaml_tab_key in model_config:
            tabpfn_regressor_kwargs_dict = model_config[yaml_tab_key]
            general_model_args.pop(yaml_tab_key, None)
        else:
            tabpfn_regressor_kwargs_dict = general_model_args.pop(yaml_tab_key, {})


        logger.info('Resolved %s for constructor: %s', constructor_pp_arg_name, preprocessor_kwargs_dict)
        logger.info('Resolved %s for constructor: %s', constructor_enc_arg_name, encoder_kwargs_dict)
        logger.info('Resolved %s for constructor: %s', constructor_tab_arg_name, tabpfn_regressor_kwargs_dict)
        logger.info('Remaining general_model_args to be spread for AgriTabPFNRegressor: %s', general_model_args)

        # Pass to AgriTabPFNRegressor using *_kwargs argument names
        model = AgriTabPFNRegressor(
            **{constructor_pp_arg_name: preprocessor_kwargs_dict},
            **{constructor_enc_arg_name: encoder_kwargs_dict},
            **{constructor_tab_arg_name: tabpfn_regressor_kwargs_dict},
            **general_model_args # Pass other general args like device, N_ensemble_configurations, etc.
        )
    elif model_config.get('model_class_path'): # For test mock models
        module_path, class_name = model_config['model_class_path'].rsplit('.',1)
        try:
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            model = model_class(**general_model_args) # Pass all general_model_args to mock model
            logger.info("Instantiated mock model %s from path %s", class_name, module_path)
        except Exception as e:
            logger.error("Failed to instantiate mock model from path %s: %s", model_config['model_class_path'], e, exc_info=True)
            raise
    else:
        raise ValueError(f'Unsupported model_name or configuration: {model_name}')

    logger.info('Starting model training.')
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='No features_standardize set, standardizing features by default',
            category=UserWarning
        )
        warnings.filterwarnings(
            'ignore',
            message=(
                'Using style_string autoencoder for 2 features. This can lead to'
                ' issues.'
            ),
            category=UserWarning,
        )
        model.fit(x_train, y_train)
    training_time = time.time() - start_time
    logger.info('Model training completed in %.2f seconds.', training_time)

    logger.info('Starting model prediction.')
    start_time = time.time()
    y_pred = model.predict(x_test)
    prediction_time = time.time() - start_time
    logger.info('Model prediction completed in %.2f seconds.', prediction_time)

    task_type = config.get('task_type', 'regression')
    if isinstance(model, TabPFNClassifier):
        task_type = 'classification'
    elif isinstance(model, (TabPFNRegressor, AgriTabPFNRegressor)):
        task_type = 'regression'

    logger.info('Evaluating task_type: %s', task_type)
    metrics = {'training_time': training_time, 'prediction_time': prediction_time}
    eval_metrics_config = config.get('evaluation', {}).get('metrics', [])

    if task_type == 'classification':
        if not eval_metrics_config: eval_metrics_config = ['accuracy', 'f1_macro']
        if 'accuracy' in eval_metrics_config:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        if 'f1_macro' in eval_metrics_config:
            metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        if 'f1_micro' in eval_metrics_config:
            metrics['f1_micro'] = f1_score(y_test, y_pred, average='micro')
        if 'f1_weighted' in eval_metrics_config and len(np.unique(y_train)) > 2:
             metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    elif task_type == 'regression':
        if not eval_metrics_config: eval_metrics_config = ['mse', 'mae', 'r2']
        if 'mse' in eval_metrics_config or 'rmse' in eval_metrics_config:
            mse_val = mean_squared_error(y_test, y_pred)
            if 'mse' in eval_metrics_config: metrics['mse'] = mse_val
            if 'rmse' in eval_metrics_config: metrics['rmse'] = np.sqrt(mse_val)
        if 'mae' in eval_metrics_config:
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
        if 'r2' in eval_metrics_config:
            metrics['r2'] = r2_score(y_test, y_pred)
    else:
        logger.warning('Unsupported task_type for evaluation: %s. No metrics calculated.', task_type)

    logger.info('Evaluation metrics: %s', metrics)

    logger.info('Attempting to save results to results.yaml...')
    results_path = os.path.join(output_dir, 'results.yaml')
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            yaml.dump(metrics, f)
        logger.info('Results saved successfully to %s', results_path)
    except Exception as e: # pylint: disable=broad-except
        logger.error("Failed to save results.yaml: %s", e, exc_info=True)

    if config.get('output', {}).get('save_model', False):
        model_path = os.path.join(output_dir, 'model.joblib')
        try:
            import joblib
            joblib.dump(model, model_path)
            logger.info('Model saved to %s', model_path)
        except Exception as e: # pylint: disable=broad-except
            logger.error('Failed to save model: %s', e, exc_info=True)

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    return metrics


def main_cli():
    """Command Line Interface for running experiments."""
    parser = argparse.ArgumentParser(
        description='Run a TabPFN experiment using a YAML configuration file.'
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to the experiment configuration YAML file.',
    )
    args = parser.parse_args()

    try:
        run_single_experiment(args.config_path)
        logger.info('Experiment finished successfully.')
    except Exception as e: # pylint: disable=broad-except
        logger.critical(
            'CRITICAL ERROR in main_cli while running experiment: %s',
            e,
            exc_info=True,
        )
        raise

if __name__ == '__main__':
    main_cli()
