# File: TabPFN/tests/test_run_experiment.py
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
# pylint: disable=line-too-long,protected-access,duplicate-code
"""
Unit and integration tests for the run_experiment.py script.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import yaml
import sys
import logging
import time
import json # Added missing import for json
from typing import Union, List, Dict, Any
from unittest.mock import patch, MagicMock

# Ensure src and scripts paths are available
SCRIPT_DIR_TEST_RUN = Path(__file__).resolve().parent
TABPFN_ROOT_TEST_RUN = SCRIPT_DIR_TEST_RUN.parent
SRC_DIR_TEST_RUN = TABPFN_ROOT_TEST_RUN / "src"
SCRIPTS_MODULE_DIR_TEST_RUN = TABPFN_ROOT_TEST_RUN

if SRC_DIR_TEST_RUN.exists() and str(SRC_DIR_TEST_RUN) not in sys.path:
    sys.path.insert(0, str(SRC_DIR_TEST_RUN))
if SCRIPTS_MODULE_DIR_TEST_RUN.exists() and str(SCRIPTS_MODULE_DIR_TEST_RUN) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_MODULE_DIR_TEST_RUN))


try:
    from scripts.run_experiment import run_single_experiment, main_cli
    from tabpfn.agri_utils.experiment_utils import (
        METRICS_FILENAME, PREDICTIONS_FILENAME, MODEL_CHECKPOINT_DIRNAME,
        CONFIG_USED_FILENAME, RUN_LOG_FILENAME
    )
except ImportError as e:
    print(f"Critical ImportError in test_run_experiment.py setup: {e}", file=sys.stderr)
    raise

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
module_logger = logging.getLogger(__name__)


class MockRegressor:
    """A mock regressor for testing purposes."""
    def __init__(self, random_state: Union[int, None] = None, **kwargs: Any):
        self.random_state = random_state
        self.kwargs = kwargs
        self.is_fitted_ = False
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, X: Union[List[np.ndarray], np.ndarray], y: np.ndarray) -> "MockRegressor":
        if isinstance(X, list):
            module_logger.info(f"MockRegressor fitting with List[np.ndarray] of length {len(X)}")
            if X:
                module_logger.info(f"  First item shape: {X[0].shape}")
        elif isinstance(X, np.ndarray):
            module_logger.info(f"MockRegressor fitting with np.ndarray of shape {X.shape}")
        else:
            raise TypeError(f"MockRegressor X type not recognized: {type(X)}")
        module_logger.info(f"MockRegressor fitting with y of shape {y.shape}")
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted yet.")
        if isinstance(X, list):
            num_samples = len(X)
        elif isinstance(X, np.ndarray):
            num_samples = X.shape[0]
        else:
            raise TypeError("X type not recognized for predict")
        
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state + num_samples)
            return rng.rand(num_samples) * 100
        else:
            return np.random.rand(num_samples) * 100


class TestRunExperiment(unittest.TestCase):
    """Tests for the run_experiment.py script."""

    @classmethod
    def setUpClass(cls):
        cls.logger = logging.getLogger(f"{__name__}.{cls.__name__}")
        cls.logger.setLevel(logging.DEBUG)


    def setUp(self):
        self.base_test_dir = Path(tempfile.mkdtemp(prefix="run_exp_base_"))
        self.dummy_data_dir = self.base_test_dir / "dummy_npz_data"
        self.dummy_data_dir.mkdir(parents=True, exist_ok=True)
        self.dummy_configs_dir = self.base_test_dir / "dummy_configs"
        self.dummy_configs_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_outputs_dir = self.base_test_dir / "test_experiment_outputs"
        
        self._create_dummy_npz_data(self.dummy_data_dir / "train_processed.npz", num_samples=10, num_timesteps=5, num_features=3)
        self._create_dummy_npz_data(self.dummy_data_dir / "test_processed.npz", num_samples=5, num_timesteps=5, num_features=3)
        self.logger.info(f"Test directories created in {self.base_test_dir}")

    def tearDown(self):
        shutil.rmtree(self.base_test_dir)
        self.logger.info(f"Test directories removed from {self.base_test_dir}")

    def _create_dummy_npz_data(self, file_path: Path, num_samples: int, num_timesteps: int, num_features: int):
        features_list = []
        for i in range(num_samples):
            temporal_data = np.random.rand(num_timesteps, num_features).astype(np.float32)
            static_data = np.random.rand(1, num_features).astype(np.float32)
            sample_array = np.vstack((temporal_data, static_data))
            features_list.append(sample_array)

        targets = np.random.rand(num_samples).astype(np.float32) * 200
        info_list = [(f"sample_{i}", 2020 + i // 5, -90.0 + i, 40.0 + i) for i in range(num_samples)]

        np.savez_compressed(
            file_path,
            features=np.array(features_list, dtype=object),
            targets=targets,
            info=np.array(info_list, dtype=object)
        )
        self.logger.debug(f"Created dummy NPZ data at {file_path}")


    def _create_dummy_config_yaml(
            self, config_file_path: Path, model_name: str, model_class_path: str,
            model_params: Dict[str, Any] = None, baseline_enabled: bool = False) -> Dict[str, Any]:
        if model_params is None:
            model_params = {}

        timestamp_micros = time.strftime('%H%M%S') + f"_{time.time_ns() // 1000 % 1000000:06}"
        config_content: Dict[str, Any] = {
            "experiment_metadata": {
                "experiment_name": f"TestExp_{model_name}",
                "experiment_id": f"test_{model_name}_{timestamp_micros}",
                "description": "A test experiment.",
                "random_seed": 123
            },
            "data": {
                "train_npz_path": str(self.dummy_data_dir / "train_processed.npz"),
                "test_npz_path": str(self.dummy_data_dir / "test_processed.npz")
            },
            "model": {
                "model_name": model_name,
                "model_class_path": model_class_path,
                "model_params": model_params
            },
            "baseline_preprocessing": {
                "enabled": baseline_enabled,
                "static_row_index": -1,
                "temporal_aggregation": {"methods": ["mean", "std"]}
            },
            "evaluation": {
                "metrics": ["rmse", "mae"],
                "evaluate_on_train_set": True
            },
            "output": {
                "base_results_dir": str(self.experiment_outputs_dir),
                "experiment_subdir_template": "{experiment_id}",
                "save_model": True,
                "save_predictions": True,
                "log_level": "DEBUG"
            }
        }
        with open(config_file_path, "w", encoding="utf-8") as f:
            yaml.dump(config_content, f)
        self.logger.debug(f"Created dummy config file at {config_file_path}")
        return config_content

    @patch(f"{MockRegressor.__module__}.MockRegressor", MockRegressor)
    def test_run_with_mock_sklearn_model(self):
        config_filename = "test_config_mock_sklearn.yaml"
        config_path = self.dummy_configs_dir / config_filename
        mock_model_class_path = f"{MockRegressor.__module__}.MockRegressor"
        config_dict = self._create_dummy_config_yaml(
            config_path,
            model_name="MockRegressor",
            model_class_path=mock_model_class_path,
            model_params={"some_param": 10, "random_state": 123}, 
            baseline_enabled=True 
        )
        results = run_single_experiment(config_path)
        self.assertIsNotNone(results)
        self.assertIn("rmse", results)
        exp_id = config_dict["experiment_metadata"]["experiment_id"]
        expected_output_dir = self.experiment_outputs_dir / exp_id
        self.assertTrue(expected_output_dir.is_dir())
        self.assertTrue((expected_output_dir / METRICS_FILENAME).exists())
        with open(expected_output_dir / METRICS_FILENAME, "r", encoding="utf-8") as f:
            metrics_data = json.load(f) # Ensure json is imported
        self.assertEqual(metrics_data["experiment_id"], exp_id)
        self.assertTrue((expected_output_dir / MODEL_CHECKPOINT_DIRNAME / "MockRegressor_trained.joblib").exists())


    @patch('tabpfn.agri_utils.experiment_utils.dynamic_import_and_instantiate_model')
    def test_run_with_mocked_agritabpfn(self, mock_dynamic_import: MagicMock):
        config_filename = "test_config_mock_agritabpfn.yaml"
        config_path = self.dummy_configs_dir / config_filename
        mock_agri_model_instance = MagicMock(spec=MockRegressor) 
        mock_agri_model_instance.fit.return_value = None
        dummy_test_data = np.load(self.dummy_data_dir / "test_processed.npz", allow_pickle=True)
        num_test_samples = len(dummy_test_data["features"])
        mock_agri_model_instance.predict.return_value = np.random.rand(num_test_samples)
        mock_dynamic_import.return_value = mock_agri_model_instance
        _ = self._create_dummy_config_yaml(
            config_path,
            model_name="AgriTabPFNRegressor",
            model_class_path="tabpfn.agri_tabpfn.AgriTabPFNRegressor", 
            model_params={ 
                "agri_preprocessor_kwargs": {"static_row_index": -1},
                "agri_encoder_kwargs": {"temporal_hidden_dim": 16},
                "tabpfn_regressor_kwargs": {"device": "cpu", "N_ensemble_configurations": 2},
                "random_state": 42
            },
            baseline_enabled=False 
        )
        results = run_single_experiment(config_path)
        self.assertIsNotNone(results)
        mock_dynamic_import.assert_called_once()
        mock_agri_model_instance.fit.assert_called_once()
        fit_args, _ = mock_agri_model_instance.fit.call_args
        self.assertIsInstance(fit_args[0], list) 
        if fit_args[0]: 
            self.assertIsInstance(fit_args[0][0], np.ndarray)

    def test_main_cli_with_mock_experiment(self):
        config_filename = "test_cli_config.yaml"
        config_path = self.dummy_configs_dir / config_filename
        self._create_dummy_config_yaml(
            config_path, "MockCLIModel", f"{MockRegressor.__module__}.MockRegressor", baseline_enabled=True
        )
        test_args = ["run_experiment.py", "--config", str(config_path)]
        with patch.object(sys, 'argv', test_args):
            with patch(f"{MockRegressor.__module__}.MockRegressor", MockRegressor): 
                try:
                    main_cli()
                except SystemExit as e:
                    self.assertIsNone(e.code, f"main_cli exited with code {e.code}") 
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict_cli = yaml.safe_load(f)
                exp_id_cli = config_dict_cli["experiment_metadata"]["experiment_id"]
                expected_output_dir_cli = self.experiment_outputs_dir / exp_id_cli
                self.assertTrue((expected_output_dir_cli / METRICS_FILENAME).exists())

if __name__ == "__main__":
    unittest.main()
