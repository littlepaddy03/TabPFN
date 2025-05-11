# File: TabPFN/tests/test_preprocess_agri_datasets.py
# pylint: disable=line-too-long,protected-access
"""
Unit tests for the preprocess_agri_datasets.py script.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import sys
from typing import List, Dict, Any

# Ensure the agri_utils and scripts module can be found
SCRIPT_DIR_TEST = Path(__file__).resolve().parent
TABPFN_ROOT_TEST = SCRIPT_DIR_TEST.parent
SRC_DIR_TEST = TABPFN_ROOT_TEST / "src"
SCRIPTS_MODULE_DIR_TEST = TABPFN_ROOT_TEST # Assuming scripts is directly under TabPFN

# Add src and the directory containing 'scripts' to sys.path if not already there
if SRC_DIR_TEST.exists() and str(SRC_DIR_TEST) not in sys.path:
    sys.path.insert(0, str(SRC_DIR_TEST))
if SCRIPTS_MODULE_DIR_TEST.exists() and str(SCRIPTS_MODULE_DIR_TEST) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_MODULE_DIR_TEST))


try:
    from scripts.preprocess_agri_datasets import main as preprocess_main
    from scripts.preprocess_agri_datasets import save_processed_data # For direct testing if needed
    from tabpfn.agri_utils.data_loader import (
        TEMPORAL_FEATURE_COLS, STATIC_SOIL_FEATURE_COLS, DEFAULT_CROP_MAPPING,
        DATE_COL, YEAR_COL, CROP_COL, YIELD_COL, LONGITUDE_COL, LATITUDE_COL
    )
except ImportError as e:
    logging.error(f"ImportError in test setup: {e}. Check sys.path and script location.")
    # If you still have issues, print sys.path here to debug
    # logging.error(f"Current sys.path: {sys.path}")
    raise

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestPreprocessAgriDatasets(unittest.TestCase):
    """Tests for the preprocess_agri_datasets.py script."""

    def setUp(self):
        """Creates temporary directories and dummy data for testing."""
        self.base_test_dir = Path(tempfile.mkdtemp(prefix="preprocess_base_"))
        self.input_data_dir = self.base_test_dir / "input_data"
        self.output_data_dir = self.base_test_dir / "output_processed"

        self.train_input_dir = self.input_data_dir / "train"
        self.test_input_dir = self.input_data_dir / "test"

        self.train_input_dir.mkdir(parents=True, exist_ok=True)
        self.test_input_dir.mkdir(parents=True, exist_ok=True)
        self.output_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Test input directory created: {self.input_data_dir}")
        logger.info(f"Test output directory created: {self.output_data_dir}")

        # Columns for dummy CSV
        self.all_cols_for_csv = [
            DATE_COL, YEAR_COL, CROP_COL, YIELD_COL, LONGITUDE_COL, LATITUDE_COL
        ] + TEMPORAL_FEATURE_COLS + STATIC_SOIL_FEATURE_COLS

    def tearDown(self):
        """Removes the temporary directories after tests."""
        shutil.rmtree(self.base_test_dir)
        logger.info(f"Base test directory removed: {self.base_test_dir}")

    def _create_dummy_csv_file(self, file_path: Path, data_rows: List[Dict[str, Any]]):
        """Helper to create a dummy CSV with all expected columns."""
        df = pd.DataFrame(data_rows)
        for col in self.all_cols_for_csv:
            if col not in df.columns:
                # Add default values for any missing columns to ensure CSV is valid
                if col == DATE_COL: df[col] = '2000-01-01'
                elif col in [YEAR_COL, LONGITUDE_COL, LATITUDE_COL, YIELD_COL]: df[col] = 0
                elif col == CROP_COL: df[col] = list(DEFAULT_CROP_MAPPING.keys())[0] # Default crop
                else: df[col] = 0.0 # Default feature value
        df.to_csv(file_path, index=False)


    def _populate_dummy_data(self, target_dir: Path, num_samples: int, crop_name: str, start_year: int, yield_base: float):
        """Populates a directory with dummy CSV data for a given crop."""
        crop_data_dir = target_dir / f"{crop_name}_unify"
        crop_data_dir.mkdir(exist_ok=True)

        data_for_csv: List[Dict[str, Any]] = []
        for i in range(num_samples):
            current_year = start_year + i
            current_lon = -90.0 + i
            current_lat = 40.0 + i
            current_yield = yield_base + i * 10

            common_attrs: Dict[str, Any] = {
                YEAR_COL: current_year, CROP_COL: crop_name, YIELD_COL: current_yield,
                LONGITUDE_COL: current_lon, LATITUDE_COL: current_lat
            }
            for soil_idx, soil_col in enumerate(STATIC_SOIL_FEATURE_COLS):
                common_attrs[soil_col] = 10.0 + soil_idx + i # Vary soil data slightly per sample

            # Create 2 time steps per sample
            for day_offset in range(2):
                row: Dict[str, Any] = {
                    DATE_COL: f'{current_year}-05-0{day_offset+1}',
                    **common_attrs
                }
                for temp_idx, temp_col in enumerate(TEMPORAL_FEATURE_COLS):
                    row[temp_col] = (1.0 + temp_idx + i) * (day_offset + 1)
                data_for_csv.append(row)
        
        self._create_dummy_csv_file(crop_data_dir / f"State{crop_name.capitalize()}.csv", data_for_csv)


    def test_script_execution_and_output(self):
        """Tests the main script execution, data loading, and saving."""
        # Populate dummy data
        self._populate_dummy_data(self.train_input_dir, num_samples=2, crop_name='corn', start_year=2020, yield_base=150)
        self._populate_dummy_data(self.train_input_dir, num_samples=1, crop_name='soybeans', start_year=2020, yield_base=50)
        self._populate_dummy_data(self.test_input_dir, num_samples=1, crop_name='corn', start_year=2022, yield_base=170)

        # Create a mock args object
        class MockArgs:
            base_data_dir = str(self.input_data_dir)
            output_dir = str(self.output_data_dir)

        # Override sys.argv for the duration of the test if preprocess_main uses it directly
        original_argv = sys.argv
        sys.argv = [
            "preprocess_agri_datasets.py",
            f"--base_data_dir={self.input_data_dir}",
            f"--output_dir={self.output_data_dir}"
        ]
        
        try:
            X_train, y_train, train_info, X_test, y_test, test_info = preprocess_main()
        finally:
            sys.argv = original_argv # Restore original argv

        # --- Verify loaded data directly from preprocess_main return values ---
        self.assertEqual(len(X_train), 3) # 2 corn + 1 soybean samples
        self.assertEqual(len(y_train), 3)
        self.assertEqual(len(train_info), 3)

        self.assertEqual(len(X_test), 1) # 1 corn sample
        self.assertEqual(len(y_test), 1)
        self.assertEqual(len(test_info), 1)

        self.assertIsInstance(X_train[0], np.ndarray)
        self.assertEqual(X_train[0].shape[0], 2 + 1) # 2 time steps + 1 static row

        # --- Verify saved files ---
        train_npz_path = self.output_data_dir / "train_processed.npz"
        test_npz_path = self.output_data_dir / "test_processed.npz"

        self.assertTrue(train_npz_path.exists())
        self.assertTrue(test_npz_path.exists())

        # Load and verify train data from NPZ
        train_loaded_data = np.load(train_npz_path, allow_pickle=True)
        self.assertIn("features", train_loaded_data)
        self.assertIn("targets", train_loaded_data)
        self.assertIn("info", train_loaded_data)

        loaded_X_train = train_loaded_data["features"]
        loaded_y_train = train_loaded_data["targets"]
        # loaded_train_info = train_loaded_data["info"] # Not strictly checking info content here

        self.assertEqual(len(loaded_X_train), 3)
        self.assertEqual(len(loaded_y_train), 3)
        self.assertIsInstance(loaded_X_train[0], np.ndarray) # Check if elements are arrays
        np.testing.assert_array_equal(y_train, loaded_y_train)


        # Load and verify test data from NPZ
        test_loaded_data = np.load(test_npz_path, allow_pickle=True)
        self.assertIn("features", test_loaded_data)
        self.assertIn("targets", test_loaded_data)
        self.assertIn("info", test_loaded_data)

        loaded_X_test = test_loaded_data["features"]
        loaded_y_test = test_loaded_data["targets"]

        self.assertEqual(len(loaded_X_test), 1)
        self.assertEqual(len(loaded_y_test), 1)
        self.assertIsInstance(loaded_X_test[0], np.ndarray)
        np.testing.assert_array_equal(y_test, loaded_y_test)

    def test_script_execution_no_output_dir(self):
        """Tests script execution when no output directory is provided."""
        self._populate_dummy_data(self.train_input_dir, num_samples=1, crop_name='corn', start_year=2020, yield_base=150)
        self._populate_dummy_data(self.test_input_dir, num_samples=1, crop_name='corn', start_year=2022, yield_base=170)

        original_argv = sys.argv
        sys.argv = [
            "preprocess_agri_datasets.py",
            f"--base_data_dir={self.input_data_dir}"
            # No --output_dir
        ]
        
        try:
            X_train, y_train, _, X_test, y_test, _ = preprocess_main()
        finally:
            sys.argv = original_argv

        self.assertEqual(len(X_train), 1)
        self.assertEqual(len(y_train), 1)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_test), 1)

        # Ensure no files were created in the default output (if any) or current dir
        self.assertFalse((self.output_data_dir / "train_processed.npz").exists())
        self.assertFalse((self.output_data_dir / "test_processed.npz").exists())
        # Also check current directory just in case
        self.assertFalse((Path(".") / "train_processed.npz").exists())


if __name__ == "__main__":
    unittest.main()
