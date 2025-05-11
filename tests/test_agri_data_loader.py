# File: TabPFN/tests/test_agri_data_loader.py
# pylint: disable=line-too-long,protected-access
"""
Unit tests for the agricultural data loader.
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import List, Dict # Added import for List and Dict

# Assuming agri_data_loader is in src.tabpfn.agri_utils
# Adjust the import path based on your project structure
# from tabpfn.agri_utils.data_loader import (
# This is a placeholder, actual import will depend on where the file is placed.
# For now, let's assume it's discoverable or we add its path.
# For testing purposes, if the module is not installed, you might need to add its parent to sys.path
import sys
# This assumes the test is run from the root of the TabPFN project
# or that TabPFN/src is in PYTHONPATH
try:
    from tabpfn.agri_utils.data_loader import (
        load_and_transform_agri_data,
        TEMPORAL_FEATURE_COLS,
        STATIC_SOIL_FEATURE_COLS,
        DEFAULT_CROP_MAPPING,
        GROUP_BY_COLS,
        YIELD_COL,
        DATE_COL,
        N_TEMPORAL_FEATURES,
        N_STATIC_SOIL_FEATURES, # Added N_STATIC_SOIL_FEATURES
        N_STATIC_FEATURES_INC_CROP,
        N_FEATURES_UNIFIED,
        CROP_COL,
        YEAR_COL,
        LONGITUDE_COL,
        LATITUDE_COL
    )
except ImportError:
    # Fallback for environments where the package structure isn't set up perfectly for direct run
    # This is common in some IDEs or ad-hoc test runs.
    # Adjust path as necessary if TabPFN/src is not directly in sys.path
    # For a robust solution, ensure your project is installed (e.g. pip install -e .)
    # or PYTHONPATH is set correctly.
    # For this specific case, let's assume the test is run from the 'TabPFN' directory.
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from tabpfn.agri_utils.data_loader import (
        load_and_transform_agri_data,
        TEMPORAL_FEATURE_COLS,
        STATIC_SOIL_FEATURE_COLS,
        DEFAULT_CROP_MAPPING,
        GROUP_BY_COLS,
        YIELD_COL,
        DATE_COL,
        N_TEMPORAL_FEATURES,
        N_STATIC_SOIL_FEATURES, # Added N_STATIC_SOIL_FEATURES
        N_STATIC_FEATURES_INC_CROP,
        N_FEATURES_UNIFIED,
        CROP_COL,
        YEAR_COL,
        LONGITUDE_COL,
        LATITUDE_COL
    )


class TestAgriDataLoader(unittest.TestCase):
    """Tests for the load_and_transform_agri_data function."""

    def setUp(self):
        """Creates a temporary directory and dummy data for testing."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="agri_test_"))
        self.train_dir = self.test_dir / "train"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Test directory created: {self.test_dir}")

        # Define all feature columns for dummy CSV creation
        self.all_cols_for_csv = [
            DATE_COL, YEAR_COL, CROP_COL, YIELD_COL, LONGITUDE_COL, LATITUDE_COL
        ] + TEMPORAL_FEATURE_COLS + STATIC_SOIL_FEATURE_COLS


    def tearDown(self):
        """Removes the temporary directory after tests."""
        shutil.rmtree(self.test_dir)
        logging.info(f"Test directory removed: {self.test_dir}")

    def _create_dummy_csv(self, file_path: Path, data_rows: List[Dict[str, any]]): # Type hint for Dict values
        """Helper function to create a dummy CSV file."""
        df = pd.DataFrame(data_rows)
        # Ensure all expected columns are present, fill with default if missing from data_rows
        for col in self.all_cols_for_csv:
            if col not in df.columns:
                if col == DATE_COL:
                    df[col] = '2000-01-01' # Default date
                elif col in [YEAR_COL, LONGITUDE_COL, LATITUDE_COL, YIELD_COL]:
                     df[col] = 0 # Default numeric
                elif col == CROP_COL:
                    df[col] = 'unknown'
                else: # Feature columns
                    df[col] = 0.0 # Default feature value
        df.to_csv(file_path, index=False)

    def test_load_and_transform_basic_structure(self):
        """Tests basic data loading and transformation structure."""
        corn_dir = self.train_dir / "corn_unify"
        corn_dir.mkdir(exist_ok=True)

        dummy_data_corn: List[Dict[str, any]] = [] # Type hint for list elements
        # Sample 1: Corn, Year 2020, Loc 1 (-90, 40), 2 time steps
        common_attrs_s1: Dict[str, any] = { # Type hint for dictionary
            YEAR_COL: 2020, CROP_COL: 'corn', YIELD_COL: 150.0,
            LONGITUDE_COL: -90.0, LATITUDE_COL: 40.0
        }
        for i, col_name in enumerate(STATIC_SOIL_FEATURE_COLS):
            common_attrs_s1[col_name] = 10.0 + i
        for day_offset in range(2): # 2 days
            row: Dict[str, any] = {DATE_COL: f'2020-05-0{day_offset+1}', **common_attrs_s1} # Type hint
            for i, col_name in enumerate(TEMPORAL_FEATURE_COLS):
                row[col_name] = (1.0 + i) * (day_offset + 1)
            dummy_data_corn.append(row)

        self._create_dummy_csv(corn_dir / "Illinois.csv", dummy_data_corn)

        X_all, y_all, info_all = load_and_transform_agri_data(self.train_dir)

        self.assertIsInstance(X_all, list)
        self.assertIsInstance(y_all, np.ndarray)
        self.assertIsInstance(info_all, list)

        self.assertEqual(len(X_all), 1)
        self.assertEqual(len(y_all), 1)
        self.assertEqual(len(info_all), 1)

        # Check X for the first (and only) sample
        x_sample = X_all[0]
        self.assertIsInstance(x_sample, np.ndarray)
        self.assertEqual(x_sample.ndim, 2) # (n_features_rows, n_features_cols_unified)
        # n_features_rows = num_time_steps (2) + 1 (static_row) = 3
        self.assertEqual(x_sample.shape[0], 3)
        self.assertEqual(x_sample.shape[1], N_FEATURES_UNIFIED)
        self.assertEqual(x_sample.dtype, np.float32)

        # Check y
        self.assertEqual(y_all[0], 150.0)
        self.assertEqual(y_all.dtype, np.float32)

        # Check info
        expected_info = (2020, 'corn', -90.0, 40.0)
        self.assertEqual(info_all[0], expected_info)

        # Check content of static row (last row)
        static_row_data = x_sample[-1, :]
        # First N_STATIC_SOIL_FEATURES should be soil data
        expected_soil = np.array([10.0 + i for i in range(N_STATIC_SOIL_FEATURES)], dtype=np.float32)
        np.testing.assert_array_almost_equal(static_row_data[:N_STATIC_SOIL_FEATURES], expected_soil)
        # Next should be encoded crop
        self.assertEqual(static_row_data[N_STATIC_SOIL_FEATURES], DEFAULT_CROP_MAPPING['corn'])
        # Rest should be NaN (padding for static features)
        if N_STATIC_FEATURES_INC_CROP < N_FEATURES_UNIFIED:
            self.assertTrue(np.all(np.isnan(static_row_data[N_STATIC_FEATURES_INC_CROP:])))

        # Check content of a temporal row (e.g., first one)
        temporal_row_data = x_sample[0, :]
        expected_temporal_day1 = np.array([(1.0 + i) * 1 for i in range(N_TEMPORAL_FEATURES)], dtype=np.float32)
        np.testing.assert_array_almost_equal(temporal_row_data[:N_TEMPORAL_FEATURES], expected_temporal_day1)
        # Rest should be NaN (padding for temporal features)
        if N_TEMPORAL_FEATURES < N_FEATURES_UNIFIED:
            self.assertTrue(np.all(np.isnan(temporal_row_data[N_TEMPORAL_FEATURES:])))


    def test_multiple_samples_and_crops(self):
        """Tests loading data with multiple samples, crops, and files."""
        corn_dir = self.train_dir / "corn_unify"
        corn_dir.mkdir(exist_ok=True)
        soy_dir = self.train_dir / "soybeans_unify"
        soy_dir.mkdir(exist_ok=True)

        # Corn Sample 1 (from previous test)
        dummy_data_corn1: List[Dict[str, any]] = []
        common_attrs_c1: Dict[str, any] = {YEAR_COL: 2020, CROP_COL: 'corn', YIELD_COL: 150.0, LONGITUDE_COL: -90.0, LATITUDE_COL: 40.0}
        for i in range(len(STATIC_SOIL_FEATURE_COLS)): common_attrs_c1[STATIC_SOIL_FEATURE_COLS[i]] = 10.0 + i
        for day_offset in range(2):
            row: Dict[str, any] = {DATE_COL: f'2020-05-0{day_offset+1}', **common_attrs_c1}
            for i in range(len(TEMPORAL_FEATURE_COLS)): row[TEMPORAL_FEATURE_COLS[i]] = (1.0 + i) * (day_offset + 1)
            dummy_data_corn1.append(row)
        self._create_dummy_csv(corn_dir / "Illinois.csv", dummy_data_corn1)

        # Corn Sample 2 (different year, 1 time step)
        dummy_data_corn2: List[Dict[str, any]] = []
        common_attrs_c2: Dict[str, any] = {YEAR_COL: 2021, CROP_COL: 'corn', YIELD_COL: 160.0, LONGITUDE_COL: -90.0, LATITUDE_COL: 40.0}
        for i in range(len(STATIC_SOIL_FEATURE_COLS)): common_attrs_c2[STATIC_SOIL_FEATURE_COLS[i]] = 11.0 + i # Slightly different soil
        row_c2: Dict[str, any] = {DATE_COL: '2021-06-01', **common_attrs_c2}
        for i in range(len(TEMPORAL_FEATURE_COLS)): row_c2[TEMPORAL_FEATURE_COLS[i]] = (1.5 + i)
        dummy_data_corn2.append(row_c2)
        self._create_dummy_csv(corn_dir / "Ohio.csv", dummy_data_corn2) # Different file

        # Soybeans Sample 1 (3 time steps, one NaN in temporal)
        dummy_data_soy1: List[Dict[str, any]] = []
        common_attrs_s1: Dict[str, any] = {YEAR_COL: 2020, CROP_COL: 'soybeans', YIELD_COL: 50.0, LONGITUDE_COL: -91.0, LATITUDE_COL: 41.0}
        for i in range(len(STATIC_SOIL_FEATURE_COLS)): common_attrs_s1[STATIC_SOIL_FEATURE_COLS[i]] = 20.0 + i
        for day_offset in range(3):
            row: Dict[str, any] = {DATE_COL: f'2020-07-0{day_offset+1}', **common_attrs_s1}
            for i_feat, col_name_feat in enumerate(TEMPORAL_FEATURE_COLS): # Renamed i to i_feat, col_name to col_name_feat
                if i_feat == 0 and day_offset == 1: # Introduce a NaN
                    row[col_name_feat] = np.nan
                else:
                    row[col_name_feat] = (0.5 + i_feat) * (day_offset + 1)
            dummy_data_soy1.append(row)
        self._create_dummy_csv(soy_dir / "Iowa.csv", dummy_data_soy1)

        X_all, y_all, info_all = load_and_transform_agri_data(self.train_dir)

        self.assertEqual(len(X_all), 3) # 3 samples in total
        self.assertEqual(len(y_all), 3)
        self.assertEqual(len(info_all), 3)

        # Check shapes and some values (order might vary due to file iteration, so sort info)
        # To make sorting more robust, convert info tuples to strings for sorting if they contain mixed types
        # or sort by a specific key that is guaranteed to be comparable.
        # Here, sorting by the tuple directly should work if lon/lat are consistently float/int.
        # For more safety, one might sort based on a string representation or specific elements:
        # sorted_indices = np.argsort([f"{info[0]}-{info[1]}-{info[2]}-{info[3]}" for info in info_all])
        # Or if year and crop name are primary sort keys:
        sorted_indices = np.argsort([f"{info[0]}_{info[1]}" for info in info_all])


        # Corn Sample 1 (Year 2020, corn)
        # Find the original index of this sample in the unsorted list
        original_idx_c1 = -1
        for i, info_tuple in enumerate(info_all):
            if info_tuple == (2020, 'corn', -90.0, 40.0):
                original_idx_c1 = i
                break
        self.assertNotEqual(original_idx_c1, -1, "Corn Sample 1 not found in info_all")
        # Use this original_idx_c1 for accessing X_all and y_all
        self.assertEqual(X_all[original_idx_c1].shape, (2 + 1, N_FEATURES_UNIFIED))
        self.assertEqual(y_all[original_idx_c1], 150.0)
        self.assertEqual(X_all[original_idx_c1][-1, N_STATIC_SOIL_FEATURES], DEFAULT_CROP_MAPPING['corn'])


        # Soybeans Sample 1 (Year 2020, soybeans)
        original_idx_s1 = -1
        for i, info_tuple in enumerate(info_all):
            if info_tuple == (2020, 'soybeans', -91.0, 41.0):
                original_idx_s1 = i
                break
        self.assertNotEqual(original_idx_s1, -1, "Soybeans Sample 1 not found in info_all")
        self.assertEqual(X_all[original_idx_s1].shape, (3 + 1, N_FEATURES_UNIFIED))
        self.assertEqual(y_all[original_idx_s1], 50.0)
        self.assertEqual(X_all[original_idx_s1][-1, N_STATIC_SOIL_FEATURES], DEFAULT_CROP_MAPPING['soybeans'])
        self.assertTrue(np.isnan(X_all[original_idx_s1][1, 0]))

        # Corn Sample 2 (Year 2021, corn)
        original_idx_c2 = -1
        for i, info_tuple in enumerate(info_all):
            if info_tuple == (2021, 'corn', -90.0, 40.0):
                original_idx_c2 = i
                break
        self.assertNotEqual(original_idx_c2, -1, "Corn Sample 2 not found in info_all")
        self.assertEqual(X_all[original_idx_c2].shape, (1 + 1, N_FEATURES_UNIFIED))
        self.assertEqual(y_all[original_idx_c2], 160.0)


    def test_empty_data_dir(self):
        """Tests behavior with an empty data directory."""
        empty_dir = self.test_dir / "empty_train"
        empty_dir.mkdir()
        X_all, y_all, info_all = load_and_transform_agri_data(empty_dir)
        self.assertEqual(len(X_all), 0)
        self.assertEqual(len(y_all), 0)
        self.assertEqual(len(info_all), 0)

    def test_dir_with_empty_crop_folders(self):
        """Tests behavior with crop folders that contain no CSVs."""
        corn_dir = self.train_dir / "corn_unify"
        corn_dir.mkdir(exist_ok=True) # Empty crop folder
        X_all, y_all, info_all = load_and_transform_agri_data(self.train_dir)
        self.assertEqual(len(X_all), 0)

    def test_empty_csv_file(self):
        """Tests behavior with an empty CSV file."""
        corn_dir = self.train_dir / "corn_unify"
        corn_dir.mkdir(exist_ok=True)
        with open(corn_dir / "empty.csv", 'w', encoding='utf-8') as f: # Added encoding
            pass # Create empty file
        X_all, y_all, info_all = load_and_transform_agri_data(self.train_dir)
        self.assertEqual(len(X_all), 0)

    def test_csv_with_missing_columns(self):
        """Tests behavior when a CSV is missing critical columns."""
        corn_dir = self.train_dir / "corn_unify"
        corn_dir.mkdir(exist_ok=True)
        # Missing YIELD_COL
        bad_data: List[Dict[str, any]] = [{DATE_COL: '2020-01-01', YEAR_COL: 2020, CROP_COL: 'corn', LONGITUDE_COL: -90, LATITUDE_COL: 40}]
        # Create a dataframe with only these columns, then save.
        df_bad = pd.DataFrame(bad_data)
        # Manually ensure other TEMPORAL_FEATURE_COLS and STATIC_SOIL_FEATURE_COLS are added if not present,
        # otherwise, the loader might skip the file earlier due to missing feature columns.
        # For this test, we are specifically testing the skip due to missing YIELD_COL.
        # The loader checks for GROUP_BY_COLS + [YIELD_COL, DATE_COL] first.
        df_bad.to_csv(corn_dir / "missing_yield.csv", index=False)

        X_all, y_all, info_all = load_and_transform_agri_data(self.train_dir)
        # Expecting this file to be skipped, so 0 samples if it's the only file.
        self.assertEqual(len(X_all), 0)

    def test_unknown_crop_in_csv(self):
        """Tests skipping a group if its crop name is not in the mapping."""
        corn_dir = self.train_dir / "corn_unify" # Folder name is fine
        corn_dir.mkdir(exist_ok=True)

        dummy_data: List[Dict[str, any]] = []
        common_attrs: Dict[str, any] = {
            YEAR_COL: 2020, CROP_COL: 'unknown_crop', YIELD_COL: 100.0, # This crop is not in DEFAULT_CROP_MAPPING
            LONGITUDE_COL: -90.0, LATITUDE_COL: 40.0, DATE_COL: '2020-05-01'
        }
        # Add all required feature columns to avoid being skipped for missing features
        for i, col_name in enumerate(STATIC_SOIL_FEATURE_COLS): common_attrs[col_name] = 1.0 + i
        for i, col_name in enumerate(TEMPORAL_FEATURE_COLS): common_attrs[col_name] = 0.1 + i
        dummy_data.append(common_attrs)

        self._create_dummy_csv(corn_dir / "unknown_crop_file.csv", dummy_data)
        X_all, y_all, info_all = load_and_transform_agri_data(self.train_dir)
        self.assertEqual(len(X_all), 0) # The group with 'unknown_crop' should be skipped.


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    unittest.main()
