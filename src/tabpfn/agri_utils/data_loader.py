# File: TabPFN/src/tabpfn/agri_utils/data_loader.py
# pylint: disable=line-too-long
"""
Script for loading and transforming agricultural data from CSV files into a 3D format
suitable for AgriTabPFN.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Column Names and Data Structure ---
YEAR_COL = 'year'
CROP_COL = 'crop'
LONGITUDE_COL = 'longitude'
LATITUDE_COL = 'latitude'
YIELD_COL = 'yield'
DATE_COL = 'date'

# Feature columns as per the provided CSV format
TEMPORAL_FEATURE_COLS = [
    'NDVI', 'Wind_Speed_10m_Mean', 'Temperature_Air_2m_Min_24h',
    'Temperature_Air_2m_Max_24h', 'Temperature_Air_2m_Mean_24h',
    'Temperature_Air_2m_Max_Day_Time', 'Temperature_Air_2m_Mean_Day_Time',
    'Temperature_Air_2m_Min_Night_Time', 'Temperature_Air_2m_Mean_Night_Time',
    'Dew_Point_Temperature_2m_Mean', 'Precipitation_Flux',
    'Precipitation_Rain_Duration_Fraction', 'Precipitation_Solid_Duration_Fraction',
    'Snow_Thickness_Mean', 'Snow_Thickness_LWE_Mean', 'Vapour_Pressure_Mean',
    'Solar_Radiation_Flux', 'Cloud_Cover_Mean', 'Relative_Humidity_2m_06h',
    'Relative_Humidity_2m_15h'
]  # 20 temporal features

STATIC_SOIL_FEATURE_COLS = [
    'bdod_bdod_0-5cm_mean', 'bdod_bdod_100-200cm_mean', 'bdod_bdod_15-30cm_mean',
    'bdod_bdod_30-60cm_mean', 'bdod_bdod_5-15cm_mean', 'bdod_bdod_60-100cm_mean',
    'cec_cec_0-5cm_mean', 'cec_cec_100-200cm_mean', 'cec_cec_15-30cm_mean',
    'cec_cec_30-60cm_mean', 'cec_cec_5-15cm_mean', 'cec_cec_60-100cm_mean',
    'cfvo_cfvo_0-5cm_mean', 'cfvo_cfvo_100-200cm_mean', 'cfvo_cfvo_15-30cm_mean',
    'cfvo_cfvo_30-60cm_mean', 'cfvo_cfvo_5-15cm_mean', 'cfvo_cfvo_60-100cm_mean',
    'clay_clay_0-5cm_mean', 'clay_clay_100-200cm_mean', 'clay_clay_15-30cm_mean',
    'clay_clay_30-60cm_mean', 'clay_clay_5-15cm_mean', 'clay_clay_60-100cm_mean',
    'nitrogen_nitrogen_0-5cm_mean', 'nitrogen_nitrogen_100-200cm_mean',
    'nitrogen_nitrogen_15-30cm_mean', 'nitrogen_nitrogen_30-60cm_mean',
    'nitrogen_nitrogen_5-15cm_mean', 'nitrogen_nitrogen_60-100cm_mean',
    'ocd_ocd_0-5cm_mean', 'ocd_ocd_100-200cm_mean', 'ocd_ocd_15-30cm_mean',
    'ocd_ocd_30-60cm_mean', 'ocd_ocd_5-15cm_mean', 'ocd_ocd_60-100cm_mean',
    'ocs_ocs_0-30cm_mean',
    'phh2o_phh2o_0-5cm_mean', 'phh2o_phh2o_100-200cm_mean', 'phh2o_phh2o_15-30cm_mean',
    'phh2o_phh2o_30-60cm_mean', 'phh2o_phh2o_5-15cm_mean', 'phh2o_phh2o_60-100cm_mean',
    'sand_sand_0-5cm_mean', 'sand_sand_100-200cm_mean', 'sand_sand_15-30cm_mean',
    'sand_sand_30-60cm_mean', 'sand_sand_5-15cm_mean', 'sand_sand_60-100cm_mean',
    'silt_silt_0-5cm_mean', 'silt_silt_100-200cm_mean', 'silt_silt_15-30cm_mean',
    'silt_silt_30-60cm_mean', 'silt_silt_5-15cm_mean', 'silt_silt_60-100cm_mean',
    'soc_soc_0-5cm_mean', 'soc_soc_100-200cm_mean', 'soc_soc_15-30cm_mean',
    'soc_soc_30-60cm_mean', 'soc_soc_5-15cm_mean', 'soc_soc_60-100cm_mean'
]  # 61 soil features

# Default crop name to integer mapping.
# These keys should match the values in the 'crop' column of the CSV files.
DEFAULT_CROP_MAPPING = {
    'corn': 0,
    'rice': 1,
    'soybeans': 2,
    'wheat_spring_excl_drum': 3, # Assuming this is how it appears in CSV
    'wheat_winter': 4            # Assuming this is how it appears in CSV
}

# Columns for grouping individual samples
GROUP_BY_COLS = [YEAR_COL, CROP_COL, LONGITUDE_COL, LATITUDE_COL]

# Calculated feature counts
N_TEMPORAL_FEATURES = len(TEMPORAL_FEATURE_COLS)
N_STATIC_SOIL_FEATURES = len(STATIC_SOIL_FEATURE_COLS)
N_STATIC_FEATURES_INC_CROP = N_STATIC_SOIL_FEATURES + 1  # +1 for encoded crop
N_FEATURES_UNIFIED = max(N_TEMPORAL_FEATURES, N_STATIC_FEATURES_INC_CROP)


def load_and_transform_agri_data(
    data_dir: Union[str, Path],
    crop_mapping: Dict[str, int] = None
) -> Tuple[List[np.ndarray], np.ndarray, List[Tuple[Any, ...]]]:
    """
    Loads agricultural data from CSV files within a directory structure,
    transforms it into 3D arrays per sample, and returns features, targets,
    and sample identifiers.

    The directory structure is expected to be:
    data_dir/
        crop_type_unify/  (e.g., corn_unify)
            state_name.csv (e.g., Illinois.csv)

    Each 3D array for a sample will have shape:
    (n_time_steps + 1, N_FEATURES_UNIFIED), where the last row is static data.

    Args:
        data_dir: Path to the root directory containing crop subdirectories.
        crop_mapping: A dictionary mapping crop names (from CSV 'crop' column)
                      to integer codes. If None, uses DEFAULT_CROP_MAPPING.

    Returns:
        A tuple containing:
        - X_all_samples (List[np.ndarray]): A list of 3D NumPy arrays, where
          each array represents a sample (n_features_rows, n_features_cols_unified).
          n_features_rows = n_time_steps_for_sample + 1 (static row).
        - y_all_samples (np.ndarray): A 1D NumPy array of yield values (targets).
        - sample_info_list (List[Tuple[Any, ...]]): A list of tuples, where
          each tuple contains the group identifiers (year, crop_name, lon, lat)
          for the corresponding sample.
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir_path}")

    if crop_mapping is None:
        crop_mapping_to_use = DEFAULT_CROP_MAPPING
    else:
        crop_mapping_to_use = crop_mapping

    X_all_samples: List[np.ndarray] = []
    y_list: List[float] = []
    sample_info_list: List[Tuple[Any, ...]] = []

    crop_folders = [d for d in data_dir_path.iterdir() if d.is_dir()]
    if not crop_folders:
        logging.warning(f"No crop subdirectories found in {data_dir_path}")
        return [], np.array([]), []

    for crop_folder in crop_folders:
        # Example: crop_folder.name could be 'corn_unify'
        # We rely on the 'crop' column in the CSV for the actual crop name to map
        logging.info(f"Processing crop folder: {crop_folder.name}")
        csv_files = list(crop_folder.glob('*.csv'))
        if not csv_files:
            logging.warning(f"No CSV files found in {crop_folder}")
            continue

        for csv_file in csv_files:
            logging.info(f"  Loading CSV file: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
            except pd.errors.EmptyDataError:
                logging.warning(f"    CSV file is empty: {csv_file.name}")
                continue
            except Exception as e:
                logging.error(f"    Error reading CSV {csv_file.name}: {e}")
                continue

            # Validate required columns
            required_cols_for_grouping = GROUP_BY_COLS + [YIELD_COL, DATE_COL]
            missing_group_cols = [c for c in required_cols_for_grouping if c not in df.columns]
            if missing_group_cols:
                logging.warning(f"    Missing required columns for grouping/target in {csv_file.name}: {missing_group_cols}. Skipping file.")
                continue

            missing_temporal_cols = [c for c in TEMPORAL_FEATURE_COLS if c not in df.columns]
            if missing_temporal_cols:
                logging.warning(f"    Missing temporal feature columns in {csv_file.name}: {missing_temporal_cols}. Skipping file.")
                continue

            missing_static_cols = [c for c in STATIC_SOIL_FEATURE_COLS if c not in df.columns]
            if missing_static_cols:
                logging.warning(f"    Missing static soil feature columns in {csv_file.name}: {missing_static_cols}. Skipping file.")
                continue

            # Convert date column to datetime
            try:
                df[DATE_COL] = pd.to_datetime(df[DATE_COL])
            except Exception as e:
                logging.warning(f"    Could not parse date column in {csv_file.name}: {e}. Skipping file.")
                continue

            # Group by the defined columns
            grouped = df.groupby(GROUP_BY_COLS)

            for group_keys, group_df in grouped:
                year, crop_name, lon, lat = group_keys

                if not isinstance(crop_name, str) or crop_name not in crop_mapping_to_use:
                    logging.warning(f"    Crop name '{crop_name}' in group {group_keys} from {csv_file.name} not in crop_mapping. Skipping group.")
                    continue

                # Sort by date to ensure correct temporal order
                group_df_sorted = group_df.sort_values(by=DATE_COL)
                num_time_steps = len(group_df_sorted)

                if num_time_steps == 0:
                    logging.warning(f"    Group {group_keys} in {csv_file.name} has no time steps. Skipping group.")
                    continue

                # --- Extract Temporal Features ---
                temporal_features_data = group_df_sorted[TEMPORAL_FEATURE_COLS].values.astype(np.float32)

                # --- Extract Static Features ---
                # Soil features (should be constant within a group, take from first row)
                static_soil_data = group_df_sorted[STATIC_SOIL_FEATURE_COLS].iloc[0].values.astype(np.float32)
                encoded_crop = np.array([crop_mapping_to_use[crop_name]], dtype=np.float32)
                static_features_combined = np.concatenate([static_soil_data, encoded_crop])

                # --- Pad features to N_FEATURES_UNIFIED ---
                # Pad temporal features
                padded_temporal_data = np.full((num_time_steps, N_FEATURES_UNIFIED), np.nan, dtype=np.float32)
                padded_temporal_data[:, :N_TEMPORAL_FEATURES] = temporal_features_data

                # Pad static features
                padded_static_row = np.full(N_FEATURES_UNIFIED, np.nan, dtype=np.float32)
                padded_static_row[:N_STATIC_FEATURES_INC_CROP] = static_features_combined

                # --- Combine into 3D array for the sample ---
                # Static features form the last row
                sample_3d_array = np.vstack([padded_temporal_data, padded_static_row.reshape(1, -1)])

                X_all_samples.append(sample_3d_array)
                y_list.append(group_df_sorted[YIELD_COL].iloc[0]) # Yield should be constant for the group
                sample_info_list.append(group_keys) # (year, crop_name, lon, lat)

    if not X_all_samples:
        logging.warning("No valid samples were processed from the data directory.")

    return X_all_samples, np.array(y_list, dtype=np.float32), sample_info_list

if __name__ == '__main__':
    # Example usage (assuming your data is in a 'data/train' directory)
    # Create dummy data for demonstration if run directly
    dummy_data_dir = Path('dummy_agri_data_main/train')
    dummy_data_dir.mkdir(parents=True, exist_ok=True)

    corn_dir = dummy_data_dir / 'corn_unify'
    corn_dir.mkdir(exist_ok=True)
    soy_dir = dummy_data_dir / 'soybeans_unify'
    soy_dir.mkdir(exist_ok=True)

    # Define all feature columns for dummy CSV
    all_feature_cols = [DATE_COL, YEAR_COL, CROP_COL, YIELD_COL, LONGITUDE_COL, LATITUDE_COL] + \
                       TEMPORAL_FEATURE_COLS + STATIC_SOIL_FEATURE_COLS

    dummy_corn_data = []
    # Sample 1: Corn, Year 2020, Loc 1, 2 time steps
    common_corn_1 = {YEAR_COL: 2020, CROP_COL: 'corn', YIELD_COL: 150.0, LONGITUDE_COL: -90.0, LATITUDE_COL: 40.0}
    for i in range(len(STATIC_SOIL_FEATURE_COLS)): common_corn_1[STATIC_SOIL_FEATURE_COLS[i]] = 10.0 + i
    for day in range(1, 3): # 2 days
        row = {DATE_COL: f'2020-05-0{day}', **common_corn_1}
        for i in range(len(TEMPORAL_FEATURE_COLS)): row[TEMPORAL_FEATURE_COLS[i]] = (1.0 + i) * day
        dummy_corn_data.append(row)

    # Sample 2: Corn, Year 2021, Loc 1, 1 time step
    common_corn_2 = {YEAR_COL: 2021, CROP_COL: 'corn', YIELD_COL: 160.0, LONGITUDE_COL: -90.0, LATITUDE_COL: 40.0}
    for i in range(len(STATIC_SOIL_FEATURE_COLS)): common_corn_2[STATIC_SOIL_FEATURE_COLS[i]] = 15.0 + i
    row2 = {DATE_COL: '2021-06-01', **common_corn_2}
    for i in range(len(TEMPORAL_FEATURE_COLS)): row2[TEMPORAL_FEATURE_COLS[i]] = (2.0 + i)
    dummy_corn_data.append(row2)

    pd.DataFrame(dummy_corn_data).to_csv(corn_dir / 'Illinois.csv', index=False)

    dummy_soy_data = []
    # Sample 3: Soy, Year 2020, Loc 2, 3 time steps
    common_soy_1 = {YEAR_COL: 2020, CROP_COL: 'soybeans', YIELD_COL: 50.0, LONGITUDE_COL: -91.0, LATITUDE_COL: 41.0}
    for i in range(len(STATIC_SOIL_FEATURE_COLS)): common_soy_1[STATIC_SOIL_FEATURE_COLS[i]] = 20.0 + i
    for day in range(1, 4): # 3 days
        row = {DATE_COL: f'2020-07-0{day}', **common_soy_1}
        for i in range(len(TEMPORAL_FEATURE_COLS)): row[TEMPORAL_FEATURE_COLS[i]] = (0.5 + i) * day
        if TEMPORAL_FEATURE_COLS[0] in row : row[TEMPORAL_FEATURE_COLS[0]] = np.nan # Introduce a NaN
        dummy_soy_data.append(row)
    pd.DataFrame(dummy_soy_data).to_csv(soy_dir / 'Iowa.csv', index=False)


    logging.info(f"Attempting to load data from: {dummy_data_dir.resolve()}")
    x_data, y_data, info = load_and_transform_agri_data(dummy_data_dir)

    logging.info(f"Loaded {len(x_data)} samples.")
    if x_data:
        logging.info(f"First sample X shape: {x_data[0].shape}")
        logging.info(f"First sample y: {y_data[0]}")
        logging.info(f"First sample info: {info[0]}")
        logging.info(f"Static row of first sample (last row):\n{x_data[0][-1, :N_STATIC_FEATURES_INC_CROP]}")
        logging.info(f"A temporal row of first sample:\n{x_data[0][0, :N_TEMPORAL_FEATURES]}")

    # Clean up dummy directory
    import shutil
    shutil.rmtree('dummy_agri_data_main')
    logging.info("Cleaned up dummy data directory.")
