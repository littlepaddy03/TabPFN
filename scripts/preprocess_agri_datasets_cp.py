# -*- coding: utf-8 -*-
"""
Script to preprocess agricultural datasets for TabPFN adaptation.

This script reads CSV data from a structured directory (train/test subfolders),
processes each sample group (unique location/year/crop), pads/truncates
temporal sequences, combines with static features, and saves the processed data
as separate train and test .npz files.

Key changes:
- Takes a base_data_dir as input instead of a single CSV.
- Walks through train/ and test/ subdirectories to find and combine all CSVs.
- Processes and saves train_processed.npz and test_processed.npz separately.
- Removes internal train/test splitting logic.
- Corrected DEFAULT_TEMPORAL_COLS to match sample CSV header.
- Expanded DEFAULT_STATIC_COLS to include ocd_ and ocs_ columns from sample CSV.
- Added memory optimization by deleting intermediate list after stacking.
"""
import argparse
import os
import logging
from typing import List, Dict, Tuple, Any, Optional
import gc # For garbage collection

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define default column names - adjust if your CSV headers are different
DEFAULT_TEMPORAL_COLS = [
    'NDVI',
    'Wind_Speed_10m_Mean',
    'Temperature_Air_2m_Min_24h',
    'Temperature_Air_2m_Max_24h',
    'Temperature_Air_2m_Mean_24h',
    'Temperature_Air_2m_Max_Day_Time',
    'Temperature_Air_2m_Mean_Day_Time',
    'Temperature_Air_2m_Min_Night_Time',
    'Temperature_Air_2m_Mean_Night_Time',
    'Dew_Point_Temperature_2m_Mean',
    'Precipitation_Flux',
    'Precipitation_Rain_Duration_Fraction',
    'Precipitation_Solid_Duration_Fraction',
    'Snow_Thickness_Mean',
    'Snow_Thickness_LWE_Mean',
    'Vapour_Pressure_Mean',
    'Solar_Radiation_Flux',
    'Cloud_Cover_Mean',
    'Relative_Humidity_2m_06h',
    'Relative_Humidity_2m_15h'
]

DEFAULT_STATIC_COLS = [
    'bdod_bdod_0-5cm_mean', 'bdod_bdod_100-200cm_mean',
    'bdod_bdod_15-30cm_mean', 'bdod_bdod_30-60cm_mean',
    'bdod_bdod_5-15cm_mean', 'bdod_bdod_60-100cm_mean',
    'cec_cec_0-5cm_mean', 'cec_cec_100-200cm_mean',
    'cec_cec_15-30cm_mean', 'cec_cec_30-60cm_mean',
    'cec_cec_5-15cm_mean', 'cec_cec_60-100cm_mean',
    'cfvo_cfvo_0-5cm_mean', 'cfvo_cfvo_100-200cm_mean',
    'cfvo_cfvo_15-30cm_mean', 'cfvo_cfvo_30-60cm_mean',
    'cfvo_cfvo_5-15cm_mean', 'cfvo_cfvo_60-100cm_mean',
    'clay_clay_0-5cm_mean', 'clay_clay_100-200cm_mean',
    'clay_clay_15-30cm_mean', 'clay_clay_30-60cm_mean',
    'clay_clay_5-15cm_mean', 'clay_clay_60-100cm_mean',
    'nitrogen_nitrogen_0-5cm_mean', 'nitrogen_nitrogen_100-200cm_mean',
    'nitrogen_nitrogen_15-30cm_mean', 'nitrogen_nitrogen_30-60cm_mean',
    'nitrogen_nitrogen_5-15cm_mean', 'nitrogen_nitrogen_60-100cm_mean',
    'ocd_ocd_0-5cm_mean', 'ocd_ocd_100-200cm_mean', 
    'ocd_ocd_15-30cm_mean', 'ocd_ocd_30-60cm_mean',
    'ocd_ocd_5-15cm_mean', 'ocd_ocd_60-100cm_mean',
    'ocs_ocs_0-30cm_mean', 
    'phh2o_phh2o_0-5cm_mean', 'phh2o_phh2o_100-200cm_mean',
    'phh2o_phh2o_15-30cm_mean', 'phh2o_phh2o_30-60cm_mean',
    'phh2o_phh2o_5-15cm_mean', 'phh2o_phh2o_60-100cm_mean',
    'sand_sand_0-5cm_mean', 'sand_sand_100-200cm_mean',
    'sand_sand_15-30cm_mean', 'sand_sand_30-60cm_mean',
    'sand_sand_5-15cm_mean', 'sand_sand_60-100cm_mean',
    'silt_silt_0-5cm_mean', 'silt_silt_100-200cm_mean',
    'silt_silt_15-30cm_mean', 'silt_silt_30-60cm_mean',
    'silt_silt_5-15cm_mean', 'silt_silt_60-100cm_mean',
    'soc_soc_0-5cm_mean', 'soc_soc_100-200cm_mean',
    'soc_soc_15-30cm_mean', 'soc_soc_30-60cm_mean',
    'soc_soc_5-15cm_mean', 'soc_soc_60-100cm_mean'
]
DEFAULT_TARGET_COL = 'yield'
DEFAULT_GROUP_COLS = ['year', 'crop', 'longitude', 'latitude']
DEFAULT_TIME_COL = 'date'


def load_and_combine_csvs_from_dir(directory_path: str) -> Optional[pd.DataFrame]:
    """
    Walks through a directory, loads all CSV files, and concatenates them.
    """
    all_dfs: List[pd.DataFrame] = []
    logger.info(f"Scanning for CSV files in directory: {directory_path}")
    if not os.path.isdir(directory_path):
        logger.error(f"Provided path is not a directory: {directory_path}")
        return None

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                logger.info(f"Reading CSV file: {file_path}")
                try:
                    df_temp = pd.read_csv(file_path)
                    all_dfs.append(df_temp)
                except Exception as e:
                    logger.error(f"Error reading CSV file {file_path}: {e}")

    if not all_dfs:
        logger.warning(f"No CSV files found or successfully read in {directory_path}")
        return None

    logger.info(f"Concatenating {len(all_dfs)} DataFrames from {directory_path}")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")
    return combined_df


def process_sample_group(
    group_df: pd.DataFrame,
    temporal_cols: List[str],
    static_cols: List[str],
    target_col: str,
    max_len: int
) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
    """
    Processes a single sample group.
    """
    if group_df.empty:
        return None

    temp_df_copy = group_df.copy() 
    for col in temporal_cols:
        if col not in temp_df_copy.columns:
            temp_df_copy.loc[:, col] = np.nan
            
    temporal_data_df = temp_df_copy[temporal_cols]
    # Ensure consistent dtype before converting to numpy, especially if all are NaNs
    for col in temporal_data_df.columns:
        if temporal_data_df[col].isnull().all():
            temporal_data_df[col] = temporal_data_df[col].astype(np.float32)
            
    temporal_data_np = temporal_data_df.values.astype(np.float32).T 

    n_timesteps_in_group = temporal_data_np.shape[1]

    if n_timesteps_in_group > max_len:
        temporal_data_np = temporal_data_np[:, :max_len]
    elif n_timesteps_in_group < max_len:
        padding_shape = (temporal_data_np.shape[0], max_len - n_timesteps_in_group)
        padding = np.full(padding_shape, np.nan, dtype=np.float32) 
        temporal_data_np = np.concatenate([temporal_data_np, padding], axis=1)

    for col in static_cols:
        if col not in temp_df_copy.columns:
            temp_df_copy.loc[:, col] = np.nan

    static_data_series = temp_df_copy[static_cols].iloc[0] 
    # Ensure consistent dtype for static features as well
    static_data_np = static_data_series.astype(np.float32).values
    
    static_data_expanded = np.repeat(static_data_np[:, np.newaxis], max_len, axis=1) 

    feature_array = np.vstack([temporal_data_np, static_data_expanded]) 

    target_value = temp_df_copy[target_col].iloc[0]
    if pd.isna(target_value):
        return None 

    sample_meta_info = {key: temp_df_copy[key].iloc[0] for key in DEFAULT_GROUP_COLS if key in temp_df_copy}
    sample_meta_info['original_timesteps'] = n_timesteps_in_group
    
    return feature_array, float(target_value), sample_meta_info


def process_dataframe_and_save_npz(
    df: pd.DataFrame,
    output_filepath: str,
    args: argparse.Namespace,
    temporal_cols: List[str],
    static_cols: List[str],
    target_col: str,
    group_cols: List[str],
    split_name: str
):
    """
    Processes a combined DataFrame and saves the result to an NPZ file.
    """
    if df is None or df.empty:
        logger.warning(f"DataFrame for {split_name} is empty or None. Skipping NPZ generation for {output_filepath}")
        return

    logger.info(f"Processing combined DataFrame for {split_name} data. Shape: {df.shape}")

    if args.time_col and args.time_col in df.columns:
        logger.info(f"Sorting {split_name} data by group columns and time column: {args.time_col}")
        try:
            df[args.time_col] = pd.to_datetime(df[args.time_col])
            df = df.sort_values(by=group_cols + [args.time_col])
        except Exception as e:
            logger.warning(f"Could not parse time column '{args.time_col}' as datetime or sort for {split_name} data: {e}.")
    else:
        logger.info(f"No time column provided or found for sorting {split_name} data within groups.")

    grouped = df.groupby(group_cols, sort=False) 
    logger.info(f"Number of unique sample groups found in {split_name} data: {len(grouped)}")

    processed_features_list: List[np.ndarray] = []
    processed_targets_list: List[float] = []

    expected_num_feature_rows = len(temporal_cols) + len(static_cols)
    expected_feature_shape = (expected_num_feature_rows, args.max_len)
    logger.info(f"Expected shape for each sample's 2D feature array in {split_name} data: {expected_feature_shape}")

    processed_count = 0
    skipped_count = 0
    for i, (group_name, group_df) in enumerate(grouped):
        if i % 10000 == 0 and i > 0: 
            logger.info(f"Processing group {i}/{len(grouped)} for {split_name} data...")

        processed_sample = process_sample_group(
            group_df, temporal_cols, static_cols, target_col, args.max_len
        )

        if processed_sample:
            feature_array, target_value, _ = processed_sample
            if feature_array.shape != expected_feature_shape:
                skipped_count += 1
                continue
            processed_features_list.append(feature_array)
            processed_targets_list.append(target_value)
            processed_count +=1
        else:
            skipped_count += 1
    
    logger.info(f"For {split_name} data: Processed {processed_count} samples, skipped {skipped_count} samples due to errors, NaN targets, or shape inconsistencies.")

    if not processed_features_list:
        logger.error(f"No samples were successfully processed for {split_name} data. NPZ file {output_filepath} will not be generated.")
        return

    try:
        all_features_np = np.stack(processed_features_list, axis=0).astype(np.float32)
    except ValueError as e:
        logger.error(f"Failed to stack feature arrays for {split_name} data: {e}")
        logger.error("This usually means that despite checks, some feature_arrays in processed_features_list "
                     "still have inconsistent shapes. This should have been caught by the shape check earlier.")
        return
    
    # --- Memory optimization: Delete list after stacking ---
    logger.info(f"Successfully stacked features for {split_name} data. Deleting intermediate list to free memory.")
    del processed_features_list
    gc.collect() # Trigger garbage collection
    logger.info("Intermediate feature list deleted and garbage collected.")
    # --- End memory optimization ---
        
    all_targets_np = np.array(processed_targets_list, dtype=np.float32)
    # Optionally, delete processed_targets_list too if memory is extremely tight, though it's much smaller
    # del processed_targets_list
    # gc.collect()


    logger.info(f"Final processed features shape for {split_name} data: {all_features_np.shape}")
    logger.info(f"Final processed targets shape for {split_name} data: {all_targets_np.shape}")

    static_feature_row_index = len(temporal_cols)
    info_dict = {
        "description": f"Processed agricultural data for TabPFN - {split_name} set.",
        "split_type": split_name,
        "max_len": args.max_len,
        "temporal_cols_used": temporal_cols, 
        "static_cols_used": static_cols,   
        "target_col": target_col,
        "group_cols": group_cols,
        "time_col": args.time_col,
        "num_temporal_features": len(temporal_cols),
        "num_static_features": len(static_cols),
        "static_feature_row_index": static_feature_row_index,
        "total_feature_rows_per_sample": all_features_np.shape[1] if all_features_np.ndim == 3 else -1, # Check ndim
        "n_samples_in_split": all_features_np.shape[0]
    }
    logger.info(f"Global metadata for {split_name} data (info_dict): {info_dict}")

    np.savez_compressed(
        output_filepath,
        features=all_features_np,
        targets=all_targets_np,
        info=info_dict
    )
    logger.info(f"{split_name.capitalize()} data saved to {output_filepath}. Shapes: features={all_features_np.shape}, targets={all_targets_np.shape}")


def main(args: argparse.Namespace):
    logger.info(f"Starting data preprocessing from base directory: {args.base_data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max temporal length (max_len): {args.max_len}")

    os.makedirs(args.output_dir, exist_ok=True)

    temporal_cols = args.temporal_cols if args.temporal_cols else DEFAULT_TEMPORAL_COLS
    static_cols = args.static_cols if args.static_cols else DEFAULT_STATIC_COLS
    target_col = args.target_col if args.target_col else DEFAULT_TARGET_COL
    group_cols = args.group_cols if args.group_cols else DEFAULT_GROUP_COLS
    
    logger.info(f"Using {len(temporal_cols)} temporal columns: {temporal_cols}")
    logger.info(f"Using {len(static_cols)} static columns: {static_cols}")


    # --- Process Training Data ---
    train_dir_path = os.path.join(args.base_data_dir, "train")
    logger.info(f"--- Processing Training Data from {train_dir_path} ---")
    train_df_combined = load_and_combine_csvs_from_dir(train_dir_path)
    
    if train_df_combined is not None and not train_df_combined.empty:
        all_needed_cols = list(set(temporal_cols + static_cols + [target_col] + group_cols + ([args.time_col] if args.time_col and args.time_col in train_df_combined.columns else [])))
        missing_cols_in_df = [col for col in all_needed_cols if col not in train_df_combined.columns]
        if missing_cols_in_df:
            logger.error(f"Combined training DataFrame is missing required columns: {missing_cols_in_df}. "
                         "Please check CSV files or your --temporal_cols/--static_cols arguments. "
                         "Ensure column names in the script's defaults or CLI args match your CSV headers exactly.")
        else:
            train_output_filepath = os.path.join(args.output_dir, args.train_output_filename)
            process_dataframe_and_save_npz(
                train_df_combined, train_output_filepath, args,
                temporal_cols, static_cols, target_col, group_cols, "train"
            )
            # Attempt to free memory from the large combined DataFrame
            del train_df_combined
            gc.collect()
            logger.info("Processed and freed combined training DataFrame.")
    else:
        logger.warning("No training data loaded or combined. Skipping processing for training set.")

    # --- Process Testing Data ---
    test_dir_path = os.path.join(args.base_data_dir, "test")
    logger.info(f"--- Processing Testing Data from {test_dir_path} ---")
    test_df_combined = load_and_combine_csvs_from_dir(test_dir_path)

    if test_df_combined is not None and not test_df_combined.empty:
        all_needed_cols = list(set(temporal_cols + static_cols + [target_col] + group_cols + ([args.time_col] if args.time_col and args.time_col in test_df_combined.columns else [])))
        missing_cols_in_df = [col for col in all_needed_cols if col not in test_df_combined.columns]
        if missing_cols_in_df:
            logger.error(f"Combined testing DataFrame is missing required columns: {missing_cols_in_df}. "
                         "Please check CSV files or your --temporal_cols/--static_cols arguments. "
                         "Ensure column names in the script's defaults or CLI args match your CSV headers exactly.")
        else:
            test_output_filepath = os.path.join(args.output_dir, args.test_output_filename)
            process_dataframe_and_save_npz(
                test_df_combined, test_output_filepath, args,
                temporal_cols, static_cols, target_col, group_cols, "test"
            )
            # Attempt to free memory from the large combined DataFrame
            del test_df_combined
            gc.collect()
            logger.info("Processed and freed combined testing DataFrame.")
    else:
        logger.warning("No testing data loaded or combined. Skipping processing for testing set.")
        
    logger.info("Preprocessing finished for all specified data splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess agricultural data from structured directories for TabPFN.")
    parser.add_argument(
        "--base_data_dir", type=str, required=True,
        help="Path to the base directory containing 'train' and 'test' subdirectories with CSV files."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the processed .npz files."
    )
    parser.add_argument(
        "--max_len", type=int, default=62, 
        help="Maximum length (number of columns/timesteps) for temporal sequences."
    )
    parser.add_argument(
        "--temporal_cols", nargs='+', default=None,
        help="List of temporal feature column names. Overrides defaults if provided. Ensure these match CSV headers."
    )
    parser.add_argument(
        "--static_cols", nargs='+', default=None,
        help="List of static feature column names. Overrides defaults if provided. Ensure these match CSV headers."
    )
    parser.add_argument(
        "--target_col", type=str, default=None,
        help="Name of the target variable column. Uses default if provided."
    )
    parser.add_argument(
        "--group_cols", nargs='+', default=None,
        help="List of columns to group by to identify unique samples. Uses defaults if provided."
    )
    parser.add_argument(
        "--time_col", type=str, default=DEFAULT_TIME_COL,
        help="Name of the time/date column for sorting within groups (optional)."
    )
    parser.add_argument(
        "--train_output_filename", type=str, default="train_processed.npz",
        help="Filename for the training data output."
    )
    parser.add_argument(
        "--test_output_filename", type=str, default="test_processed.npz",
        help="Filename for the test data output."
    )

    args = parser.parse_args()
    main(args)
