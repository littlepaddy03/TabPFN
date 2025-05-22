#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC (adjust if necessary, or remove if not applicable to user's version)
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
Script to preprocess agricultural datasets for TabPFN adaptation.

This script reads CSV data, processes samples, pads/truncates sequences,
combines with static features (now including numerically encoded crop type),
and saves processed data as .npz files. Test data is also saved per crop.
"""
import argparse
import os
import logging
from typing import List, Dict, Tuple, Any, Optional
import gc 

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder # 导入 LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define default column names
DEFAULT_TEMPORAL_COLS = [
    'NDVI', 'Wind_Speed_10m_Mean', 'Temperature_Air_2m_Min_24h', 
    'Temperature_Air_2m_Max_24h', 'Temperature_Air_2m_Mean_24h',
    'Temperature_Air_2m_Max_Day_Time', 'Temperature_Air_2m_Mean_Day_Time',
    'Temperature_Air_2m_Min_Night_Time', 'Temperature_Air_2m_Mean_Night_Time',
    'Dew_Point_Temperature_2m_Mean', 'Precipitation_Flux',
    'Precipitation_Rain_Duration_Fraction', 'Precipitation_Solid_Duration_Fraction',
    'Snow_Thickness_Mean', 'Snow_Thickness_LWE_Mean', 'Vapour_Pressure_Mean',
    'Solar_Radiation_Flux', 'Cloud_Cover_Mean', 'Relative_Humidity_2m_06h',
    'Relative_Humidity_2m_15h'
]

DEFAULT_STATIC_COLS_ORIGINAL = [ # 原有的静态特征列
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
ENCODED_CROP_LABEL_COL = 'crop_label_encoded' # 新的编码作物特征列名

DEFAULT_TARGET_COL = 'yield'
DEFAULT_CROP_COL = 'crop' 
DEFAULT_GROUP_COLS = ['year', DEFAULT_CROP_COL, 'longitude', 'latitude']
DEFAULT_TIME_COL = 'date'


def load_and_combine_csvs_from_dir(directory_path: str) -> Optional[pd.DataFrame]:
    all_dfs: List[pd.DataFrame] = []
    logger.info(f"Scanning for CSV files in directory: {directory_path}")
    if not os.path.isdir(directory_path):
        logger.error(f"Provided path is not a directory: {directory_path}")
        return None
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                logger.debug(f"Reading CSV file: {file_path}") # Changed to debug for less verbosity
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
    static_cols: List[str], # 现在会包含 ENCODED_CROP_LABEL_COL
    target_col: str,
    max_len: int,
    group_cols_for_meta: List[str]
) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
    if group_df.empty: return None
    temp_df_copy = group_df.copy()

    # 处理时序特征
    for col in temporal_cols:
        if col not in temp_df_copy.columns: temp_df_copy.loc[:, col] = np.nan
    temporal_data_df = temp_df_copy[temporal_cols]
    for col in temporal_data_df.columns: # 确保空列是 float32
        if temporal_data_df[col].isnull().all():
            temporal_data_df[col] = temporal_data_df[col].astype(np.float32)
    temporal_data_np = temporal_data_df.values.astype(np.float32).T
    n_timesteps_in_group = temporal_data_np.shape[1]
    if n_timesteps_in_group > max_len:
        temporal_data_np = temporal_data_np[:, :max_len]
    elif n_timesteps_in_group < max_len:
        padding = np.full((temporal_data_np.shape[0], max_len - n_timesteps_in_group), np.nan, dtype=np.float32)
        temporal_data_np = np.concatenate([temporal_data_np, padding], axis=1)

    # 处理静态特征 (现在包括编码后的作物标签)
    for col in static_cols: # static_cols 现在包含了 ENCODED_CROP_LABEL_COL
        if col not in temp_df_copy.columns: temp_df_copy.loc[:, col] = np.nan
    
    static_data_series = temp_df_copy[static_cols].iloc[0] # 取第一行作为静态值
    static_data_np = static_data_series.astype(np.float32).values
    static_data_expanded = np.repeat(static_data_np[:, np.newaxis], max_len, axis=1)

    feature_array = np.vstack([temporal_data_np, static_data_expanded])
    target_value = temp_df_copy[target_col].iloc[0]
    if pd.isna(target_value): return None

    sample_meta_info = {key: temp_df_copy[key].iloc[0] for key in group_cols_for_meta if key in temp_df_copy}
    sample_meta_info['original_timesteps'] = n_timesteps_in_group
    return feature_array, float(target_value), sample_meta_info


def process_dataframe_and_save_npz(
    df: pd.DataFrame,
    output_filepath: str,
    args: argparse.Namespace,
    temporal_cols: List[str],
    static_cols: List[str], # 这个列表现在会包含编码后的作物列名
    target_col: str,
    group_cols: List[str],
    crop_identity_col: str, # 原始的作物文本列名 (e.g., 'crop')
    split_name: str,
    label_encoder: Optional[LabelEncoder] = None, # 传入拟合好的 LabelEncoder
    crop_label_mapping: Optional[Dict[str, int]] = None # 传入作物名称到整数的映射
):
    if df is None or df.empty:
        logger.warning(f"DataFrame for {split_name} is empty. Skipping NPZ for {output_filepath}")
        return

    logger.info(f"Processing DataFrame for {split_name}. Shape: {df.shape}")

    # 应用 Label Encoding 来创建新的数值作物特征列
    if label_encoder and crop_identity_col in df.columns:
        logger.info(f"Applying LabelEncoder to '{crop_identity_col}' column for {split_name} data.")
        try:
            # 对于测试集中可能出现的新作物（未在训练集中fit过的），LabelEncoder会报错
            # 因此，确保LabelEncoder在所有唯一作物上拟合，或者处理未知标签
            # 当前实现是在main函数中用train+test的所有作物拟合label_encoder
            df[ENCODED_CROP_LABEL_COL] = label_encoder.transform(df[crop_identity_col])
            df[ENCODED_CROP_LABEL_COL] = df[ENCODED_CROP_LABEL_COL].astype(np.float32) # 确保是数值类型
        except ValueError as e:
            logger.error(f"Error transforming '{crop_identity_col}' using LabelEncoder for {split_name}: {e}")
            logger.error(f"This might happen if test set contains crop labels not seen during LabelEncoder fitting.")
            logger.error(f"Unique crops in current df: {df[crop_identity_col].unique()}")
            logger.error(f"LabelEncoder was fit on: {label_encoder.classes_}")
            # 可以选择填充一个特殊值或跳过这些样本
            # For now, let it raise or handle as NaN if it becomes one.
            # A robust solution would be to map unknown to a specific category or use a try-except per row.
            # For simplicity, we assume label_encoder is fit on all possible crops.
            df[ENCODED_CROP_LABEL_COL] = -1 #  标记为未知/错误
            df[ENCODED_CROP_LABEL_COL] = df[ENCODED_CROP_LABEL_COL].astype(np.float32)


    if args.time_col and args.time_col in df.columns:
        try:
            df[args.time_col] = pd.to_datetime(df[args.time_col])
            df = df.sort_values(by=group_cols + [args.time_col])
        except Exception as e:
            logger.warning(f"Could not parse/sort time column '{args.time_col}' for {split_name}: {e}.")

    grouped = df.groupby(group_cols, sort=False)
    logger.info(f"Number of unique sample groups in {split_name}: {len(grouped)}")

    processed_features_list: List[np.ndarray] = []
    processed_targets_list: List[float] = []
    processed_crop_names_list: List[str] = []

    # static_cols 已经包含了 ENCODED_CROP_LABEL_COL (由main函数传入)
    expected_num_feature_rows = len(temporal_cols) + len(static_cols)
    expected_feature_shape = (expected_num_feature_rows, args.max_len)

    processed_count = 0; skipped_count = 0
    for i, (_group_name, group_df) in enumerate(grouped):
        if i % 20000 == 0 and i > 0: logger.info(f"Processing group {i}/{len(grouped)} for {split_name}...")
        processed_sample = process_sample_group(
            group_df, temporal_cols, static_cols, target_col, args.max_len, group_cols
        )
        if processed_sample:
            feature_array, target_value, sample_meta_info = processed_sample
            if feature_array.shape != expected_feature_shape:
                skipped_count += 1; continue
            processed_features_list.append(feature_array)
            processed_targets_list.append(target_value)
            try:
                crop_name = str(sample_meta_info[crop_identity_col])
                processed_crop_names_list.append(crop_name)
            except KeyError:
                processed_crop_names_list.append("unknown_crop")
            processed_count +=1
        else: skipped_count += 1
    logger.info(f"For {split_name}: Processed {processed_count}, skipped {skipped_count} samples.")

    if not processed_features_list:
        logger.error(f"No samples successfully processed for {split_name}. NPZ file {output_filepath} not generated."); return
    try:
        all_features_np = np.stack(processed_features_list, axis=0).astype(np.float32)
    except ValueError as e:
        logger.error(f"Failed to stack feature arrays for {split_name}: {e}"); return
    del processed_features_list; gc.collect()

    all_targets_np = np.array(processed_targets_list, dtype=np.float32)
    all_crop_names_np = np.array(processed_crop_names_list)
    if all_features_np.shape[0] != all_crop_names_np.shape[0] and all_crop_names_np.size > 0:
        logger.error(f"Mismatch: features ({all_features_np.shape[0]}) vs crop names ({all_crop_names_np.shape[0]}) for {split_name}. Aborting."); return

    logger.info(f"Final processed features shape for {split_name}: {all_features_np.shape}")
    logger.info(f"Final processed targets shape for {split_name}: {all_targets_np.shape}")

    static_feature_row_index = len(temporal_cols)
    info_dict = {
        "description": f"Processed agricultural data for TabPFN - {split_name} set.",
        "split_type": split_name, "max_len": args.max_len,
        "temporal_cols_used": temporal_cols, "static_cols_used": static_cols, # static_cols now includes encoded crop
        "target_col": target_col, "group_cols": group_cols, "time_col": args.time_col,
        "num_temporal_features": len(temporal_cols),
        "num_static_features": len(static_cols), # This count is now correct
        "static_feature_start_row_index_in_sample_matrix": static_feature_row_index,
        "total_feature_rows_per_sample": all_features_np.shape[1] if all_features_np.ndim == 3 else -1,
        "n_samples_in_split": all_features_np.shape[0],
        "crop_identity_col_used": crop_identity_col, # Original crop column name
        "encoded_crop_label_col_name": ENCODED_CROP_LABEL_COL, # Name of the new encoded column
        "crop_label_mapping": crop_label_mapping # Save the mapping
    }
    logger.info(f"Global metadata for {split_name} (info_dict, excerpt): "
                f"num_static_features={info_dict['num_static_features']}, "
                f"static_cols_used (last 5): {info_dict['static_cols_used'][-5:]}")

    np.savez_compressed(output_filepath, features=all_features_np, targets=all_targets_np, info=info_dict)
    logger.info(f"{split_name.capitalize()} data saved to {output_filepath}.")

    if split_name == 'test':
        if not all_crop_names_np.size: logger.info("No crop names for test split, cannot save per-crop files."); return
        unique_crops_in_test = np.unique(all_crop_names_np)
        base_output_dir = os.path.dirname(output_filepath)
        original_test_filename_stem = os.path.splitext(os.path.basename(output_filepath))[0]
        for crop_val in unique_crops_in_test:
            sanitized_crop_name = "".join(c if c.isalnum() else "_" for c in str(crop_val)) or "unknown_crop"
            mask = (all_crop_names_np == crop_val)
            features_crop_specific = all_features_np[mask]
            targets_crop_specific = all_targets_np[mask]
            if features_crop_specific.shape[0] == 0: continue
            info_crop_specific = info_dict.copy()
            info_crop_specific['description'] = f"Processed agricultural data for TabPFN - test set - CROP: {crop_val}."
            info_crop_specific['filtered_for_crop'] = crop_val
            info_crop_specific['n_samples_in_split'] = features_crop_specific.shape[0]
            output_filename_crop = f"{original_test_filename_stem}_{sanitized_crop_name}.npz"
            output_file_path_crop = os.path.join(base_output_dir, output_filename_crop)
            logger.info(f"Saving test data for crop '{crop_val}' to {output_file_path_crop}...")
            np.savez_compressed(output_file_path_crop, features=features_crop_specific,
                                targets=targets_crop_specific, info=info_crop_specific)
        logger.info(f"Finished saving per-crop test files to {base_output_dir}.")


def main(args: argparse.Namespace):
    logger.info(f"Starting data preprocessing. Base dir: {args.base_data_dir}, Output dir: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    temporal_cols = args.temporal_cols if args.temporal_cols else DEFAULT_TEMPORAL_COLS
    # static_cols WILL BE MODIFIED to include the encoded crop label
    base_static_cols = args.static_cols if args.static_cols else DEFAULT_STATIC_COLS_ORIGINAL.copy()
    target_col = args.target_col if args.target_col else DEFAULT_TARGET_COL
    group_cols = args.group_cols if args.group_cols else DEFAULT_GROUP_COLS.copy()
    crop_identity_col = DEFAULT_CROP_COL # This is the original text column, e.g., 'crop'
    
    # --- Step 1: Collect all unique crop names and fit LabelEncoder ---
    logger.info("Collecting all unique crop names from train and test data to fit LabelEncoder...")
    all_dfs_for_crop_scan: List[pd.DataFrame] = []
    train_dir_path_for_scan = os.path.join(args.base_data_dir, "train")
    test_dir_path_for_scan = os.path.join(args.base_data_dir, "test")

    df_train_scan = load_and_combine_csvs_from_dir(train_dir_path_for_scan)
    if df_train_scan is not None and crop_identity_col in df_train_scan.columns:
        all_dfs_for_crop_scan.append(df_train_scan[[crop_identity_col]])
    
    df_test_scan = load_and_combine_csvs_from_dir(test_dir_path_for_scan)
    if df_test_scan is not None and crop_identity_col in df_test_scan.columns:
        all_dfs_for_crop_scan.append(df_test_scan[[crop_identity_col]])

    if not all_dfs_for_crop_scan:
        logger.error(f"Could not load any data or find '{crop_identity_col}' column to determine crop types. Aborting.")
        return

    combined_crops_df = pd.concat(all_dfs_for_crop_scan, ignore_index=True)
    unique_crop_names = combined_crops_df[crop_identity_col].astype(str).unique() # Ensure string type for fitting
    
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_crop_names)
    logger.info(f"LabelEncoder fit on {len(label_encoder.classes_)} unique crop classes: {list(label_encoder.classes_)}")
    
    crop_label_mapping = {name: int(label) for name, label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    logger.info(f"Crop to Label Mapping: {crop_label_mapping}")

    # Add the new encoded crop label column to the list of static columns
    # This modified list will be used for processing both train and test data.
    final_static_cols = base_static_cols + [ENCODED_CROP_LABEL_COL]
    logger.info(f"Final static columns (including encoded crop): {final_static_cols}")
    logger.info(f"Total static features (including encoded crop): {len(final_static_cols)}")


    # --- Process Training Data ---
    train_dir_path = os.path.join(args.base_data_dir, "train")
    logger.info(f"--- Processing Training Data from {train_dir_path} ---")
    # train_df_combined was df_train_scan, can reuse or reload if memory is a concern
    train_df_combined = df_train_scan if df_train_scan is not None else load_and_combine_csvs_from_dir(train_dir_path)


    if train_df_combined is not None and not train_df_combined.empty:
        # Check for all necessary columns (including the original crop_identity_col for encoding)
        all_needed_cols_train = list(set(temporal_cols + base_static_cols + [target_col] + group_cols + 
                                         ([args.time_col] if args.time_col and args.time_col in train_df_combined.columns else []) +
                                         [crop_identity_col])) # Ensure original crop_identity_col is checked
        missing_cols_in_train_df = [col for col in all_needed_cols_train if col not in train_df_combined.columns]
        if missing_cols_in_train_df:
            logger.error(f"Combined training DataFrame is missing required columns: {missing_cols_in_train_df}.")
        else:
            train_output_filepath = os.path.join(args.output_dir, args.train_output_filename)
            process_dataframe_and_save_npz(
                train_df_combined, train_output_filepath, args,
                temporal_cols, final_static_cols, target_col, group_cols, # Use final_static_cols
                crop_identity_col, "train", label_encoder, crop_label_mapping
            )
    else: logger.warning("No training data loaded. Skipping processing for training set.")
    del train_df_combined, df_train_scan; gc.collect() # Free memory

    # --- Process Testing Data ---
    test_dir_path = os.path.join(args.base_data_dir, "test")
    logger.info(f"--- Processing Testing Data from {test_dir_path} ---")
    test_df_combined = df_test_scan if df_test_scan is not None else load_and_combine_csvs_from_dir(test_dir_path)

    if test_df_combined is not None and not test_df_combined.empty:
        all_needed_cols_test = list(set(temporal_cols + base_static_cols + [target_col] + group_cols + 
                                        ([args.time_col] if args.time_col and args.time_col in test_df_combined.columns else []) +
                                        [crop_identity_col]))
        missing_cols_in_test_df = [col for col in all_needed_cols_test if col not in test_df_combined.columns]
        if missing_cols_in_test_df:
            logger.error(f"Combined testing DataFrame is missing required columns: {missing_cols_in_test_df}.")
        else:
            test_output_filepath = os.path.join(args.output_dir, args.test_output_filename)
            process_dataframe_and_save_npz(
                test_df_combined, test_output_filepath, args,
                temporal_cols, final_static_cols, target_col, group_cols, # Use final_static_cols
                crop_identity_col, "test", label_encoder, crop_label_mapping
            )
    else: logger.warning("No testing data loaded. Skipping processing for testing set.")
    del test_df_combined, df_test_scan, combined_crops_df; gc.collect() # Free memory

    logger.info("Preprocessing finished for all specified data splits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess agricultural data for TabPFN.")
    parser.add_argument("--base_data_dir", type=str, required=True, help="Path to base dir with 'train'/'test' subdirs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed .npz files.")
    parser.add_argument("--max_len", type=int, default=128, help="Max length for temporal sequences.") # Adjusted default to match user's data
    parser.add_argument("--temporal_cols", nargs='+', default=None, help="Temporal feature column names.")
    parser.add_argument("--static_cols", nargs='+', default=None, help="Original static feature column names (excluding encoded crop).")
    parser.add_argument("--target_col", type=str, default=None, help="Target variable column name.")
    parser.add_argument("--group_cols", nargs='+', default=None, help="Columns to group by.")
    parser.add_argument("--time_col", type=str, default=DEFAULT_TIME_COL, help="Time/date column for sorting.")
    parser.add_argument("--train_output_filename", type=str, default="train_processed.npz")
    parser.add_argument("--test_output_filename", type=str, default="test_processed.npz")
    args = parser.parse_args()
    main(args)

