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

"""Main script for running baseline models for agricultural yield prediction.

Utilizes numerically encoded crop type as an input feature.
"""

import argparse
import numpy as np
import os
import traceback
from typing import Dict, Any, List, Tuple
import joblib
import pandas as pd
import glob
import re

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

import mlflow
import mlflow.sklearn

try:
  import load_data
  from src.tabpfn.agri_utils.baseline_featurizer import featurize_3d_to_2d
  import evaluate # 假设 evaluate.py 在同一目录或PYTHONPATH中
except ImportError as e:
  print(f"Error importing modules: {e}")
  print("Please ensure 'load_data.py', 'evaluate.py' are accessible and "
        "the 'src' directory is correctly structured.")
  print("Current working directory:", os.getcwd())
  exit(1)


# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_DATA_DIR = '/root/lanyun-tmp/datasets/us_processed_cp_v2/' # 假设新文件路径
DEFAULT_TEST_DATA_DIR = '/root/lanyun-tmp/datasets/us_processed_cp_v2/'   # 假设新文件路径
DEFAULT_MLFLOW_EXPERIMENT_NAME = "Baseline Models - Agri Yield - Single Crop Training"

VAL_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
AGGREGATION_FUNCTIONS = ['mean', 'std', 'median', 'min', 'max', 'sum', 'count_nan', 'count_valid']

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

CROPS_FOR_DETAILED_DEBUG = ["corn", "wheat_spring_excl_drum", "soybeans"]
ENCODED_CROP_FEATURE_COL_NAME = 'crop_label_encoded' # 与 preprocess_agri_datasets_cp_v2.py 中定义的一致


def adapt_info_keys(original_info: Dict, context: str = "Unknown") -> Dict:
  """Adapts keys from load_data's info dict to what featurizer expects."""
  adapted = {}
  print(f"    Adapting info keys for {context}...")
  try:
    required_source_keys = [
        'static_feature_start_row_index_in_sample_matrix',
        'num_temporal_features', 'num_static_features',
        'temporal_cols_used', 'static_cols_used', # 这些现在应该包含编码后的作物列
        'encoded_crop_label_col_name', # 确认新键存在
        'crop_label_mapping'           # 确认新键存在
    ]
    for key in required_source_keys:
        if key not in original_info:
            # 对于 crop_label_mapping，如果它不存在于旧的 info 中（例如处理旧数据），可以给一个默认值或警告
            if key == 'crop_label_mapping' and 'encoded_crop_label_col_name' not in original_info:
                 print(f"    警告 ({context}): '{key}' not found, and encoded crop label also not found. Proceeding without it.")
                 adapted[key] = {} # 或者 None
                 continue
            elif key == 'encoded_crop_label_col_name' and 'crop_label_mapping' not in original_info:
                 print(f"    警告 ({context}): '{key}' not found, and crop_label_mapping also not found. Proceeding without it.")
                 adapted[key] = None # 或者 ""
                 continue

            raise KeyError(f"Source key '{key}' missing in original_info for {context}. Keys: {list(original_info.keys())}")

    adapted['static_feature_row_index'] = original_info[
        'static_feature_start_row_index_in_sample_matrix']
    
    num_temporal = original_info['num_temporal_features']
    num_static = original_info['num_static_features'] # 这个值现在应该是 61 (原始) + 1 (编码作物) = 62
    
    print(f"      (adapt_info_keys for {context}) num_temporal: {num_temporal}, num_static: {num_static}")

    adapted['temporal_cols_idx'] = list(range(num_temporal))
    # 静态特征索引现在从 num_temporal 开始，直到 num_temporal + num_static -1
    adapted['static_cols_idx'] = list(range(num_temporal, num_temporal + num_static))
    
    if 'max_len' in original_info: adapted['max_len'] = original_info['max_len']
    
    # 这些原始列名列表用于特征重要性分析，确保它们与 num_features 计数匹配
    adapted['original_temporal_cols_used'] = original_info['temporal_cols_used']
    adapted['original_static_cols_used'] = original_info['static_cols_used'] # 这个列表现在应该包含 ENCODED_CROP_FEATURE_COL_NAME

    # 健全性检查
    if len(adapted['temporal_cols_idx']) != len(original_info['temporal_cols_used']):
        print(f"    警告 ({context}): 生成的 temporal_cols_idx 长度 ({len(adapted['temporal_cols_idx'])}) "
              f"与 temporal_cols_used 长度 ({len(original_info['temporal_cols_used'])}) 不匹配。")
    if len(adapted['static_cols_idx']) != len(original_info['static_cols_used']):
        print(f"    警告 ({context}): 生成的 static_cols_idx 长度 ({len(adapted['static_cols_idx'])}) "
              f"与 static_cols_used 长度 ({len(original_info['static_cols_used'])}) 不匹配。")
    
    adapted['encoded_crop_label_col_name'] = original_info['encoded_crop_label_col_name']
    adapted['crop_label_mapping'] = original_info['crop_label_mapping']

    print(f"    Adaptation successful for {context}. Num static features in adapted info: {len(adapted['static_cols_idx'])}")
    if ENCODED_CROP_FEATURE_COL_NAME not in adapted['original_static_cols_used']:
        print(f"    严重警告 ({context}): '{ENCODED_CROP_FEATURE_COL_NAME}' 未在 adapted['original_static_cols_used'] 中找到! "
              f"列表内容: {adapted['original_static_cols_used']}")


  except KeyError as e:
    print(f"    错误 ({context}): Error adapting info keys: {e}. Original keys: {list(original_info.keys())}")
    raise
  return adapted

def discover_crop_train_files(train_data_dir: str, train_prefix: str) -> List[Dict[str, str]]:
    """Discovers per-crop training NPZ files in a directory.

    Args:
        train_data_dir: Directory to scan for training files.
        train_prefix: Prefix of the training files (e.g., "train_processed_").

    Returns:
        A list of dictionaries, where each dictionary contains 'name' (crop name)
        and 'path' (full file path).
    """
    crop_files = []
    if not os.path.isdir(train_data_dir):
        print(f"警告: 训练数据目录 {train_data_dir} 不是一个有效目录。")
        return crop_files

    pattern = os.path.join(train_data_dir, f"{train_prefix}*.npz")
    discovered_files = glob.glob(pattern)

    if not discovered_files:
        print(f"警告: 在目录 {train_data_dir} 中未找到匹配前缀 '{train_prefix}' 的训练文件。")
        return crop_files

    for file_path in discovered_files:
        filename = os.path.basename(file_path)
        # Escape the prefix in case it contains special regex characters
        escaped_prefix = re.escape(train_prefix)
        match = re.match(rf"{escaped_prefix}(.+)\.npz", filename)
        if match:
            crop_name = match.group(1)
            crop_files.append({"name": crop_name, "path": file_path})
        else:
            print(f"警告: 无法从训练文件名 {filename} 中使用前缀 '{train_prefix}' 提取作物名称。")
    
    print(f"发现的训练作物文件: {[cf['name'] for cf in crop_files]}")
    return crop_files

def discover_test_sets(test_data_dir: str, test_prefix: str) -> List[Dict[str, str]]:
    """Discovers per-crop test NPZ files in a directory.

    Args:
        test_data_dir: Directory to scan for test files.
        test_prefix: Prefix of the test files (e.g., "test_processed_").

    Returns:
        A list of dictionaries, where each dictionary contains 'name' (crop name)
        and 'path' (full file path).
    """
    test_sets = []
    if not os.path.isdir(test_data_dir):
        print(f"警告: 测试数据目录 {test_data_dir} 不是一个有效目录。")
        return test_sets

    pattern = os.path.join(test_data_dir, f"{test_prefix}*.npz")
    per_crop_files = glob.glob(pattern)

    if not per_crop_files:
        print(f"警告: 在目录 {test_data_dir} 中未找到匹配前缀 '{test_prefix}' 的测试文件。")
        return test_sets

    for crop_file_path in per_crop_files:
        filename = os.path.basename(crop_file_path)
        # Escape the prefix in case it contains special regex characters
        escaped_prefix = re.escape(test_prefix)
        match = re.match(rf"{escaped_prefix}(.+)\.npz", filename)
        if match:
            crop_name = match.group(1)
            test_sets.append({"name": crop_name, "path": crop_file_path})
        else:
            print(f"警告: 无法从测试文件名 {filename} 中使用前缀 '{test_prefix}' 提取作物名称。")
            
    print(f"发现的测试集作物文件: {[ts['name'] for ts in test_sets]}")
    return test_sets

def print_prediction_diagnostics(
    y_original: np.ndarray, y_pred_scaled: np.ndarray, y_pred_original: np.ndarray,
    y_scaler: StandardScaler, test_set_name: str
):
    # ... (与之前版本相同) ...
    print(f"\n--- '{test_set_name}' 预测值诊断 ---")
    print(f"  y_original 形状: {y_original.shape}, 前5个: {y_original[:5]}")
    print(f"  y_pred_scaled (模型直接输出) 形状: {y_pred_scaled.shape}, 前5个: {y_pred_scaled[:5]}")
    print(f"  y_pred_original (逆标准化后) 形状: {y_pred_original.shape}, 前5个: {y_pred_original[:5]}")
    global_y_mean_original = y_scaler.mean_[0]
    global_y_std_original = np.sqrt(y_scaler.var_[0]) if y_scaler.var_[0] > 1e-9 else 1.0
    print(f"  全局训练集 y 均值 (原始尺度): {global_y_mean_original:.4f}")
    if y_original.size > 0:
        crop_y_mean_original = np.mean(y_original)
        print(f"  '{test_set_name}' y 均值 (原始尺度): {crop_y_mean_original:.4f}")
        crop_y_mean_scaled = (crop_y_mean_original - global_y_mean_original) / global_y_std_original if global_y_std_original > 1e-6 else float('nan')
        print(f"  '{test_set_name}' y 均值 (转换到全局标准化尺度后): {crop_y_mean_scaled:.4f}")
        if y_pred_scaled.size > 0:
            mean_pred_scaled = np.mean(y_pred_scaled)
            print(f"  '{test_set_name}' y_pred_scaled 均值: {mean_pred_scaled:.4f}")
    print("--- 预测值诊断结束 ---")


def train_model(
    model_name: str, model_instance: Any, x_train_scaled: np.ndarray, y_train_scaled: np.ndarray
) -> Any:
    # ... (与之前版本相同) ...
    print(f"\n--- Training Model: {model_name} ---")
    try:
        print(f"   Fitting {model_name}...")
        model_instance.fit(x_train_scaled, y_train_scaled.ravel())
        print(f"   {model_name} trained successfully.")
        return model_instance
    except Exception as e:
        print(f"Error during training of {model_name}: {e}"); traceback.print_exc(); return None

def evaluate_on_test_set( 
    fitted_model: Any, test_set_name: str, x_test_scaled: np.ndarray,
    y_test_original: np.ndarray, y_scaler: StandardScaler
) -> Dict:
    # ... (与之前版本相同) ...
    print(f"   Evaluating on test set: {test_set_name}...")
    results = {"metrics": None, "y_pred_original": None, "y_pred_scaled": None, "error": None}
    try:
        y_pred_scaled_raw = fitted_model.predict(x_test_scaled) 
        results["y_pred_scaled"] = y_pred_scaled_raw 
        y_pred_original = y_scaler.inverse_transform(y_pred_scaled_raw.reshape(-1, 1)).ravel()
        results["y_pred_original"] = y_pred_original
        # print(f"    evaluate_on_test_set ({test_set_name}):") # 可减少打印
        # print(f"      y_test_original shape: {y_test_original.shape}, first 5: {y_test_original[:5]}")
        # print(f"      y_pred_original shape: {y_pred_original.shape}, first 5: {y_pred_original[:5]}")
        metrics = evaluate.calculate_regression_metrics(y_test_original, y_pred_original)
        results["metrics"] = metrics
        print(f"   Metrics for {test_set_name}: {metrics}")
    except Exception as e:
        print(f"Error during evaluation on {test_set_name} for model {getattr(fitted_model, '__class__', type(fitted_model).__name__)}: {e}")
        traceback.print_exc(); results["error"] = str(e)
    return results


def main(args):
  print(f"--- 使用编码作物特征运行基线模型 ---")
  mlflow.set_experiment(args.mlflow_experiment_name)

  # print(f"\n1. 从 '{args.train_data_dir}' 加载并准备训练/验证数据...") # MODIFIED
  # try:
  #   # TODO: This part needs to be updated to handle per-crop loading based on new args.
  #   # For now, it will likely fail if the path is a directory.
  #   # This will be addressed in a subsequent subtask.
  #   x_full_train_3d, y_full_train_original, original_train_info = load_data.load_npz_data(
  #       args.train_data_dir) # MODIFIED
    
  #   print(f"  原始训练数据 info['num_static_features']: {original_train_info.get('num_static_features')}")
  #   print(f"  原始训练数据 info['static_cols_used'] (最后5个): {original_train_info.get('static_cols_used', [])[-5:]}")
  #   print(f"  原始训练数据 info['encoded_crop_label_col_name']: {original_train_info.get('encoded_crop_label_col_name')}")
  #   # print(f"  原始训练数据 info['crop_label_mapping']: {original_train_info.get('crop_label_mapping')}")

  #   train_info_for_featurizer = adapt_info_keys(original_train_info, "训练集")
    
  #   y_train_original_stats = {"mean": np.mean(y_full_train_original), "std": np.std(y_full_train_original),
  #                             "min": np.min(y_full_train_original), "max": np.max(y_full_train_original)}
  #   x_train_3d, x_val_3d, y_train_original, y_val_original = load_data.split_data(
  #       x_full_train_3d, y_full_train_original,
  #       test_size=args.val_split_size, random_state=args.random_seed)
  # except Exception as e: print(f"加载或拆分训练数据时出错: {e}"); traceback.print_exc(); return

  # print(f"\n2. 将训练集和验证集的3D特征转换为2D...")
  # try:
  #   x_train_2d = featurize_3d_to_2d(x_train_3d, train_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
  #   x_val_2d = featurize_3d_to_2d(x_val_3d, train_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
  #   print(f"   x_train_2d 形状: {x_train_2d.shape}") # 应该有 20*2 + 62 = 102 列
  # except Exception as e: print(f"转换3D特征到2D时出错: {e}"); traceback.print_exc(); return

  # print("\n3. 标准化特征和目标变量 (使用训练数据拟合)...")
  # x_scaler = StandardScaler(); y_scaler = StandardScaler()
  # try:
  #   y_train_original_reshaped = y_train_original.reshape(-1, 1)
  #   x_train_scaled = x_scaler.fit_transform(x_train_2d)
  #   y_train_scaled = y_scaler.fit_transform(y_train_original_reshaped)
  #   x_val_scaled = x_scaler.transform(x_val_2d)
  #   print("   特征和目标标准化器已在训练数据上拟合。")
  # except Exception as e: print(f"标准化数据时出错: {e}"); traceback.print_exc(); return

  print("\n--- Initializing Data Discovery ---")
  # Discover training files for each crop
  # This script will now iterate over crops for training, so we need to find them first.
  # The actual iteration logic will be in a subsequent subtask.
  # For now, we just call the discovery function.
  discovered_train_crop_files = discover_crop_train_files(args.train_data_dir, args.train_file_prefix)
  if not discovered_train_crop_files:
      print(f"错误: 在 {args.train_data_dir} 使用前缀 '{args.train_file_prefix}' 未找到任何训练作物文件。脚本将退出。")
      return

  # Discover test sets (per-crop)
  # The per_crop_test_dir argument is currently unused here but might be used if
  # test files are in a different structure than train files.
  # For now, assuming test_data_dir and test_file_prefix are primary.
  all_test_sets = discover_test_sets(args.test_data_dir, args.test_file_prefix)
  if not all_test_sets:
      print(f"警告: 在 {args.test_data_dir} 使用前缀 '{args.test_file_prefix}' 未找到任何测试作物文件。评估将受限。")
      # Depending on requirements, might want to return if no test sets are found.
  
  test_set_map = {ts['name']: ts['path'] for ts in all_test_sets}
  
  baseline_models_config = [
      {"name": "MeanPredictor", "instance": DummyRegressor(strategy="mean"), "params": {"strategy": "mean"}},
      {"name": "LinearRegression", "instance": LinearRegression(), "params": {}},
      {"name": "RandomForestRegressor_WithCropFeature", # 更新模型名以反映变化
       "instance": RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                       random_state=args.random_seed, n_jobs=-1),
       "params": {"n_estimators": RF_N_ESTIMATORS, "max_depth": RF_MAX_DEPTH, "random_state": args.random_seed}}
  ]
  
  overall_summary_results_all_crops = [] # To store summaries from all crops

  print("\n--- Starting Per-Crop Training and Evaluation ---")
  for crop_info in discovered_train_crop_files:
      crop_name = crop_info['name']
      crop_train_path = crop_info['path']
      print(f"\n===== PROCESSING CROP: {crop_name} =====")
      print(f"  Training data file: {crop_train_path}")

      # 1. Load crop-specific training data
      try:
          x_full_train_3d_crop, y_full_train_original_crop, original_train_info_crop = \
              load_data.load_npz_data(crop_train_path)
          print(f"  Loaded training data for {crop_name}:")
          print(f"    x_full_train_3d_crop shape: {x_full_train_3d_crop.shape}")
          print(f"    y_full_train_original_crop shape: {y_full_train_original_crop.shape}")
      except Exception as e:
          print(f"  错误: 加载作物 '{crop_name}' 的训练数据 ({crop_train_path}) 时失败: {e}")
          traceback.print_exc()
          continue # Skip to the next crop

      # 2. Match and load corresponding test data
      crop_test_path = test_set_map.get(crop_name)
      if not crop_test_path or not os.path.exists(crop_test_path):
          print(f"  警告: 未找到作物 '{crop_name}' 的测试集文件，或路径 '{crop_test_path}' 不存在。跳过此作物。")
          continue
      
      try:
          x_test_3d_crop, y_test_original_crop, original_test_info_crop = \
              load_data.load_npz_data(crop_test_path)
          print(f"  Loaded test data for {crop_name}:")
          print(f"    x_test_3d_crop shape: {x_test_3d_crop.shape}")
          print(f"    y_test_original_crop shape: {y_test_original_crop.shape}")
      except Exception as e:
          print(f"  错误: 加载作物 '{crop_name}' 的测试数据 ({crop_test_path}) 时失败: {e}")
          traceback.print_exc()
          continue # Skip to the next crop

      # 3. Adapt variable names (no validation split for now)
      x_train_3d_crop = x_full_train_3d_crop
      y_train_original_crop = y_full_train_original_crop
      # original_train_info_crop and original_test_info_crop are loaded per crop.

      try:
          print(f"  Adapting info keys for {crop_name} train and test data...")
          train_info_for_featurizer_crop = adapt_info_keys(original_train_info_crop, f"{crop_name} Train")
          test_info_for_featurizer_crop = adapt_info_keys(original_test_info_crop, f"{crop_name} Test")

          print(f"  Featurizing 3D data to 2D for {crop_name}...")
          x_train_2d_crop = featurize_3d_to_2d(x_train_3d_crop, train_info_for_featurizer_crop, agg_funcs=AGGREGATION_FUNCTIONS)
          x_test_2d_crop = featurize_3d_to_2d(x_test_3d_crop, test_info_for_featurizer_crop, agg_funcs=AGGREGATION_FUNCTIONS)
          print(f"    x_train_2d_crop shape for {crop_name}: {x_train_2d_crop.shape}")
          print(f"    x_test_2d_crop shape for {crop_name}: {x_test_2d_crop.shape}")

          print(f"  Scaling features and target for {crop_name}...")
          x_scaler_crop = StandardScaler()
          y_scaler_crop = StandardScaler()

          y_train_original_reshaped_crop = y_train_original_crop.reshape(-1, 1)
          x_train_scaled_crop = x_scaler_crop.fit_transform(x_train_2d_crop)
          y_train_scaled_crop = y_scaler_crop.fit_transform(y_train_original_reshaped_crop)
          
          x_test_scaled_crop = x_scaler_crop.transform(x_test_2d_crop)
          print(f"    Per-crop scaling completed for {crop_name}.")

      except Exception as e:
          print(f"  错误: 在处理作物 '{crop_name}' 的特征工程或缩放时失败: {e}")
          traceback.print_exc()
          continue # Skip to the next crop

      # TODO (in subsequent steps):
      # - Per-crop MLflow logging
      # - Aggregate results for this crop and append to overall_summary_results_all_crops
      
      for model_config in baseline_models_config:
          model_name = model_config["name"]
          model_instance_template = model_config["instance"] # This is a new instance for each crop and model
          model_params_to_log = model_config["params"]

          print(f"\n  --- Processing Model: {model_name} for Crop: {crop_name} ---")
          
          fitted_model_crop = train_model(
              model_name=f"{model_name}_{crop_name}", # Model name includes crop
              model_instance=model_instance_template, 
              x_train_scaled=x_train_scaled_crop, 
              y_train_scaled=y_train_scaled_crop
          )

          current_model_summary = {
              "crop_name": crop_name,
              "model_name": model_name,
              "metrics": None, # Default if training fails or evaluation error
              "error": None
          }

          if not fitted_model_crop:
              print(f"  错误: 模型 '{model_name}' 在作物 '{crop_name}' 上训练失败。")
              current_model_summary["error"] = "Training failed"
              overall_summary_results_all_crops.append(current_model_summary)
              continue # To the next model in baseline_models_config

          # Feature Importance (for RandomForest)
          if "RandomForest" in model_name and hasattr(fitted_model_crop, 'feature_importances_'):
              print(f"\n  --- {model_name}_{crop_name} 特征重要性分析 ---")
              importances = fitted_model_crop.feature_importances_
              
              feature_names_2d = []
              # Use train_info_for_featurizer_crop for feature names
              temp_cols_orig = train_info_for_featurizer_crop.get('original_temporal_cols_used', [])
              for temp_col_name in temp_cols_orig:
                  for agg_func_name in AGGREGATION_FUNCTIONS: # AGGREGATION_FUNCTIONS is global
                      feature_names_2d.append(f"{temp_col_name}_{agg_func_name}")
              
              stat_cols_orig = train_info_for_featurizer_crop.get('original_static_cols_used', []) 
              feature_names_2d.extend(stat_cols_orig)

              if len(importances) == len(feature_names_2d):
                  feature_importance_df = pd.DataFrame({'feature': feature_names_2d, 'importance': importances})
                  feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
                  print("    最重要的前15个特征:")
                  print(feature_importance_df.head(15))
                  
                  encoded_crop_col_name_from_info = train_info_for_featurizer_crop.get('encoded_crop_label_col_name')
                  print(f"    从info中获取的编码作物列名: {encoded_crop_col_name_from_info}")
                  
                  if encoded_crop_col_name_from_info and encoded_crop_col_name_from_info in feature_names_2d:
                      try:
                          crop_feat_importance_val = feature_importance_df.loc[feature_importance_df['feature'] == encoded_crop_col_name_from_info, 'importance'].iloc[0]
                          print(f"    特征 '{encoded_crop_col_name_from_info}' 的重要性: {crop_feat_importance_val:.4f}")
                          rank_series = feature_importance_df['feature'].reset_index(drop=True)
                          rank = rank_series[rank_series == encoded_crop_col_name_from_info].index[0] + 1
                          print(f"    特征 '{encoded_crop_col_name_from_info}' 的重要性排名: {rank} / {len(feature_names_2d)}")
                      except IndexError:
                          print(f"    警告: 无法获取特征 '{encoded_crop_col_name_from_info}' 的重要性或排名。")
                  else:
                      print(f"    警告: 编码后的作物特征列 '{encoded_crop_col_name_from_info}' 未在2D特征名称列表或原始静态列中找到。")
              else:
                  print(f"    警告: 特征重要性数量 ({len(importances)}) 与2D特征名称数量 ({len(feature_names_2d)}) 不匹配。")
              print(f"  --- 特征重要性分析结束 for {model_name}_{crop_name} ---")

          # Per-Crop Evaluation
          print(f"  Evaluating {model_name}_{crop_name} on test data for {crop_name}...")
          eval_results_specific_crop = evaluate_on_test_set(
              fitted_model=fitted_model_crop, 
              test_set_name=crop_name, # test_set_name is now the crop_name
              x_test_scaled=x_test_scaled_crop, 
              y_test_original=y_test_original_crop, 
              y_scaler=y_scaler_crop # Use the crop-specific y_scaler
          )
          
          current_model_summary["metrics"] = eval_results_specific_crop.get("metrics")
          current_model_summary["error"] = eval_results_specific_crop.get("error")


          if crop_name in CROPS_FOR_DETAILED_DEBUG and eval_results_specific_crop.get("y_pred_scaled") is not None:
              print_prediction_diagnostics(
                  y_test_original_crop, 
                  eval_results_specific_crop["y_pred_scaled"],
                  eval_results_specific_crop["y_pred_original"], 
                  y_scaler_crop, # Use crop-specific y_scaler
                  crop_name
              )
          
          
          with mlflow.start_run(run_name=f"{model_name}_{crop_name}"):
              print(f"    Logging to MLflow for run: {model_name}_{crop_name}")
              mlflow.log_param("crop_name", crop_name)
              mlflow.log_param("model_name", model_name) # Log base model name
              mlflow.log_params(args.__dict__) # Log general script args
              mlflow.log_param("aggregation_functions", str(AGGREGATION_FUNCTIONS))
              for param_name, param_value in model_params_to_log.items(): # Log specific model params
                  mlflow.log_param(param_name, param_value)

              mlflow.sklearn.log_model(fitted_model_crop, artifact_path="model")

              # Scalers
              scaler_crop_dir = os.path.join("scalers", crop_name)
              os.makedirs(scaler_crop_dir, exist_ok=True)
              x_scaler_path_crop = os.path.join(scaler_crop_dir, f"{model_name}_{crop_name}_x_scaler.joblib")
              y_scaler_path_crop = os.path.join(scaler_crop_dir, f"{model_name}_{crop_name}_y_scaler.joblib")
              joblib.dump(x_scaler_crop, x_scaler_path_crop)
              joblib.dump(y_scaler_crop, y_scaler_path_crop)
              mlflow.log_artifact(x_scaler_path_crop, artifact_path=f"scalers/{crop_name}")
              mlflow.log_artifact(y_scaler_path_crop, artifact_path=f"scalers/{crop_name}")

              # Metrics
              if eval_results_specific_crop.get("metrics"):
                  for k, v in eval_results_specific_crop["metrics"].items():
                      mlflow.log_metric(k, v) # Logs R2, MAE etc. directly
              
              if eval_results_specific_crop.get("error"):
                   mlflow.log_param(f"{crop_name}_{model_name}_evaluation_error", eval_results_specific_crop.get("error"))


              # Predictions CSV
              if eval_results_specific_crop.get("y_pred_original") is not None:
                  predictions_crop_dir = os.path.join("predictions", crop_name)
                  os.makedirs(predictions_crop_dir, exist_ok=True)
                  predictions_csv_path_crop = os.path.join(predictions_crop_dir, f"{model_name}_{crop_name}_test_predictions.csv")
                  
                  predictions_df = pd.DataFrame({
                      'y_true': y_test_original_crop.ravel(),
                      'y_pred': eval_results_specific_crop["y_pred_original"].ravel()
                  })
                  predictions_df.to_csv(predictions_csv_path_crop, index=False)
                  mlflow.log_artifact(predictions_csv_path_crop, artifact_path=f"predictions/{crop_name}")

              mlflow.set_tag("status", "COMPLETED_PER_CROP")
          
          overall_summary_results_all_crops.append(current_model_summary)

  # The existing code block for model training and evaluation has been moved into the loop above.
  # overall_summary_results = [] # This is now overall_summary_results_all_crops

  # print("\n5. 训练基线模型并在所有测试集上评估... (EXISTING CODE BLOCK - TO BE REFACTORED/REMOVED)")
  # This loop will be moved inside the per-crop loop and adapted.
  # for model_config in baseline_models_config:
  #   model_name = model_config["name"]
  #   model_instance_template = model_config["instance"]
    # model_params_to_log = model_config["params"]

    # fitted_model = train_model(
    #     model_name=model_name, model_instance=model_instance_template,
    #     x_train_scaled=x_train_scaled, y_train_scaled=y_train_scaled # These need to be crop-specific
    # )
    # if not fitted_model:
    #     overall_summary_results.append({"model_name": model_name, "error": "Training failed"}) # This should be overall_summary_results_all_crops
    #     continue

    # # 特征重要性分析 (RandomForest)
    # if "RandomForest" in model_name and hasattr(fitted_model, 'feature_importances_'):
    #     print(f"\n--- {model_name} 特征重要性分析 ---")
    #     importances = fitted_model.feature_importances_
        
    #     # This feature_names_2d needs to be derived from crop-specific train_info_for_featurizer
    #     feature_names_2d = [] 
    #     temp_cols_orig = train_info_for_featurizer.get('original_temporal_cols_used', [])
    #     for temp_col_name in temp_cols_orig:
    #         for agg_func_name in AGGREGATION_FUNCTIONS:
    #             feature_names_2d.append(f"{temp_col_name}_{agg_func_name}")
        
    #     stat_cols_orig = train_info_for_featurizer.get('original_static_cols_used', []) 
    #     feature_names_2d.extend(stat_cols_orig)

    #     if len(importances) == len(feature_names_2d):
    #         feature_importance_df = pd.DataFrame({'feature': feature_names_2d, 'importance': importances})
    #         feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    #         print("  最重要的前15个特征:") 
    #         print(feature_importance_df.head(15))
            
    #         encoded_crop_col_name_from_info = train_info_for_featurizer.get('encoded_crop_label_col_name')
    #         print(f"  从info中获取的编码作物列名: {encoded_crop_col_name_from_info}")
    #         print(f"  检查 '{encoded_crop_col_name_from_info}' 是否在 stat_cols_orig ({len(stat_cols_orig)}个)中: {encoded_crop_col_name_from_info in stat_cols_orig}")
    #         print(f"  stat_cols_orig (最后5个): {stat_cols_orig[-5:]}")

    #         if encoded_crop_col_name_from_info and encoded_crop_col_name_from_info in feature_names_2d:
    #             try:
    #                 crop_feat_importance = feature_importance_df.loc[feature_importance_df['feature'] == encoded_crop_col_name_from_info, 'importance'].iloc[0]
    #                 print(f"  特征 '{encoded_crop_col_name_from_info}' 的重要性: {crop_feat_importance:.4f}")
    #                 rank_series = feature_importance_df['feature'].reset_index(drop=True)
    #                 rank = rank_series[rank_series == encoded_crop_col_name_from_info].index[0] + 1
    #                 print(f"  特征 '{encoded_crop_col_name_from_info}' 的重要性排名: {rank} / {len(feature_names_2d)}")
    #             except IndexError: 
    #                 print(f"  警告: 无法获取特征 '{encoded_crop_col_name_from_info}' 的重要性或排名。")
    #         else:
    #             print(f"  警告: 编码后的作物特征列 '{encoded_crop_col_name_from_info}' 未在2D特征名称列表或原始静态列中找到。")
    #     else:
    #         print(f"  警告: 特征重要性数量 ({len(importances)}) 与2D特征名称数量 ({len(feature_names_2d)}) 不匹配。")
    #     print("--- 特征重要性分析结束 ---")

    # with mlflow.start_run(run_name=f"{model_name}_{crop_name}"): # Crop-specific run name
    #   mlflow.log_param("crop_name", crop_name)
    #   mlflow.log_param("model_name", model_name); mlflow.log_params(args.__dict__); mlflow.log_param("aggregation_functions", str(AGGREGATION_FUNCTIONS))
    #   for param_name, param_value in model_params_to_log.items(): mlflow.log_param(param_name, param_value)
    #   mlflow.sklearn.log_model(fitted_model, "model")
      
    #   scaler_path_dir = f"scalers_{crop_name}"; os.makedirs(scaler_path_dir, exist_ok=True)
    #   # x_scaler and y_scaler need to be crop-specific
    #   x_scaler_path = os.path.join(scaler_path_dir, f"{model_name}_x_scaler.joblib"); joblib.dump(x_scaler, x_scaler_path)
    #   y_scaler_path = os.path.join(scaler_path_dir, f"{model_name}_y_scaler.joblib"); joblib.dump(y_scaler, y_scaler_path)
    #   mlflow.log_artifact(x_scaler_path, f"scalers/{crop_name}"); mlflow.log_artifact(y_scaler_path, f"scalers/{crop_name}")

    #   # Validation set evaluation needs to use crop-specific val data (if we re-introduce val split)
    #   # val_eval_results = evaluate_on_test_set(
    #   #     fitted_model, "ValidationSet", x_val_scaled_crop, y_val_original_crop, y_scaler_crop)
    #   # if val_eval_results.get("metrics"):
    #   #     for k, v in val_eval_results["metrics"].items(): mlflow.log_metric(f"ValidationSet_{k}", v)
      
    #   model_summary_entry = {"model_name": model_name, "crop_name": crop_name, "test_metrics": {}} # "validation_metrics": val_eval_results.get("metrics"),

    #   # Test set evaluation should use the matched crop-specific test set
    #   # For now, this iterates all test_sets_to_evaluate, which is not right for single crop training.
    #   # It should use the specific x_test_scaled_crop, y_test_original_crop for the current crop_name
    #   # if test_set_info["name"] == crop_name: ... (logic to use the matched test set)
    #   for test_set_info in test_sets_to_evaluate: # This needs to be restricted to current crop's test set
    #     if test_set_info["name"] == crop_name: # Only evaluate on the matched test set
    #         test_set_name = test_set_info["name"] # Should be crop_name
    #         # test_file_path = test_set_info["path"] # Already loaded as crop_test_path
    #         print(f"\n     校验并评估测试集: {test_set_name} for crop {crop_name}")
    #         try:
    #             # Data already loaded as x_test_3d_crop, y_test_original_crop, original_test_info_crop
    #             # Featurization and scaling needs to happen here for test_data_crop
    #             # test_info_for_featurizer_specific = adapt_info_keys(original_test_info_crop, test_set_name)
    #             # x_test_2d_specific = featurize_3d_to_2d(
    #             #     x_test_3d_crop, test_info_for_featurizer_specific, agg_funcs=AGGREGATION_FUNCTIONS
    #             # )
    #             # x_test_scaled_specific = x_scaler.transform(x_test_2d_specific) # Use crop-specific x_scaler

    #             # eval_results_specific = evaluate_on_test_set(
    #             #     fitted_model, test_set_name, x_test_scaled_specific, # This should be x_test_2d_crop_scaled
    #             #     y_test_original_crop, y_scaler # Use crop-specific y_scaler
    #             # )
    #             # model_summary_entry["test_metrics"] = eval_results_specific.get("metrics")
    #             pass # Placeholder for actual evaluation
    #         except Exception as e_test_set:
    #             print(f"  处理测试集 {test_set_name} 时出错: {e_test_set}"); traceback.print_exc()
    #             model_summary_entry["test_metrics"] = {"error": str(e_test_set)}
    #             if mlflow.active_run(): mlflow.log_param(f"{test_set_name}_error", str(e_test_set))
    #   overall_summary_results_all_crops.append(model_summary_entry) # Correct list
  
  
  print("\n--- Overall Summary of Results ---")
  if not overall_summary_results_all_crops:
      print("  No results to summarize.")
  else:
      for result_entry in overall_summary_results_all_crops:
          print(f"Crop: {result_entry.get('crop_name')}, Model: {result_entry.get('model_name')}")
          if result_entry.get("metrics"):
              print(f"  Metrics: {result_entry.get('metrics')}")
          if result_entry.get("error"): # Check if error key exists and has a value
              print(f"  Error: {result_entry.get('error')}")
          print("---")

  print("\n--- Baseline model evaluation (per-crop with encoded crop feature) completed ---")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="Run baseline models with encoded crop feature and per-crop evaluation."
  )
  parser.add_argument("--train_data_dir", type=str, default=DEFAULT_TRAIN_DATA_DIR,
                        help="Directory containing per-crop training NPZ files.")
  parser.add_argument("--test_data_dir", type=str, default=DEFAULT_TEST_DATA_DIR,
                        help="Directory containing per-crop test NPZ files.")
  parser.add_argument("--train_file_prefix", type=str, default='train_processed_',
                        help="Prefix for per-crop training NPZ files (e.g., 'train_processed_').")
  parser.add_argument("--test_file_prefix", type=str, default='test_processed_',
                        help="Prefix for per-crop test NPZ files (e.g., 'test_processed_').")
  parser.add_argument("--per_crop_test_dir", type=str, default=None)
  parser.add_argument("--val_split_size", type=float, default=VAL_SPLIT_SIZE)
  parser.add_argument("--random_seed", type=int, default=RANDOM_SEED)
  parser.add_argument("--mlflow_experiment_name", type=str, default=DEFAULT_MLFLOW_EXPERIMENT_NAME)
  
  args = parser.parse_args()
  main(args)