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

Debug Phase 3: Model Behavior and Evaluation Logic Validation.
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
  import evaluate
except ImportError as e:
  print(f"Error importing modules: {e}")
  # ... (错误处理与之前相同) ...
  exit(1)


# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_DATA_PATH = '/root/lanyun-tmp/datasets/us_processed_cp/train_processed.npz'
DEFAULT_TEST_DATA_PATH = '/root/lanyun-tmp/datasets/us_processed_cp/test_processed.npz'
DEFAULT_MLFLOW_EXPERIMENT_NAME = "Baseline Models - Agri Yield Prediction - DebugPhase3"

VAL_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
AGGREGATION_FUNCTIONS = ['mean', 'std']

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

CROPS_FOR_DETAILED_DEBUG = ["corn", "wheat_spring_excl_drum", "soybeans"]
ASSUMED_CROP_FEATURE_NAME_IN_STATIC_COLS = 'crop_identity'


def adapt_info_keys(original_info: Dict, context: str = "Unknown") -> Dict:
  adapted = {}
  try:
    required_source_keys = [
        'static_feature_start_row_index_in_sample_matrix',
        'num_temporal_features', 'num_static_features',
        'temporal_cols_used', 'static_cols_used'
    ]
    for key in required_source_keys:
        if key not in original_info:
            raise KeyError(f"Source key '{key}' missing in original_info for {context}.")
    adapted['static_feature_row_index'] = original_info[
        'static_feature_start_row_index_in_sample_matrix']
    num_temporal = original_info['num_temporal_features']
    num_static = original_info['num_static_features']
    adapted['temporal_cols_idx'] = list(range(num_temporal))
    adapted['static_cols_idx'] = list(range(num_temporal, num_temporal + num_static))
    if 'max_len' in original_info: adapted['max_len'] = original_info['max_len']
    adapted['original_temporal_cols_used'] = original_info['temporal_cols_used']
    adapted['original_static_cols_used'] = original_info['static_cols_used']

  except KeyError as e:
    print(f"    错误 ({context}): Error adapting info keys: {e}. Original keys: {list(original_info.keys())}")
    raise
  return adapted

def discover_test_sets(overall_test_path: str, per_crop_test_dir: str = None) -> List[Dict[str, str]]:
    test_sets = []
    if os.path.exists(overall_test_path):
        test_sets.append({"name": "Overall", "path": overall_test_path})
    else: print(f"警告: 整体测试文件未找到于 {overall_test_path}")
    search_dir = per_crop_test_dir if per_crop_test_dir else os.path.dirname(overall_test_path)
    if not os.path.isdir(search_dir):
        print(f"警告: 按作物划分的测试目录 {search_dir} 不是一个有效目录。")
        return test_sets
    pattern = os.path.join(search_dir, "test_processed_*.npz")
    per_crop_files = glob.glob(pattern)
    for crop_file_path in per_crop_files:
        if crop_file_path == overall_test_path: continue
        filename = os.path.basename(crop_file_path)
        match = re.match(r"test_processed_(.+)\.npz", filename)
        if match: test_sets.append({"name": match.group(1), "path": crop_file_path})
        else: print(f"警告: 无法从文件名 {filename} 中提取作物名称。")
    print(f"发现的测试集名称: {[ts['name'] for ts in test_sets]}")
    return test_sets


def print_info_comparison(train_info: Dict, test_info: Dict, test_set_name: str):
    pass 

def print_data_sample_diagnostics(x_3d: np.ndarray, y_original: np.ndarray,
                                  y_train_original_stats: Dict, test_set_name: str):
    pass 

def print_featurizer_output_diagnostics(x_2d: np.ndarray, test_set_name: str,
                                        num_temporal_agg_features: int):
    pass 

def print_standardization_diagnostics(
    x_scaled: np.ndarray, test_set_name: str, x_scaler: StandardScaler,
    y_scaler: StandardScaler, y_crop_stats: Dict
):
    pass 

def print_prediction_diagnostics(
    y_original: np.ndarray, y_pred_scaled: np.ndarray, y_pred_original: np.ndarray,
    y_scaler: StandardScaler, test_set_name: str
):
    print(f"\n--- '{test_set_name}' 预测值诊断 ---")
    print(f"  y_original 形状: {y_original.shape}, 前5个: {y_original[:5]}")
    print(f"  y_pred_scaled (模型直接输出) 形状: {y_pred_scaled.shape}, 前5个: {y_pred_scaled[:5]}")
    print(f"  y_pred_original (逆标准化后) 形状: {y_pred_original.shape}, 前5个: {y_pred_original[:5]}")
    
    global_y_mean_original = y_scaler.mean_[0]
    global_y_std_original = np.sqrt(y_scaler.var_[0])
    print(f"  全局训练集 y 均值 (原始尺度): {global_y_mean_original:.4f}")
    print(f"  全局训练集 y 均值 (标准化尺度): 0.0")

    if y_original.size > 0:
        crop_y_mean_original = np.mean(y_original)
        print(f"  '{test_set_name}' y 均值 (原始尺度): {crop_y_mean_original:.4f}")
        
        crop_y_mean_scaled = (crop_y_mean_original - global_y_mean_original) / global_y_std_original
        print(f"  '{test_set_name}' y 均值 (转换到全局标准化尺度后): {crop_y_mean_scaled:.4f}")
        
        if y_pred_scaled.size > 0:
            mean_pred_scaled = np.mean(y_pred_scaled)
            print(f"  '{test_set_name}' y_pred_scaled 均值: {mean_pred_scaled:.4f}")
    print("--- 预测值诊断结束 ---")


def train_model(
    model_name: str, model_instance: Any, x_train_scaled: np.ndarray, y_train_scaled: np.ndarray
) -> Any:
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
    print(f"   Evaluating on test set: {test_set_name}...")
    results = {"metrics": None, "y_pred_original": None, "y_pred_scaled": None, "error": None}
    try:
        y_pred_scaled_raw = fitted_model.predict(x_test_scaled) 
        results["y_pred_scaled"] = y_pred_scaled_raw 
        
        y_pred_original = y_scaler.inverse_transform(y_pred_scaled_raw.reshape(-1, 1)).ravel()
        results["y_pred_original"] = y_pred_original
        
        print(f"    evaluate_on_test_set ({test_set_name}):")
        print(f"      y_test_original shape: {y_test_original.shape}, first 5: {y_test_original[:5]}")
        print(f"      y_pred_original shape: {y_pred_original.shape}, first 5: {y_pred_original[:5]}")
        if y_test_original.shape != y_pred_original.shape:
            print(f"      警告! y_test_original 和 y_pred_original 形状不匹配!")
            
        metrics = evaluate.calculate_regression_metrics(y_test_original, y_pred_original)
        results["metrics"] = metrics
        print(f"   Metrics for {test_set_name}: {metrics}")
    except Exception as e:
        print(f"Error during evaluation on {test_set_name} for model {getattr(fitted_model, '__class__', type(fitted_model).__name__)}: {e}")
        traceback.print_exc(); results["error"] = str(e)
    return results


def main(args):
  print(f"--- 排查阶段三：模型行为与评估逻辑校验 ---")
  mlflow.set_experiment(args.mlflow_experiment_name)

  print(f"\n1. 从 '{args.train_data_path}' 加载并准备训练/验证数据...")
  try:
    x_full_train_3d, y_full_train_original, original_train_info = load_data.load_npz_data(
        args.train_data_path)
    train_info_for_featurizer = adapt_info_keys(original_train_info, "训练集")
    
    y_train_original_stats = {"mean": np.mean(y_full_train_original), "std": np.std(y_full_train_original),
                              "min": np.min(y_full_train_original), "max": np.max(y_full_train_original)}
    x_train_3d, x_val_3d, y_train_original, y_val_original = load_data.split_data(
        x_full_train_3d, y_full_train_original,
        test_size=args.val_split_size, random_state=args.random_seed)
  except Exception as e: print(f"加载或拆分训练数据时出错: {e}"); traceback.print_exc(); return

  print(f"\n2. 将训练集和验证集的3D特征转换为2D...")
  try:
    x_train_2d = featurize_3d_to_2d(x_train_3d, train_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
    x_val_2d = featurize_3d_to_2d(x_val_3d, train_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
  except Exception as e: print(f"转换3D特征到2D时出错: {e}"); traceback.print_exc(); return

  print("\n3. 标准化特征和目标变量 (使用训练数据拟合)...")
  x_scaler = StandardScaler(); y_scaler = StandardScaler()
  try:
    y_train_original_reshaped = y_train_original.reshape(-1, 1)
    x_train_scaled = x_scaler.fit_transform(x_train_2d)
    y_train_scaled = y_scaler.fit_transform(y_train_original_reshaped)
    x_val_scaled = x_scaler.transform(x_val_2d)
    print("   特征和目标标准化器已在训练数据上拟合。")
  except Exception as e: print(f"标准化数据时出错: {e}"); traceback.print_exc(); return

  print("\n4. 发现测试集...")
  test_data_directory = args.per_crop_test_dir if args.per_crop_test_dir else os.path.dirname(args.test_data_path)
  test_sets_to_evaluate = discover_test_sets(args.test_data_path, test_data_directory)
  if not test_sets_to_evaluate: print("错误: 未找到任何测试集。"); return
  
  baseline_models_config = [
      {"name": "MeanPredictor", "instance": DummyRegressor(strategy="mean"), "params": {"strategy": "mean"}},
      {"name": "LinearRegression", "instance": LinearRegression(), "params": {}},
      {"name": "RandomForestRegressor_Simple",
       "instance": RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                                       random_state=args.random_seed, n_jobs=-1),
       "params": {"n_estimators": RF_N_ESTIMATORS, "max_depth": RF_MAX_DEPTH, "random_state": args.random_seed}}
  ]
  
  overall_summary_results = []

  print("\n5. 训练基线模型并在所有测试集上评估 (包含阶段三诊断)...")
  for model_config in baseline_models_config:
    model_name = model_config["name"]
    model_instance_template = model_config["instance"]
    model_params_to_log = model_config["params"]

    fitted_model = train_model(
        model_name=model_name, model_instance=model_instance_template,
        x_train_scaled=x_train_scaled, y_train_scaled=y_train_scaled
    )
    if not fitted_model:
        print(f"模型 {model_name} 训练失败，跳过评估。")
        overall_summary_results.append({"model_name": model_name, "error": "Training failed"})
        continue

    if model_name == "RandomForestRegressor_Simple" and hasattr(fitted_model, 'feature_importances_'):
        print(f"\n--- RandomForestRegressor_Simple 特征重要性分析 ---")
        importances = fitted_model.feature_importances_
        feature_names_2d = []
        temp_cols_orig = train_info_for_featurizer.get('original_temporal_cols_used', [])
        for temp_col_name in temp_cols_orig:
            for agg_func_name in AGGREGATION_FUNCTIONS: # Corrected variable name
                feature_names_2d.append(f"{temp_col_name}_{agg_func_name}")
        stat_cols_orig = train_info_for_featurizer.get('original_static_cols_used', [])
        feature_names_2d.extend(stat_cols_orig)

        if len(importances) == len(feature_names_2d):
            feature_importance_df = pd.DataFrame({'feature': feature_names_2d, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
            print("  最重要的前10个特征:")
            print(feature_importance_df.head(10))
            crop_identity_col_name = original_train_info.get('crop_identity_col_used')
            if crop_identity_col_name and crop_identity_col_name in stat_cols_orig:
                try:
                    crop_feat_idx_in_2d = feature_names_2d.index(crop_identity_col_name)
                    crop_feat_importance = importances[crop_feat_idx_in_2d]
                    print(f"  特征 '{crop_identity_col_name}' 的重要性: {crop_feat_importance:.4f}")
                    rank = feature_importance_df[feature_importance_df['feature'] == crop_identity_col_name].index[0] +1
                    print(f"  特征 '{crop_identity_col_name}' 的重要性排名: {rank} / {len(feature_names_2d)}")
                except ValueError:
                    print(f"  警告: 未能在2D特征列表中找到作物身份特征列名 '{crop_identity_col_name}'。")
                except IndexError:
                    print(f"  警告: 无法获取特征 '{crop_identity_col_name}' 的排名。")
            else:
                print(f"  信息: 未在元数据中找到 'crop_identity_col_used' ({crop_identity_col_name}) "
                      f"或者它不在静态特征列表 ({stat_cols_orig}) 中。无法定位作物特征重要性。")
        else:
            print(f"  警告: 特征重要性数量 ({len(importances)}) 与2D特征名称数量 ({len(feature_names_2d)}) 不匹配。")
        print("--- 特征重要性分析结束 ---")

    with mlflow.start_run(run_name=model_name + "_DebugRunPhase3"):
      mlflow.log_param("model_name", model_name); mlflow.log_params(args.__dict__); mlflow.log_param("aggregation_functions", str(AGGREGATION_FUNCTIONS))
      for param_name, param_value in model_params_to_log.items(): mlflow.log_param(param_name, param_value)
      mlflow.sklearn.log_model(fitted_model, "model")
      scaler_path_dir = "scalers"; os.makedirs(scaler_path_dir, exist_ok=True)
      x_scaler_path = os.path.join(scaler_path_dir, f"{model_name}_x_scaler.joblib"); joblib.dump(x_scaler, x_scaler_path)
      y_scaler_path = os.path.join(scaler_path_dir, f"{model_name}_y_scaler.joblib"); joblib.dump(y_scaler, y_scaler_path)
      mlflow.log_artifact(x_scaler_path, "scalers"); mlflow.log_artifact(y_scaler_path, "scalers")

      val_eval_results = evaluate_on_test_set(
          fitted_model, "ValidationSet", x_val_scaled, y_val_original, y_scaler)
      if val_eval_results.get("metrics"):
          for k, v in val_eval_results["metrics"].items(): mlflow.log_metric(f"ValidationSet_{k}", v)
      
      model_summary_entry = {"model_name": model_name, "validation_metrics": val_eval_results.get("metrics"), "per_crop_test_metrics": {}}

      for test_set_info in test_sets_to_evaluate:
        test_set_name = test_set_info["name"]
        test_file_path = test_set_info["path"]
        print(f"\n     校验并评估测试集: {test_set_name} from {test_file_path}")
        try:
            x_test_3d_specific, y_test_original_specific, original_test_info_specific = \
                load_data.load_npz_data(test_file_path)
            if x_test_3d_specific.size == 0: print(f"  测试集 {test_set_name} 为空，跳过。"); continue
            
            test_info_for_featurizer_specific = adapt_info_keys(original_test_info_specific, test_set_name)
            x_test_2d_specific = featurize_3d_to_2d(
                x_test_3d_specific, test_info_for_featurizer_specific, agg_funcs=AGGREGATION_FUNCTIONS # Corrected variable name
            )
            x_test_scaled_specific = x_scaler.transform(x_test_2d_specific)

            eval_results_specific = evaluate_on_test_set(
                fitted_model, test_set_name, x_test_scaled_specific,
                y_test_original_specific, y_scaler
            )
            model_summary_entry["per_crop_test_metrics"][test_set_name] = eval_results_specific.get("metrics")

            if eval_results_specific.get("metrics"):
                for k, v in eval_results_specific["metrics"].items(): mlflow.log_metric(f"{test_set_name}_test_{k}", v)
            
            if test_set_name in CROPS_FOR_DETAILED_DEBUG and eval_results_specific.get("y_pred_scaled") is not None:
                print_prediction_diagnostics(
                    y_test_original_specific,
                    eval_results_specific["y_pred_scaled"],
                    eval_results_specific["y_pred_original"],
                    y_scaler,
                    test_set_name
                )
            
            if eval_results_specific.get("y_pred_original") is not None:
                # Placeholder for saving predictions CSV if needed in future
                pass

        except Exception as e_test_set:
            print(f"  处理测试集 {test_set_name} 时出错: {e_test_set}"); traceback.print_exc()
            model_summary_entry["per_crop_test_metrics"][test_set_name] = {"error": str(e_test_set)}
            if mlflow.active_run(): mlflow.log_param(f"{test_set_name}_error", str(e_test_set))
      
      if mlflow.active_run(): mlflow.set_tag("status", "DEBUG_PHASE3_COMPLETED")
      overall_summary_results.append(model_summary_entry)
  
  print("\n--- 所有基线模型在各测试集上的评估汇总 ---")
  for result_entry in overall_summary_results:
      print(f"模型: {result_entry.get('model_name')}")
      if result_entry.get("validation_metrics"):
          print(f"  验证集指标: {result_entry.get('validation_metrics')}")
      if result_entry.get("per_crop_test_metrics"):
          for test_name, metrics in result_entry.get("per_crop_test_metrics", {}).items():
              print(f"  测试集 ({test_name}) 指标: {metrics}")
      if result_entry.get("error"):
           print(f"  训练错误: {result_entry.get('error')}")
      print("-" * 40)


  print("\n--- 排查阶段三完成 ---")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="Run baseline models (Debug Phase 3: Model Behavior Validation)."
  )
  parser.add_argument("--train_data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH)
  parser.add_argument("--test_data_path", type=str, default=DEFAULT_TEST_DATA_PATH)
  parser.add_argument("--per_crop_test_dir", type=str, default=None)
  parser.add_argument("--val_split_size", type=float, default=VAL_SPLIT_SIZE)
  parser.add_argument("--random_seed", type=int, default=RANDOM_SEED)
  parser.add_argument("--mlflow_experiment_name", type=str, default=DEFAULT_MLFLOW_EXPERIMENT_NAME)
  
  args = parser.parse_args()
  main(args)
