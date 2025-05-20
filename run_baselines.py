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

Step 6: Integrate MLflow for experiment tracking.
"""

import argparse
import numpy as np
import os
import traceback
from typing import Dict, Any, List
import joblib # 用于保存scaler
import pandas as pd # 用于保存预测结果

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

import mlflow # 导入 mlflow
import mlflow.sklearn # 导入 mlflow.sklearn

try:
  import load_data
  from src.tabpfn.agri_utils.baseline_featurizer import featurize_3d_to_2d
  import evaluate
except ImportError as e:
  print(f"Error importing modules: {e}")
  print("Please ensure 'load_data.py', 'evaluate.py' are accessible and "
        "the 'src' directory is correctly structured.")
  print("Current working directory:", os.getcwd())
  exit(1)


# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_DATA_PATH = '/root/lanyun-tmp/datasets/us_processed_cp/train_processed.npz'
DEFAULT_TEST_DATA_PATH = '/root/lanyun-tmp/datasets/us_processed_cp/test_processed.npz'
DEFAULT_MLFLOW_EXPERIMENT_NAME = "Baseline Models - Agri Yield Prediction" # MLflow 实验名称

VAL_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
AGGREGATION_FUNCTIONS = ['mean', 'std']

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10


def adapt_info_keys(original_info: Dict) -> Dict:
  """Adapts keys from load_data's info dict to what featurizer expects."""
  adapted = {}
  try:
    adapted['static_feature_row_index'] = original_info[
        'static_feature_start_row_index_in_sample_matrix']
    num_temporal = original_info['num_temporal_features']
    num_static = original_info['num_static_features']
    adapted['temporal_cols_idx'] = list(range(num_temporal))
    adapted['static_cols_idx'] = list(range(num_temporal, num_temporal + num_static))
    if 'max_len' in original_info:
        adapted['max_len'] = original_info['max_len']
  except KeyError as e:
    raise KeyError(
        f"Error adapting info keys. Original info dict is missing a required "
        f"source key for mapping: {e}. Original keys: {list(original_info.keys())}"
    ) from e
  return adapted

def train_and_evaluate_model( # 返回训练好的模型和预测结果
    model_name: str,
    model_instance: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val_original: np.ndarray,
    x_test: np.ndarray,
    y_test_original: np.ndarray,
    y_scaler: StandardScaler
) -> Dict:
    """Trains a model, evaluates it, and returns model, metrics, and predictions."""
    print(f"\n--- Training and Evaluating: {model_name} ---")
    # model_instance 在这里被 fit，所以它会被修改并成为训练好的模型
    results = {
        "model_name": model_name,
        "fitted_model": None, # 初始化
        "validation_metrics": None,
        "test_metrics": None,
        "y_val_pred_original": None,
        "y_test_pred_original": None,
        "error": None
        }

    try:
        print(f"   Training {model_name}...")
        model_instance.fit(x_train, y_train.ravel())
        results["fitted_model"] = model_instance # 保存训练好的模型
        print(f"   {model_name} trained successfully.")

        print("   Evaluating on validation set...")
        y_val_pred_scaled = model_instance.predict(x_val)
        y_val_pred_original = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
        results["y_val_pred_original"] = y_val_pred_original
        val_metrics = evaluate.calculate_regression_metrics(y_val_original, y_val_pred_original)
        results["validation_metrics"] = val_metrics
        print(f"   Validation Metrics: {val_metrics}")

        print("   Evaluating on test set...")
        y_test_pred_scaled = model_instance.predict(x_test)
        y_test_pred_original = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
        results["y_test_pred_original"] = y_test_pred_original
        test_metrics = evaluate.calculate_regression_metrics(y_test_original, y_test_pred_original)
        results["test_metrics"] = test_metrics
        print(f"   Test Metrics: {test_metrics}")

    except Exception as e:
        print(f"Error during training or evaluation of {model_name}: {e}")
        traceback.print_exc()
        results["error"] = str(e)
    
    return results


def main(args):
  """主函数：加载数据，转换特征，标准化，训练和评估多个基线模型，并使用MLflow追踪。"""
  print(f"--- 步骤6：集成 MLflow 进行实验追踪 ---")
  print(f"MLflow Experiment Name: {args.mlflow_experiment_name}")
  print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

  # 设置 MLflow 实验
  try:
    experiment = mlflow.get_experiment_by_name(args.mlflow_experiment_name)
    if experiment is None:
        print(f"Experiment '{args.mlflow_experiment_name}' not found, creating new experiment.")
        mlflow.create_experiment(args.mlflow_experiment_name)
    mlflow.set_experiment(args.mlflow_experiment_name)
  except Exception as e:
      print(f"Could not set MLflow experiment: {e}")
      print("Please ensure MLflow server is running or tracking URI is correctly configured.")
      # return # 可以选择在这里退出，或者让后续的mlflow调用失败

  train_data_path = args.train_data_path
  test_data_path = args.test_data_path

  # ... (数据加载、拆分、3D->2D转换、标准化的代码与步骤5相同) ...
  # 1. 加载数据
  print(f"\n1. 从 '{train_data_path}' 加载原始训练数据...")
  try:
    x_full_train_3d, y_full_train_original, original_train_info = load_data.load_npz_data(
        train_data_path)
    train_info_for_featurizer = adapt_info_keys(original_train_info)
  except Exception as e:
    print(f"加载或适配原始训练数据时出错: {e}")
    traceback.print_exc()
    return

  print(f"\n   将原始训练数据拆分为训练集和验证集 (验证集比例: {args.val_split_size})...")
  try:
    x_train_3d, x_val_3d, y_train_original, y_val_original = load_data.split_data(
        x_full_train_3d,
        y_full_train_original,
        test_size=args.val_split_size,
        random_state=args.random_seed)
  except Exception as e:
    print(f"拆分训练数据时出错: {e}")
    traceback.print_exc()
    return

  print(f"\n   从 '{test_data_path}' 加载测试数据...")
  try:
    x_test_3d, y_test_original, original_test_info = load_data.load_npz_data(test_data_path)
    test_info_for_featurizer = adapt_info_keys(original_test_info)
  except Exception as e:
    print(f"加载或适配测试数据时出错: {e}")
    traceback.print_exc()
    return

  # 2. 3D到2D特征转换
  print(f"\n2. 使用 {AGGREGATION_FUNCTIONS} 将3D特征转换为2D特征...")
  try:
    x_train_2d = featurize_3d_to_2d(x_train_3d, train_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
    x_val_2d = featurize_3d_to_2d(x_val_3d, train_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
    x_test_2d = featurize_3d_to_2d(x_test_3d, test_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
  except Exception as e:
    print(f"转换3D特征到2D时出错: {e}")
    traceback.print_exc()
    return

  # 3. 特征和目标标准化
  print("\n3. 标准化特征和目标变量...")
  x_scaler = StandardScaler()
  y_scaler = StandardScaler()
  try:
    y_train_original_reshaped = y_train_original.reshape(-1, 1)
    x_train_scaled = x_scaler.fit_transform(x_train_2d)
    y_train_scaled = y_scaler.fit_transform(y_train_original_reshaped)

    x_val_scaled = x_scaler.transform(x_val_2d)
    x_test_scaled = x_scaler.transform(x_test_2d)
    print("   特征和目标标准化完成。")
  except Exception as e:
    print(f"标准化数据时出错: {e}")
    traceback.print_exc()
    return

  # 定义要训练和评估的基线模型列表
  baseline_models_config = [
      {
          "name": "MeanPredictor",
          "instance": DummyRegressor(strategy="mean"),
          "params": {"strategy": "mean"} # 记录 DummyRegressor 的参数
      },
      {
          "name": "LinearRegression",
          "instance": LinearRegression(),
          "params": {} # LinearRegression 使用默认参数
      },
      {
          "name": "RandomForestRegressor_Simple",
          "instance": RandomForestRegressor(
              n_estimators=RF_N_ESTIMATORS,
              max_depth=RF_MAX_DEPTH,
              random_state=args.random_seed,
              n_jobs=-1
          ),
          "params": { # 记录 RandomForest 的参数
              "n_estimators": RF_N_ESTIMATORS,
              "max_depth": RF_MAX_DEPTH,
              "random_state": args.random_seed
          }
      }
  ]

  all_run_results = []

  print("\n4. 训练、评估并使用 MLflow 追踪所有基线模型...")
  for model_config in baseline_models_config:
    model_name = model_config["name"]
    model_instance = model_config["instance"] # model_instance 会在 train_and_evaluate_model 中被 fit
    model_params_to_log = model_config["params"]

    with mlflow.start_run(run_name=model_name): # 为每个模型开始一个新的 MLflow run
      print(f"\nStarting MLflow run for: {model_name}")
      
      # 记录一般参数
      mlflow.log_param("model_name", model_name)
      mlflow.log_param("random_seed", args.random_seed)
      mlflow.log_param("val_split_size", args.val_split_size)
      mlflow.log_param("aggregation_functions", str(AGGREGATION_FUNCTIONS))
      
      # 记录模型特定参数
      for param_name, param_value in model_params_to_log.items():
          mlflow.log_param(param_name, param_value)

      # 训练和评估
      # train_and_evaluate_model 现在返回一个包含 "fitted_model" 的字典
      run_result_dict = train_and_evaluate_model(
          model_name=model_name, # 传递给函数内打印
          model_instance=model_instance, # 这个实例会被fit
          x_train=x_train_scaled,
          y_train=y_train_scaled,
          x_val=x_val_scaled,
          y_val_original=y_val_original,
          x_test=x_test_scaled,
          y_test_original=y_test_original,
          y_scaler=y_scaler
      )
      all_run_results.append(run_result_dict)

      if run_result_dict.get("error"):
          mlflow.set_tag("status", "FAILED")
          mlflow.log_param("error_message", run_result_dict["error"])
          print(f"MLflow run for {model_name} marked as FAILED.")
      else:
          # 记录指标
          if run_result_dict.get("validation_metrics"):
              for metric_name, metric_value in run_result_dict["validation_metrics"].items():
                  mlflow.log_metric(f"val_{metric_name}", metric_value)
          if run_result_dict.get("test_metrics"):
              for metric_name, metric_value in run_result_dict["test_metrics"].items():
                  mlflow.log_metric(f"test_{metric_name}", metric_value)
          
          # 记录模型
          fitted_model = run_result_dict.get("fitted_model")
          if fitted_model:
              mlflow.sklearn.log_model(fitted_model, "model")
              print(f"   Logged model '{model_name}' to MLflow.")

          # (可选) 记录 Scalers 作为 artifact
          # 为了简单起见，我们可以将它们保存在临时文件中然后记录
          scaler_path_dir = "scalers"
          if not os.path.exists(scaler_path_dir):
              os.makedirs(scaler_path_dir)
          
          x_scaler_path = os.path.join(scaler_path_dir, f"{model_name}_x_scaler.joblib")
          y_scaler_path = os.path.join(scaler_path_dir, f"{model_name}_y_scaler.joblib")
          joblib.dump(x_scaler, x_scaler_path)
          joblib.dump(y_scaler, y_scaler_path)
          mlflow.log_artifact(x_scaler_path, artifact_path="scalers")
          mlflow.log_artifact(y_scaler_path, artifact_path="scalers")
          print(f"   Logged scalers to MLflow artifacts path 'scalers'.")

          # (可选) 记录测试集预测结果作为 artifact
          if run_result_dict.get("y_test_pred_original") is not None:
              predictions_df = pd.DataFrame({
                  'y_true': y_test_original.ravel(),
                  'y_pred': run_result_dict["y_test_pred_original"].ravel()
              })
              predictions_path_dir = "predictions"
              if not os.path.exists(predictions_path_dir):
                  os.makedirs(predictions_path_dir)
              predictions_csv_path = os.path.join(predictions_path_dir, f"{model_name}_test_predictions.csv")
              predictions_df.to_csv(predictions_csv_path, index=False)
              mlflow.log_artifact(predictions_csv_path, artifact_path="predictions")
              print(f"   Logged test predictions to MLflow artifacts path 'predictions'.")

          mlflow.set_tag("status", "SUCCESS")
          print(f"MLflow run for {model_name} completed and logged.")
  
  print("\n--- 所有基线模型评估汇总 (与步骤5相同，MLflow是主要记录方式) ---")
  for result in all_run_results:
      print(f"模型: {result.get('model_name')}")
      if "error" in result and result['error']:
          print(f"  错误: {result['error']}")
      else:
          print(f"  验证集指标: {result.get('validation_metrics')}")
          print(f"  测试集指标: {result.get('test_metrics')}")
      print("-" * 30)

  print("\n--- 所有基线模型训练、评估和MLflow追踪完成 ---")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="Run baseline models for agricultural yield prediction with MLflow tracking."
  )
  parser.add_argument(
      "--train_data_path", type=str, default=DEFAULT_TRAIN_DATA_PATH,
      help="Path to the training data NPZ file (train_processed.npz)")
  parser.add_argument(
      "--test_data_path", type=str, default=DEFAULT_TEST_DATA_PATH,
      help="Path to the test data NPZ file (test_processed.npz)")
  parser.add_argument(
      "--val_split_size", type=float, default=VAL_SPLIT_SIZE,
      help="Fraction of training data to use for validation.")
  parser.add_argument(
      "--random_seed", type=int, default=RANDOM_SEED, help="Random seed.")
  parser.add_argument(
      "--mlflow_experiment_name", type=str, default=DEFAULT_MLFLOW_EXPERIMENT_NAME,
      help="Name of the MLflow experiment to use/create.")
  
  args = parser.parse_args()
  main(args)
