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

Step 4: Training and evaluation workflow for a single baseline model.
"""

import argparse
import numpy as np
import os
import traceback
from typing import Dict, Any # 添加 Any

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression # 导入线性回归模型

try:
  import load_data
  from src.tabpfn.agri_utils.baseline_featurizer import featurize_3d_to_2d
  import evaluate # 导入评估模块
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

VAL_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
AGGREGATION_FUNCTIONS = ['mean', 'std']


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

def train_and_evaluate_model(
    model_name: str,
    model_instance: Any, # scikit-learn 模型实例
    x_train: np.ndarray,
    y_train: np.ndarray, # 标准化后的y_train_scaled (n_samples, 1)
    x_val: np.ndarray,
    y_val_original: np.ndarray, # 原始尺度的y_val (n_samples,)
    x_test: np.ndarray,
    y_test_original: np.ndarray, # 原始尺度的y_test (n_samples,)
    y_scaler: StandardScaler
) -> Dict:
    """Trains a model, evaluates it on validation and test sets."""
    print(f"\n--- Training and Evaluating: {model_name} ---")
    results = {"model_name": model_name}

    try:
        # 训练模型
        # 注意: scikit-learn的fit通常期望y是1D的，所以使用.ravel()
        print(f"   Training {model_name}...")
        model_instance.fit(x_train, y_train.ravel())
        print(f"   {model_name} trained successfully.")

        # 在验证集上预测和评估
        print("   Evaluating on validation set...")
        y_val_pred_scaled = model_instance.predict(x_val)
        # 逆标准化预测结果
        y_val_pred_original = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
        
        val_metrics = evaluate.calculate_regression_metrics(y_val_original, y_val_pred_original)
        results["validation_metrics"] = val_metrics
        print(f"   Validation Metrics: {val_metrics}")

        # 在测试集上预测和评估
        print("   Evaluating on test set...")
        y_test_pred_scaled = model_instance.predict(x_test)
        # 逆标准化预测结果
        y_test_pred_original = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

        test_metrics = evaluate.calculate_regression_metrics(y_test_original, y_test_pred_original)
        results["test_metrics"] = test_metrics
        print(f"   Test Metrics: {test_metrics}")

    except Exception as e:
        print(f"Error during training or evaluation of {model_name}: {e}")
        traceback.print_exc()
        results["error"] = str(e)
    
    return results


def main(args):
  """主函数：加载数据，转换特征，标准化，训练和评估单个模型。"""
  print("--- 步骤4：单个基线模型的训练与评估流程 ---")

  train_data_path = args.train_data_path
  test_data_path = args.test_data_path

  if not os.path.exists(train_data_path):
    print(f"错误：训练数据文件未找到于 {train_data_path}")
    return
  if not os.path.exists(test_data_path):
    print(f"错误：测试数据文件未找到于 {test_data_path}")
    return

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
        y_full_train_original, # 使用带 _original 后缀的变量以示区分
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
    y_train_scaled = y_scaler.fit_transform(y_train_original_reshaped) # y_train_scaled 是标准化后的

    x_val_scaled = x_scaler.transform(x_val_2d)
    # y_val_scaled 不需要显式创建，因为我们直接用 y_val_original 进行评估
    x_test_scaled = x_scaler.transform(x_test_2d)
    # y_test_scaled 也不需要显式创建
    print("   特征和目标标准化完成。")
  except Exception as e:
    print(f"标准化数据时出错: {e}")
    traceback.print_exc()
    return

  # 4. 训练和评估单个基线模型 (例如，LinearRegression)
  print("\n4. 训练和评估基线模型...")
  
  # 实例化模型
  linear_model = LinearRegression()

  # 调用训练和评估函数
  # 注意：传递给 y_train 的是 y_train_scaled
  # 传递给 y_val_original 和 y_test_original 的是原始未缩放的目标值
  lr_results = train_and_evaluate_model(
      model_name="LinearRegression",
      model_instance=linear_model,
      x_train=x_train_scaled,
      y_train=y_train_scaled, # 使用标准化后的y进行训练
      x_val=x_val_scaled,
      y_val_original=y_val_original, # 使用原始y进行评估
      x_test=x_test_scaled,
      y_test_original=y_test_original, # 使用原始y进行评估
      y_scaler=y_scaler # 传递y_scaler用于逆标准化
  )
  
  print(f"\nLinear Regression 最终结果: {lr_results}")

  print("\n--- 单个基线模型训练与评估完成 ---")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description="Run baseline models for agricultural yield prediction."
  )
  parser.add_argument(
      "--train_data_path",
      type=str,
      default=DEFAULT_TRAIN_DATA_PATH,
      help="Path to the training data NPZ file (train_processed.npz)",
  )
  parser.add_argument(
      "--test_data_path",
      type=str,
      default=DEFAULT_TEST_DATA_PATH,
      help="Path to the test data NPZ file (test_processed.npz)",
  )
  parser.add_argument(
      "--val_split_size",
      type=float,
      default=VAL_SPLIT_SIZE,
      help="Fraction of training data to use for validation.",
  )
  parser.add_argument(
      "--random_seed", type=int, default=RANDOM_SEED, help="Random seed."
  )
  args = parser.parse_args()

  main(args)
