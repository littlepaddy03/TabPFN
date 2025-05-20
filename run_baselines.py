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

Step 2: Data loading and initial 3D to 2D transformation validation.
(Applied fix for KeyError and TypeError in info dictionary processing)
"""

import argparse
import numpy as np
import os # 用于路径操作
from typing import Dict # 用于类型提示

# 假设 load_data.py 在项目根目录
# 并且 baseline_featurizer.py 在 src/tabpfn/agri_utils/
try:
  import load_data # 假设 load_data.py 在同一目录或PYTHONPATH中
  from src.tabpfn.agri_utils.baseline_featurizer import featurize_3d_to_2d
except ImportError as e:
  print(f"Error importing modules: {e}")
  print("Please ensure 'load_data.py' is accessible and the 'src' directory"
        " is correctly structured and in PYTHONPATH if necessary.")
  print("Current working directory:", os.getcwd())
  exit(1)


# --- 配置参数 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 从您的日志中获取路径作为默认值，如果命令行未提供
DEFAULT_TRAIN_DATA_PATH = '/root/lanyun-tmp/datasets/us_processed_cp/train_processed.npz'
DEFAULT_TEST_DATA_PATH = '/root/lanyun-tmp/datasets/us_processed_cp/test_processed.npz'


VAL_SPLIT_SIZE = 0.2
RANDOM_SEED = 42
AGGREGATION_FUNCTIONS = ['mean', 'std']


def adapt_info_keys(original_info: Dict) -> Dict:
  """Adapts keys from load_data's info dict to what featurizer expects.

  It now generates integer column indices based on num_temporal_features
  and num_static_features.
  """
  adapted = {}
  try:
    adapted['static_feature_row_index'] = original_info[
        'static_feature_start_row_index_in_sample_matrix']

    num_temporal = original_info['num_temporal_features']
    num_static = original_info['num_static_features']

    # 生成整数索引列表
    # 假设X的最后一维的列是先排列所有时序特征，然后排列所有静态特征
    adapted['temporal_cols_idx'] = list(range(num_temporal))
    adapted['static_cols_idx'] = list(range(num_temporal, num_temporal + num_static))

    # 验证生成的索引列表长度是否与原始名称列表长度一致 (可选的健全性检查)
    if len(adapted['temporal_cols_idx']) != len(original_info['temporal_cols_used']):
        print("警告: 生成的 temporal_cols_idx 长度与 temporal_cols_used 长度不匹配。")
    if len(adapted['static_cols_idx']) != len(original_info['static_cols_used']):
        print("警告: 生成的 static_cols_idx 长度与 static_cols_used 长度不匹配。")


    if 'max_len' in original_info:
        adapted['max_len'] = original_info['max_len']

  except KeyError as e:
    raise KeyError(
        f"Error adapting info keys. Original info dict is missing a required "
        f"source key for mapping: {e}. Original keys: {list(original_info.keys())}"
    ) from e
  return adapted


def main(args):
  """主函数：加载数据，执行3D到2D转换，并验证形状。"""
  print("--- 步骤2：数据加载与初步转换验证 (已修正键名和类型问题) ---")

  train_data_path = args.train_data_path
  test_data_path = args.test_data_path

  if not os.path.exists(train_data_path):
    print(f"错误：训练数据文件未找到于 {train_data_path}")
    return
  if not os.path.exists(test_data_path):
    print(f"错误：测试数据文件未找到于 {test_data_path}")
    return

  print(f"\n1. 从 '{train_data_path}' 加载原始训练数据...")
  try:
    x_full_train_3d, y_full_train, original_train_info = load_data.load_npz_data(
        train_data_path)
    print("原始训练数据加载成功。")
    print(f"   X_full_train_3d 形状: {x_full_train_3d.shape}, dtype: {x_full_train_3d.dtype}")
    print(f"   y_full_train 形状: {y_full_train.shape}, dtype: {y_full_train.dtype}")
    print(f"   original_train_info 键: {list(original_train_info.keys())}")
    train_info_for_featurizer = adapt_info_keys(original_train_info)
    print(f"   适配后的 train_info_for_featurizer (部分内容):")
    print(f"     static_feature_row_index: {train_info_for_featurizer['static_feature_row_index']}")
    print(f"     temporal_cols_idx (前5个): {train_info_for_featurizer['temporal_cols_idx'][:5]}... (总计 {len(train_info_for_featurizer['temporal_cols_idx'])} 个)")
    print(f"     static_cols_idx (前5个): {train_info_for_featurizer['static_cols_idx'][:5]}... (总计 {len(train_info_for_featurizer['static_cols_idx'])} 个)")

  except Exception as e:
    print(f"加载或适配原始训练数据时出错: {e}")
    # 打印更详细的堆栈跟踪以帮助调试
    import traceback
    traceback.print_exc()
    return

  print(f"\n2. 将原始训练数据拆分为训练集和验证集 (验证集比例: {args.val_split_size})...")
  try:
    x_train_3d, x_val_3d, y_train, y_val = load_data.split_data(
        x_full_train_3d,
        y_full_train,
        test_size=args.val_split_size,
        random_state=args.random_seed)
    print("数据拆分成功。")
    print(f"   X_train_3d 形状: {x_train_3d.shape}, y_train 形状: {y_train.shape}")
    print(f"   X_val_3d 形状: {x_val_3d.shape}, y_val 形状: {y_val.shape}")
  except Exception as e:
    print(f"拆分训练数据时出错: {e}")
    import traceback
    traceback.print_exc()
    return

  print(f"\n3. 从 '{test_data_path}' 加载测试数据...")
  try:
    x_test_3d, y_test, original_test_info = load_data.load_npz_data(test_data_path)
    print("测试数据加载成功。")
    print(f"   X_test_3d 形状: {x_test_3d.shape}, dtype: {x_test_3d.dtype}")
    print(f"   y_test 形状: {y_test.shape}, dtype: {y_test.dtype}")
    print(f"   original_test_info 键: {list(original_test_info.keys())}")
    test_info_for_featurizer = adapt_info_keys(original_test_info)
    print(f"   适配后的 test_info_for_featurizer (部分内容):")
    print(f"     static_feature_row_index: {test_info_for_featurizer['static_feature_row_index']}")
    print(f"     temporal_cols_idx (前5个): {test_info_for_featurizer['temporal_cols_idx'][:5]}... (总计 {len(test_info_for_featurizer['temporal_cols_idx'])} 个)")
    print(f"     static_cols_idx (前5个): {test_info_for_featurizer['static_cols_idx'][:5]}... (总计 {len(test_info_for_featurizer['static_cols_idx'])} 个)")
  except Exception as e:
    print(f"加载或适配测试数据时出错: {e}")
    import traceback
    traceback.print_exc()
    return

  print(f"\n4. 使用 {AGGREGATION_FUNCTIONS} 将3D特征转换为2D特征...")

  print("   转换训练数据 (X_train_3d)...")
  try:
    x_train_2d = featurize_3d_to_2d(x_train_3d, train_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
    print(f"   X_train_2d 形状: {x_train_2d.shape}, dtype: {x_train_2d.dtype}")
  except Exception as e:
    print(f"转换训练数据 (X_train_3d) 时出错: {e}")
    import traceback
    traceback.print_exc()
    return

  print("   转换验证数据 (X_val_3d)...")
  try:
    x_val_2d = featurize_3d_to_2d(x_val_3d, train_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
    print(f"   X_val_2d 形状: {x_val_2d.shape}, dtype: {x_val_2d.dtype}")
  except Exception as e:
    print(f"转换验证数据 (X_val_3d) 时出错: {e}")
    import traceback
    traceback.print_exc()
    return

  print("   转换测试数据 (X_test_3d)...")
  try:
    x_test_2d = featurize_3d_to_2d(x_test_3d, test_info_for_featurizer, agg_funcs=AGGREGATION_FUNCTIONS)
    print(f"   X_test_2d 形状: {x_test_2d.shape}, dtype: {x_test_2d.dtype}")
  except Exception as e:
    print(f"转换测试数据 (X_test_3d) 时出错: {e}")
    import traceback
    traceback.print_exc()
    return

  print("\n--- 初步转换验证完成 ---")
  print("预期行为：")
  print(" - X_train_2d, X_val_2d, X_test_2d 的样本数应分别与 X_train_3d, X_val_3d, X_test_3d 一致。")
  print(" - X_..._2d 的特征维度数应为 (时序特征数 *聚合函数数量) + 静态特征数。")
  if 'temporal_cols_idx' in train_info_for_featurizer and 'static_cols_idx' in train_info_for_featurizer:
      num_temporal_features = len(train_info_for_featurizer['temporal_cols_idx'])
      num_static_features = len(train_info_for_featurizer['static_cols_idx'])
      num_agg_funcs = len(AGGREGATION_FUNCTIONS)
      expected_cols = num_temporal_features * num_agg_funcs + num_static_features
      print(f"   例如，对于训练数据，期望的列数 = "
            f"({num_temporal_features} 时序特征 * {num_agg_funcs} 聚合函数) + "
            f"{num_static_features} 静态特征 = {expected_cols}。")
      if x_train_2d.shape[1] != expected_cols:
          print(f"   警告: X_train_2d 的实际列数 ({x_train_2d.shape[1]}) 与期望列数 ({expected_cols}) 不符!")


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
