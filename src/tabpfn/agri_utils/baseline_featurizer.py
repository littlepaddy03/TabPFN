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

"""Converts 3D agricultural data to a 2D feature matrix for baseline models.

This module provides a function to transform 3D input arrays, typically
representing samples with temporal and static features, into a 2D array
suitable for standard machine learning models. It allows for flexible
aggregation of temporal features.
"""

import collections
import functools
import numpy as np
from typing import Dict, List, Union, Callable, Sequence


# 定义支持的聚合函数及其对应的 NumPy 实现 (NaN-safe)
_AGG_FUNC_MAP: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'mean': functools.partial(np.nanmean, axis=1),
    'std': functools.partial(np.nanstd, axis=1, ddof=0),
    'median': functools.partial(np.nanmedian, axis=1),
    'min': functools.partial(np.nanmin, axis=1),
    'max': functools.partial(np.nanmax, axis=1),
    'sum': functools.partial(np.nansum, axis=1),
    'count_nan': lambda x: np.sum(np.isnan(x), axis=1),
    'count_valid': lambda x: np.sum(~np.isnan(x), axis=1),
}


def featurize_3d_to_2d(
    x_3d: np.ndarray,
    info: Dict,
    agg_funcs: Union[Sequence[str], Sequence[Callable]] = ('mean',)
) -> np.ndarray:
  """Converts 3D agricultural data to a 2D feature matrix with extensibility.

  The 3D input `x_3d` is expected to have dimensions corresponding to
  (n_samples, n_feature_rows, n_feature_cols). The `info` dictionary
  provides metadata to distinguish between temporal and static features.

  Temporal features are aggregated across time steps using specified
  aggregation functions. Static features are extracted directly. The aggregated
  temporal features and static features are then concatenated.

  Args:
    x_3d: A 3D NumPy array with shape (n_samples, n_feature_rows,
      n_feature_cols). `n_feature_rows` includes time steps and potentially
      a dedicated row for static features.
    info: A dictionary containing metadata. Expected keys:
      'static_feature_row_index': An integer index specifying which row in the
        `n_feature_rows` dimension contains the static features.
      'temporal_cols_idx': A list of integer column indices (for the
        `n_feature_cols` dimension) that correspond to temporal features.
      'static_cols_idx': A list of integer column indices (for the
        `n_feature_cols` dimension) that correspond to static features.
    agg_funcs: A sequence of strings or callable functions for aggregating
      temporal features. If strings, they must be keys in `_AGG_FUNC_MAP`.
      Defaults to ('mean',). Example: ['mean', 'std', np.var].

  Returns:
    A 2D NumPy array. The columns are ordered as:
    [
      agg_func1(temporal_feat1), agg_func2(temporal_feat1), ...,
      agg_func1(temporal_feat2), agg_func2(temporal_feat2), ...,
      ...,
      static_feat1, static_feat2, ...
    ]

  Raises:
    ValueError: If `static_feature_row_index` is out of bounds, if
      `temporal_cols_idx` or `static_cols_idx` contain invalid indices,
      or if an unknown string aggregation function is provided.
    KeyError: If essential keys are missing from the `info` dictionary.
    TypeError: If `agg_funcs` contains elements that are neither valid strings
      nor callables.
  """
  try:
    static_feature_row_idx = info['static_feature_row_index']
    temporal_cols_idx = info['temporal_cols_idx']
    static_cols_idx = info['static_cols_idx']
  except KeyError as e:
    raise KeyError(
        f"Missing essential key in 'info' dictionary: {e}. Required keys are "
        "'static_feature_row_index', 'temporal_cols_idx', 'static_cols_idx'."
    ) from e

  n_samples, n_feature_rows, n_feature_cols = x_3d.shape

  if not (0 <= static_feature_row_idx < n_feature_rows):
    raise ValueError(
        f"'static_feature_row_index' ({static_feature_row_idx}) is out of "
        f"bounds for n_feature_rows ({n_feature_rows}).")

  if temporal_cols_idx and (
      np.max(temporal_cols_idx) >= n_feature_cols or np.min(temporal_cols_idx) < 0):
    raise ValueError(
        "Invalid column index in 'temporal_cols_idx'. Indices must be within "
        f"[0, {n_feature_cols - 1}]."
    )
  if static_cols_idx and (
      np.max(static_cols_idx) >= n_feature_cols or np.min(static_cols_idx) < 0):
    raise ValueError(
        "Invalid column index in 'static_cols_idx'. Indices must be within "
        f"[0, {n_feature_cols - 1}]."
    )

  # --- 提取静态特征 ---
  static_data_all_cols_for_row = x_3d[:, static_feature_row_idx, :]
  if static_cols_idx:
    x_static_selected = static_data_all_cols_for_row[:, static_cols_idx]
  else:
    x_static_selected = np.empty((n_samples, 0))

  if x_static_selected.ndim == 1 and n_samples > 0 :
      x_static_selected = x_static_selected.reshape(n_samples, -1)
  elif n_samples == 0 and x_static_selected.ndim == 1:
      x_static_selected = x_static_selected.reshape(0, len(static_cols_idx) if static_cols_idx else 0)


  # --- 提取并聚合时序特征 ---
  temporal_rows_mask = np.ones(n_feature_rows, dtype=bool)
  temporal_rows_mask[static_feature_row_idx] = False

  all_aggregated_temporal_features: List[np.ndarray] = []

  if np.any(temporal_rows_mask) and temporal_cols_idx:
    x_temporal_all_cols_all_steps = x_3d[:, temporal_rows_mask, :]
    x_temporal_selected_cols_all_steps = x_temporal_all_cols_all_steps[:, :,
                                                                       temporal_cols_idx]

    for i in range(x_temporal_selected_cols_all_steps.shape[2]):
      current_temporal_feature_data = x_temporal_selected_cols_all_steps[:, :, i]

      for agg_func_spec in agg_funcs:
        aggregated_values: np.ndarray # 声明变量以确保其在作用域内

        if isinstance(agg_func_spec, str):
          if agg_func_spec not in _AGG_FUNC_MAP:
            raise ValueError(
                f"Unknown aggregation function string: '{agg_func_spec}'. "
                f"Available: {list(_AGG_FUNC_MAP.keys())}")
          agg_callable = _AGG_FUNC_MAP[agg_func_spec]
          aggregated_values = agg_callable(current_temporal_feature_data) # 计算并赋值
        elif callable(agg_func_spec):
          try:
            aggregated_values = agg_func_spec(current_temporal_feature_data)
          except TypeError as te:
            try:
              wrapped_callable = functools.partial(agg_func_spec, axis=1)
              aggregated_values = wrapped_callable(current_temporal_feature_data)
            except TypeError:
              raise TypeError(
                  f"Callable aggregation function {getattr(agg_func_spec, '__name__', str(agg_func_spec))} "
                  f"could not be applied directly or with axis=1. Ensure it handles 2D input "
                  f"(n_samples, n_time_steps) and aggregates appropriately. Original error: {te}"
              ) from te
        else:
          raise TypeError(
              f"Aggregation function specifier must be a string or callable, "
              f"got {type(agg_func_spec)}.")

        if aggregated_values.ndim == 1:
          aggregated_values = aggregated_values.reshape(-1, 1)
        all_aggregated_temporal_features.append(aggregated_values)
  
  if all_aggregated_temporal_features:
    final_aggregated_temporal_features = np.concatenate(
        all_aggregated_temporal_features, axis=1)
  else:
    final_aggregated_temporal_features = np.empty((n_samples, 0))

  if final_aggregated_temporal_features.ndim == 1 and n_samples > 0:
    final_aggregated_temporal_features = final_aggregated_temporal_features.reshape(n_samples, -1)
  elif n_samples == 0 and final_aggregated_temporal_features.ndim == 1:
     final_aggregated_temporal_features = final_aggregated_temporal_features.reshape(0,0)

  x_2d = np.concatenate(
      (final_aggregated_temporal_features, x_static_selected), axis=1)

  return x_2d
