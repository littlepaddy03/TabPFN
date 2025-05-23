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
"""
为每种作物独立训练和评估基线回归模型，用于农业产量预测。

该脚本负责：
1. 解析命令行参数，指定包含按作物划分的数据的目录。
2. 发现可用的作物类型。
3. 对每种作物：
    a. 加载其专属的训练和测试数据。
    b. 使用 BaselineFeaturizer 将 3D 数据转换为 2D，并在当前作物数据上拟合。
    c. 在当前作物数据上拟合特征和目标标准化器。
    d. 训练和评估一系列基线模型（均值预测器、线性回归、随机森林）。
    e. 使用 MLflow 记录实验参数、指标和针对该作物训练的模型。
    f. 对特定作物进行预测值诊断。
"""

import argparse
import logging
import os
import re # 用于从文件名提取作物名称
import sys
from typing import Any, Dict, List, Tuple, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

# Optuna (如果使用)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger_optuna = logging.getLogger(__name__) # 在尝试导入后定义
    logger_optuna.info("Optuna 未安装。如果请求，随机森林的超参数优化将被跳过。")


# 将 src 目录添加到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from src.tabpfn.agri_utils.data_loader import (
    load_npz_data,
    validate_data,
    adapt_info_keys,
)
from src.tabpfn.agri_utils.baseline_featurizer import BaselineFeaturizer
from evaluate import calculate_regression_metrics, log_metrics_to_mlflow

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__) # 主日志记录器

# 定义可以进行预测值诊断的作物名称列表 (这些名称应与从文件名中解析出的一致)
CROP_NAMES_FOR_DIAGNOSTICS: List[str] = ["corn", "soybeans", "wheat_spring_excl_drum"]
DEFAULT_TRAIN_FILE_PREFIX = "train_processed_"
DEFAULT_TEST_FILE_PREFIX = "test_processed_"


class MeanPredictor(DummyRegressor):
    """一个简单的预测器，总是预测训练集目标的均值。"""
    def __init__(self):
        super().__init__(strategy="mean")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run per-crop baseline regression models."
    )
    parser.add_argument(
        "--train_crop_data_dir",
        type=str,
        required=True,
        help="Directory containing per-crop training NPZ files (e.g., train_processed_corn.npz).",
    )
    parser.add_argument(
        "--test_crop_data_dir",
        type=str,
        required=True,
        help="Directory containing per-crop test NPZ files (e.g., test_processed_corn.npz).",
    )
    parser.add_argument(
        "--train_file_prefix",
        type=str,
        default=DEFAULT_TRAIN_FILE_PREFIX,
        help=f"Prefix for training NPZ files (default: '{DEFAULT_TRAIN_FILE_PREFIX}'). Crop name is expected after this prefix.",
    )
    parser.add_argument(
        "--test_file_prefix",
        type=str,
        default=DEFAULT_TEST_FILE_PREFIX,
        help=f"Prefix for test NPZ files (default: '{DEFAULT_TEST_FILE_PREFIX}'). Crop name is expected after this prefix.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="PerCrop_Baseline_Models_Experiment",
        help="Name of the MLflow experiment.",
    )
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="PerCropBaselineRun",
        help="Prefix for the MLflow main run name.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["MeanPredictor", "LinearRegression", "RandomForestRegressor_WithCropFeature"],
        help="List of baseline models to run for each crop.",
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=-1, help="Number of jobs for scikit-learn models (-1 uses all processors)."
    )

    # BaselineFeaturizer 参数
    parser.add_argument(
        "--temporal_agg_methods",
        nargs="+",
        default=["mean", "std"],
        help="List of aggregation methods for temporal features.",
    )
    parser.add_argument(
        "--static_feature_usage_strategy",
        type=str,
        default="first_row",
        choices=["first_row"], # 对于单作物，通常静态特征是一致的
        help="Strategy to use static features from the 3D input.",
    )
    parser.add_argument(
        "--scaler_type",
        type=str,
        default="StandardScaler",
        choices=["StandardScaler", "MinMaxScaler", "None"],
        help="Type of scaler to use for 2D features. 'None' for no scaling.",
    )
    parser.add_argument(
        "--static_row_index_for_featurizer",
        type=int,
        default=-1,
        help="Row index in the 3D sample matrix that contains static features for BaselineFeaturizer.",
    )

    # RandomForest 参数
    parser.add_argument(
        "--rf_n_estimators", type=int, default=100, help="Number of trees for RandomForest."
    )
    parser.add_argument(
        "--rf_max_depth", type=int, default=None, help="Max depth for RandomForest."
    )
    parser.add_argument(
        "--rf_min_samples_split", type=int, default=2, help="Min samples split for RandomForest."
    )
    parser.add_argument(
        "--rf_min_samples_leaf", type=int, default=1, help="Min samples leaf for RandomForest."
    )
    # Optuna HPO for RandomForest
    parser.add_argument(
        "--enable_optuna_rfr",
        action="store_true",
        help="Enable Optuna hyperparameter optimization for RandomForestRegressor (per crop).",
    )
    parser.add_argument(
        "--n_trials_rfr_optuna",
        type=int,
        default=20,
        help="Number of Optuna trials for RandomForestRegressor HPO (per crop).",
    )

    args = parser.parse_args()
    if args.enable_optuna_rfr and not OPTUNA_AVAILABLE:
        logger.warning(
            "Optuna HPO for RFR requested (--enable_optuna_rfr) but Optuna is not available. "
            "Skipping HPO."
        )
        args.enable_optuna_rfr = False
    return args


def discover_crop_datasets(
    data_dir: str, file_prefix: str
) -> Dict[str, str]:
    """
    从给定目录中发现按作物划分的数据集文件。

    Args:
        data_dir: 包含数据文件的目录。
        file_prefix: 数据文件的前缀 (例如 "train_processed_")。

    Returns:
        一个字典，键是作物名称，值是数据文件的完整路径。
    """
    crop_datasets: Dict[str, str] = {}
    if not os.path.isdir(data_dir):
        logger.warning(f"Data directory not found: {data_dir}")
        return crop_datasets

    pattern = re.compile(rf"^{re.escape(file_prefix)}(.+)\.npz$")
    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            crop_name = match.group(1)
            crop_datasets[crop_name] = os.path.join(data_dir, fname)
    logger.info(f"Discovered {len(crop_datasets)} crop datasets in '{data_dir}' with prefix '{file_prefix}': {list(crop_datasets.keys())}")
    return crop_datasets


def optimize_rfr_for_crop(
    trial: optuna.trial.Trial,
    X_train_crop_scaled: np.ndarray,
    y_train_crop_scaled: np.ndarray,
    X_val_crop_scaled: np.ndarray,
    y_val_crop_scaled: np.ndarray, # 验证集目标，已标准化
    random_state: int,
    n_jobs: int,
) -> float:
    """Optuna 目标函数，用于优化特定作物的 scikit-learn RandomForestRegressor。"""
    n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
    max_depth_choice = trial.suggest_categorical("rf_max_depth_choice", [None, 10, 20, 30, 50])
    max_depth = None if max_depth_choice == "None" else int(max_depth_choice)

    min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 32)
    min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 32)
    max_features = trial.suggest_categorical("rf_max_features", ["sqrt", "log2", 1.0]) # 1.0 for all features

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features, # type: ignore
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train_crop_scaled, y_train_crop_scaled)
    y_pred_val_scaled = model.predict(X_val_crop_scaled)
    # Optuna 默认最小化目标，返回 RMSE
    rmse = np.sqrt(np.mean((y_val_crop_scaled - y_pred_val_scaled) ** 2))
    return rmse


def main():
    """主执行函数。"""
    args = parse_args()

    logger.info(f"--- Running Per-Crop Baseline Models ---")

    # 1. 发现按作物划分的训练和测试数据集
    train_crop_files = discover_crop_datasets(args.train_crop_data_dir, args.train_file_prefix)
    test_crop_files = discover_crop_datasets(args.test_crop_data_dir, args.test_file_prefix)

    if not train_crop_files:
        logger.error(f"No training data files found in {args.train_crop_data_dir} with prefix {args.train_file_prefix}. Exiting.")
        sys.exit(1)

    # 设置 MLflow 主实验
    mlflow.set_experiment(args.exp_name)
    main_run_name = f"{args.run_name_prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=main_run_name) as main_run: # 主运行
        mlflow.log_params(vars(args))
        logger.info(f"MLflow Main Run ID: {main_run.info.run_id} for experiment '{args.exp_name}'")

        overall_metrics_all_crops_all_models: Dict[str, Dict[str, Dict[str, float]]] = {}

        # 2. 遍历每个作物进行独立训练和测试
        for crop_name, train_file_path in train_crop_files.items():
            logger.info(f"\n\n=== Processing Crop: {crop_name} ===")
            test_file_path = test_crop_files.get(crop_name)
            if not test_file_path:
                logger.warning(f"No test data file found for crop '{crop_name}' in {args.test_crop_data_dir} with prefix {args.test_file_prefix}. Skipping evaluation for this crop.")
                continue

            # 为当前作物创建一个 MLflow 父级运行
            with mlflow.start_run(run_name=f"Crop_{crop_name}", nested=True) as crop_run:
                mlflow.log_param("crop_name", crop_name)
                logger.info(f"  MLflow Crop Run ID for '{crop_name}': {crop_run.info.run_id}")

                # 2a. 加载当前作物的数据
                logger.info(f"  Loading training data for '{crop_name}' from: {train_file_path}")
                X_train_crop_3d_full, y_train_crop_full, info_train_crop_full = load_npz_data(
                    train_file_path, data_split_type=f"train_{crop_name}"
                )
                validate_data(X_train_crop_3d_full, y_train_crop_full, info_train_crop_full, f"Train_full_{crop_name}")
                info_train_crop_full = adapt_info_keys(info_train_crop_full, f"train_full_{crop_name}", logger)

                # 移除 y 中的 NaN
                nan_mask_train_crop = ~np.isnan(y_train_crop_full)
                if not np.all(nan_mask_train_crop):
                    logger.info(f"  Removing {np.sum(~nan_mask_train_crop)} NaN targets from '{crop_name}' training data.")
                    X_train_crop_3d_full = X_train_crop_3d_full[nan_mask_train_crop]
                    y_train_crop_full = y_train_crop_full[nan_mask_train_crop]

                if X_train_crop_3d_full.shape[0] < 2 : # 需要至少2个样本进行划分或训练
                    logger.warning(f"  Not enough training samples for crop '{crop_name}' after NaN removal ({X_train_crop_3d_full.shape[0]}). Skipping this crop.")
                    continue
                
                # 为 Optuna HPO 划分内部验证集 (如果启用)
                # 如果样本量非常小，可以考虑不划分，或使用交叉验证 (更复杂)
                # 这里简化处理：如果样本太少，不进行 HPO 的内部验证划分
                if args.enable_optuna_rfr and X_train_crop_3d_full.shape[0] > 10: # 示例阈值
                    X_train_crop_3d_fit, X_val_crop_3d_internal, y_train_crop_fit, y_val_crop_internal = train_test_split(
                        X_train_crop_3d_full, y_train_crop_full, test_size=0.2, random_state=args.random_state
                    )
                    info_train_crop_fit = info_train_crop_full.copy()
                    info_val_crop_internal = info_train_crop_full.copy()
                else: # 不进行 HPO 或样本太少，使用全部数据进行拟合
                    X_train_crop_3d_fit = X_train_crop_3d_full
                    y_train_crop_fit = y_train_crop_full
                    info_train_crop_fit = info_train_crop_full
                    X_val_crop_3d_internal, y_val_crop_internal, info_val_crop_internal = None, None, None


                logger.info(f"  Loading test data for '{crop_name}' from: {test_file_path}")
                X_test_crop_3d, y_test_crop, info_test_crop = load_npz_data(
                    test_file_path, data_split_type=f"test_{crop_name}"
                )
                validate_data(X_test_crop_3d, y_test_crop, info_test_crop, f"Test_{crop_name}")
                info_test_crop = adapt_info_keys(info_test_crop, f"test_{crop_name}", logger)
                
                nan_mask_test_crop = ~np.isnan(y_test_crop)
                if not np.all(nan_mask_test_crop):
                    logger.info(f"  Removing {np.sum(~nan_mask_test_crop)} NaN targets from '{crop_name}' test data.")
                    X_test_crop_3d = X_test_crop_3d[nan_mask_test_crop]
                    y_test_crop = y_test_crop[nan_mask_test_crop]

                if X_test_crop_3d.shape[0] == 0:
                    logger.warning(f"  Test data for crop '{crop_name}' is empty after NaN removal. Skipping evaluation for this crop.")
                    continue


                # 2b. 3D 特征转换为 2D (针对当前作物)
                logger.info(f"  Converting 3D features to 2D for '{crop_name}'...")
                featurizer_crop = BaselineFeaturizer(
                    temporal_agg_methods=args.temporal_agg_methods,
                    static_feature_usage_strategy=args.static_feature_usage_strategy,
                    scaler_type="None", # 先不缩放
                    static_row_index=args.static_row_index_for_featurizer,
                    add_dummy_crop_feature=False, # 数据已按作物划分
                )
                X_train_crop_2d_unscaled = featurizer_crop.fit_transform(X_train_crop_3d_fit, info_train_crop_fit)
                feature_names_2d_crop = featurizer_crop.get_feature_names_out()
                
                if X_val_crop_3d_internal is not None:
                    X_val_crop_2d_internal_unscaled = featurizer_crop.transform(X_val_crop_3d_internal, info_val_crop_internal)
                else:
                    X_val_crop_2d_internal_unscaled = None

                X_test_crop_2d_unscaled = featurizer_crop.transform(X_test_crop_3d, info_test_crop)
                logger.info(f"    x_train_crop_2d_unscaled shape: {X_train_crop_2d_unscaled.shape}")

                # 2c. 标准化特征和目标变量 (针对当前作物)
                logger.info(f"  Standardizing 2D features and target variables for '{crop_name}'...")
                feature_scaler_crop: Optional[StandardScaler] = None
                if args.scaler_type != "None":
                    if args.scaler_type == "StandardScaler":
                        feature_scaler_crop = StandardScaler()
                    elif args.scaler_type == "MinMaxScaler":
                        from sklearn.preprocessing import MinMaxScaler
                        feature_scaler_crop = MinMaxScaler() # type: ignore
                    
                    if feature_scaler_crop is not None:
                        X_train_crop_2d_scaled = feature_scaler_crop.fit_transform(X_train_crop_2d_unscaled)
                        if X_val_crop_2d_internal_unscaled is not None:
                             X_val_crop_2d_internal_scaled = feature_scaler_crop.transform(X_val_crop_2d_internal_unscaled)
                        else:
                            X_val_crop_2d_internal_scaled = None
                        X_test_crop_2d_scaled = feature_scaler_crop.transform(X_test_crop_2d_unscaled)
                else:
                    X_train_crop_2d_scaled = X_train_crop_2d_unscaled
                    X_val_crop_2d_internal_scaled = X_val_crop_2d_internal_unscaled
                    X_test_crop_2d_scaled = X_test_crop_2d_unscaled

                target_scaler_crop = StandardScaler()
                y_train_crop_scaled = target_scaler_crop.fit_transform(y_train_crop_fit.reshape(-1, 1)).ravel()
                if y_val_crop_internal is not None:
                    y_val_crop_internal_scaled = target_scaler_crop.transform(y_val_crop_internal.reshape(-1,1)).ravel()
                else:
                    y_val_crop_internal_scaled = None


                crop_model_metrics_summary: Dict[str, Dict[str, float]] = {}
                # 2d. 训练和评估模型 (针对当前作物)
                for model_name in args.model_names:
                    with mlflow.start_run(run_name=f"Model_{model_name}_For_{crop_name}", nested=True) as model_run:
                        mlflow.log_param("model_name_for_crop", model_name)
                        mlflow.log_param("crop_name_for_model", crop_name)
                        logger.info(f"\n    --- Training Model: {model_name} for Crop: {crop_name} ---")
                        logger.info(f"      MLflow Model Run ID for {model_name} on '{crop_name}': {model_run.info.run_id}")

                        model_crop: Any = None
                        if model_name == "MeanPredictor":
                            model_crop = MeanPredictor()
                        elif model_name == "LinearRegression":
                            model_crop = LinearRegression(n_jobs=args.n_jobs)
                        elif model_name == "RandomForestRegressor_WithCropFeature":
                            if args.enable_optuna_rfr and OPTUNA_AVAILABLE and \
                               X_val_crop_2d_internal_scaled is not None and y_val_crop_internal_scaled is not None:
                                logger.info(f"      Optimizing RandomForest for '{crop_name}' with Optuna...")
                                study_crop = optuna.create_study(direction="minimize")
                                study_crop.optimize(
                                    lambda trial: optimize_rfr_for_crop(
                                        trial,
                                        X_train_crop_2d_scaled,
                                        y_train_crop_scaled,
                                        X_val_crop_2d_internal_scaled, # type: ignore
                                        y_val_crop_internal_scaled,    # type: ignore
                                        args.random_state,
                                        args.n_jobs,
                                    ),
                                    n_trials=args.n_trials_rfr_optuna,
                                )
                                best_params_rfr_crop = study_crop.best_params
                                logger.info(f"      Best Optuna HPO params for RFR on '{crop_name}': {best_params_rfr_crop}")
                                mlflow.log_params({f"optuna_best_{k}": v for k,v in best_params_rfr_crop.items()})
                                model_crop = RandomForestRegressor(
                                    n_estimators=best_params_rfr_crop.get("rf_n_estimators", args.rf_n_estimators),
                                    max_depth=best_params_rfr_crop.get("rf_max_depth_choice", args.rf_max_depth), # Adapt key
                                    min_samples_split=best_params_rfr_crop.get("rf_min_samples_split", args.rf_min_samples_split),
                                    min_samples_leaf=best_params_rfr_crop.get("rf_min_samples_leaf", args.rf_min_samples_leaf),
                                    max_features=best_params_rfr_crop.get("rf_max_features", "sqrt"), # Adapt key
                                    random_state=args.random_state,
                                    n_jobs=args.n_jobs,
                                )
                            else:
                                model_crop = RandomForestRegressor(
                                    n_estimators=args.rf_n_estimators,
                                    max_depth=args.rf_max_depth,
                                    min_samples_split=args.rf_min_samples_split,
                                    min_samples_leaf=args.rf_min_samples_leaf,
                                    random_state=args.random_state,
                                    n_jobs=args.n_jobs,
                                )
                        else:
                            logger.warning(f"      Model {model_name} not recognized. Skipping for crop {crop_name}.")
                            continue

                        logger.info(f"      Fitting {model_name} for '{crop_name}'...")
                        model_crop.fit(X_train_crop_2d_scaled, y_train_crop_scaled)
                        logger.info(f"      {model_name} for '{crop_name}' trained successfully.")
                        try:
                            mlflow.sklearn.log_model(model_crop, f"{model_name}_{crop_name}_model")
                        except Exception as e:
                             logger.warning(f"MLflow: Model logged without a signature for {model_name} on {crop_name}: {e}")


                        logger.info(f"      Evaluating {model_name} on test set for '{crop_name}'...")
                        y_pred_crop_scaled = model_crop.predict(X_test_crop_2d_scaled)
                        y_pred_crop_original = target_scaler_crop.inverse_transform(
                            y_pred_crop_scaled.reshape(-1, 1)
                        ).ravel()

                        metrics_crop = calculate_regression_metrics(y_test_crop, y_pred_crop_original)
                        logger.info(f"      Metrics for {model_name} on '{crop_name}' test set: {metrics_crop}")
                        log_metrics_to_mlflow(metrics_crop, prefix=f"test_{crop_name}_") # Log with crop prefix
                        crop_model_metrics_summary[model_name] = metrics_crop

                        # Prediction diagnostics for specific crops
                        if crop_name in CROP_NAMES_FOR_DIAGNOSTICS:
                            logger.info(f"      --- '{crop_name}' Prediction Diagnostics for {model_name} ---")
                            logger.info(f"         y_original (test) shape: {y_test_crop.shape}, first 5: {y_test_crop[:5]}")
                            logger.info(f"         y_pred_scaled (model direct output) shape: {y_pred_crop_scaled.shape}, first 5: {y_pred_crop_scaled[:5]}")
                            logger.info(f"         y_pred_original (inverse-standardized) shape: {y_pred_crop_original.shape}, first 5: {y_pred_crop_original[:5]}")
                            logger.info(f"         Crop-specific training y mean (original scale): {target_scaler_crop.mean_[0]:.4f}")
                            logger.info(f"         '{crop_name}' test y mean (original scale): {np.mean(y_test_crop):.4f}")
                            if target_scaler_crop.scale_[0] > 1e-6:
                                y_test_mean_crop_scaled = (np.mean(y_test_crop) - target_scaler_crop.mean_[0]) / target_scaler_crop.scale_[0]
                                logger.info(f"         '{crop_name}' test y mean (to crop-specific standardized scale): {y_test_mean_crop_scaled:.4f}")
                                y_pred_crop_scaled_mean = np.mean(y_pred_crop_scaled)
                                logger.info(f"         '{crop_name}' y_pred_scaled mean: {y_pred_crop_scaled_mean:.4f}")
                            else:
                                logger.info(f"         '{crop_name}' test y mean (to crop-specific standardized scale): Cannot compute, crop target_scaler.scale_ is too small.")
                            logger.info(f"      --- Prediction Diagnostics End for '{crop_name}', Model {model_name} ---")
                        
                        # Feature importance for RandomForest on this crop
                        if model_name == "RandomForestRegressor_WithCropFeature" and hasattr(model_crop, "feature_importances_"):
                            logger.info(f"    --- {model_name} Feature Importance for Crop: {crop_name} ---")
                            importances_crop = model_crop.feature_importances_
                            feature_importance_df_crop = pd.DataFrame(
                                {"feature": feature_names_2d_crop, "importance": importances_crop}
                            ).sort_values("importance", ascending=False)
                            logger.info(f"      Top 15 features for '{crop_name}':\n{feature_importance_df_crop.head(15)}")
                            mlflow.log_dict(feature_importance_df_crop.to_dict(), f"feature_importances_{crop_name}.json")
                            logger.info(f"    --- Feature Importance Analysis End for '{crop_name}' ---")

                overall_metrics_all_crops_all_models[crop_name] = crop_model_metrics_summary
                logger.info(f"=== Finished processing Crop: {crop_name} ===")


        logger.info("\n\n--- Per-Crop Baseline Models Overall Evaluation Summary ---")
        for crop_name_summary, models_data in overall_metrics_all_crops_all_models.items():
            logger.info(f"Crop: {crop_name_summary}")
            for model_name_summary, metrics_values in models_data.items():
                logger.info(f"  Model: {model_name_summary}, Metrics: {metrics_values}")
            logger.info("----------------------------------------")
        logger.info(f"--- All per-crop baseline model runs finished. Main Run ID: {main_run.info.run_id} ---")

if __name__ == "__main__":
    main()
