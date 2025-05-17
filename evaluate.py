# evaluate.py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    prefix: str = "val"  # Prefix for metric names, e.g., "val", "test"
) -> Dict[str, float]:
    """
    计算回归任务的评估指标。

    参数:
        y_true (np.ndarray): 真实目标值。
        y_pred (np.ndarray): 预测目标值。
        prefix (str): 指标名称的前缀 (例如 "val", "test")。

    返回:
        Dict[str, float]: 包含 RMSE, MAE, R2 指标的字典。
    """
    if y_true.shape != y_pred.shape:
        logger.error(
            f"目标值和预测值的形状不匹配: y_true shape {y_true.shape}, y_pred shape {y_pred.shape}"
        )
        # 返回NaN或空字典，或引发错误，取决于策略
        return {
            f"{prefix}_rmse": float('nan'),
            f"{prefix}_mae": float('nan'),
            f"{prefix}_r2": float('nan')
        }
    
    if len(y_true) == 0:
        logger.warning("y_true 为空，无法计算指标。")
        return {
            f"{prefix}_rmse": float('nan'),
            f"{prefix}_mae": float('nan'),
            f"{prefix}_r2": float('nan')
        }

    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            f"{prefix}_rmse": rmse,
            f"{prefix}_mae": mae,
            f"{prefix}_r2": r2
        }
        logger.info(f"计算得到的指标 ({prefix}): {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"计算指标时发生错误: {e}")
        return {
            f"{prefix}_rmse": float('nan'),
            f"{prefix}_mae": float('nan'),
            f"{prefix}_r2": float('nan')
        }

def log_metrics_to_mlflow(metrics: Dict[str, Any]):
    """
    将指标记录到 MLflow (如果 MLflow 已被导入并处于活动运行中)。

    参数:
        metrics (Dict[str, Any]): 要记录的指标字典。
    """
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metrics(metrics)
            logger.info(f"指标已记录到 MLflow: {metrics}")
        else:
            logger.warning("没有活动的 MLflow 运行。指标未记录到 MLflow。")
    except ImportError:
        logger.info("MLflow 未安装。指标未记录到 MLflow。")
    except Exception as e:
        logger.error(f"记录指标到 MLflow 时发生错误: {e}")

if __name__ == '__main__':
    # 简单测试
    y_true_sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_sample = np.array([1.1, 2.2, 2.8, 4.3, 4.8])
    
    print("--- 测试 calculate_regression_metrics ---")
    metrics_calc = calculate_regression_metrics(y_true_sample, y_pred_sample, prefix="test_sample")
    print(f"计算得到的指标: {metrics_calc}")

    print("\n--- 测试 log_metrics_to_mlflow (MLflow 可能未运行) ---")
    # 尝试记录，如果 MLflow 未配置或运行，会打印警告/信息
    # 要真正测试 MLflow，需要在 MLflow 运行上下文中调用
    # import mlflow
    # with mlflow.start_run():
    #     log_metrics_to_mlflow(metrics_calc)
    log_metrics_to_mlflow(metrics_calc) # 这会打印警告，因为没有活动运行

    y_empty = np.array([])
    metrics_empty = calculate_regression_metrics(y_empty, y_empty, prefix="empty_sample")
    print(f"空数组指标: {metrics_empty}")

    y_mismatch_true = np.array([1,2,3])
    y_mismatch_pred = np.array([1,2])
    metrics_mismatch = calculate_regression_metrics(y_mismatch_true, y_mismatch_pred, prefix="mismatch_sample")
    print(f"形状不匹配指标: {metrics_mismatch}")
