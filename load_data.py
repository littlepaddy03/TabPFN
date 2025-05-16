# load_data.py
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_npz_data(file_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """
    从指定的 .npz 文件加载特征、目标和元信息。

    参数:
        file_path (str): .npz 文件的路径。

    返回:
        Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]: 
        一个元组，包含特征 (features), 目标 (targets), 和元信息 (info)。
        如果加载失败则返回 None。
    """
    try:
        logger.info(f"开始加载数据从: {file_path}")
        data = np.load(file_path, allow_pickle=True)
        
        features = data['features']
        targets = data['targets']
        # info 通常是作为 object array 存储的字典，需要 .item() 来提取
        info = data['info'].item() if 'info' in data and hasattr(data['info'], 'item') else data.get('info', {})

        logger.info(f"成功从 {file_path} 加载数据。")
        logger.info(f"  特征形状: {features.shape}")
        logger.info(f"  目标形状: {targets.shape}")
        logger.info(f"  元信息键: {list(info.keys()) if isinstance(info, dict) else 'N/A'}")
        
        return features, targets, info
    except FileNotFoundError:
        logger.error(f"错误: 文件未找到 {file_path}")
        return None
    except KeyError as e:
        logger.error(f"错误: .npz 文件中缺少键 {e} (在 {file_path} 中)")
        return None
    except Exception as e:
        logger.error(f"加载 {file_path} 时发生未知错误: {e}")
        return None

def validate_data(
    features: np.ndarray, 
    targets: np.ndarray, 
    info: Dict[str, Any],
    expected_feature_shape_dims: int = 3,
    expected_target_shape_dims: int = 1
) -> bool:
    """
    校验加载的数据是否符合基本预期。

    参数:
        features (np.ndarray): 特征数组。
        targets (np.ndarray): 目标数组。
        info (Dict[str, Any]): 元信息字典。
        expected_feature_shape_dims (int): 特征数组预期的维度数量。
        expected_target_shape_dims (int): 目标数组预期的维度数量。

    返回:
        bool: 如果数据通过校验则为 True，否则为 False。
    """
    logger.info("开始数据校验...")
    valid = True

    # 1. 校验特征形状
    if features.ndim != expected_feature_shape_dims:
        logger.error(f"特征形状维度错误: 预期 {expected_feature_shape_dims}, 得到 {features.ndim}")
        valid = False
    else:
        logger.info(f"特征形状维度正确: {features.ndim} (样本数, 特征行数, 特征列数)")

    # 2. 校验目标形状
    if targets.ndim != expected_target_shape_dims:
        logger.error(f"目标形状维度错误: 预期 {expected_target_shape_dims}, 得到 {targets.ndim}")
        valid = False
    else:
        logger.info(f"目标形状维度正确: {targets.ndim}")

    # 3. 校验样本数量是否一致
    if features.shape[0] != targets.shape[0]:
        logger.error(
            f"特征和目标的样本数量不匹配: "
            f"特征样本数 {features.shape[0]}, 目标样本数 {targets.shape[0]}"
        )
        valid = False
    else:
        logger.info(f"特征和目标的样本数量匹配: {features.shape[0]}")

    # 4. 校验元信息
    required_info_keys = ['max_len', 'temporal_cols_used', 'static_cols_used', 'static_feature_row_index']
    if not isinstance(info, dict):
        logger.error(f"元信息 (info) 不是一个字典。得到类型: {type(info)}")
        valid = False
    else:
        for key in required_info_keys:
            if key not in info:
                logger.warning(f"元信息中缺少推荐键: {key}")
                # 根据严格程度，这里也可以设置为 valid = False
            else:
                logger.info(f"元信息键 '{key}' 存在。值为: {info[key]}")
        
        # 进一步校验 static_feature_row_index 是否合理
        if 'static_feature_row_index' in info and 'num_temporal_features' in info:
            if info['static_feature_row_index'] != info['num_temporal_features']:
                logger.warning(
                    f"元信息中 'static_feature_row_index' ({info['static_feature_row_index']}) "
                    f"与 'num_temporal_features' ({info['num_temporal_features']}) 不一致。"
                )
        if 'static_feature_row_index' in info and features.ndim == 3:
             if not (0 <= info['static_feature_row_index'] <= features.shape[1]):
                  logger.error(
                      f"元信息中的 'static_feature_row_index' ({info['static_feature_row_index']}) "
                      f"超出了特征的第二维度范围 (0-{features.shape[1]})。"
                  )
                  valid = False


    # 5. 检查目标值中的 NaN
    if np.isnan(targets).any():
        nan_count = np.isnan(targets).sum()
        logger.warning(f"目标值中发现 {nan_count} 个 NaN 值。可能需要在后续步骤中处理。")
        # 根据策略，如果存在NaN则校验失败
        # valid = False 
    else:
        logger.info("目标值中未发现 NaN。")

    if valid:
        logger.info("数据校验通过。")
    else:
        logger.error("数据校验失败。请检查上述错误/警告。")
    return valid

def split_data(
    features: np.ndarray, 
    targets: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将原始训练数据划分为新的训练集和验证集。

    参数:
        features (np.ndarray): 原始训练特征。
        targets (np.ndarray): 原始训练目标。
        test_size (float): 验证集所占的比例。
        random_state (int): 随机种子，用于可复现的划分。

    返回:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_val, y_train, y_val
    """
    logger.info(f"开始划分数据为训练集和验证集。验证集比例: {test_size}, 随机种子: {random_state}")
    X_train, X_val, y_train, y_val = train_test_split(
        features, targets, test_size=test_size, random_state=random_state, shuffle=True
    )
    logger.info("数据划分完成。")
    logger.info(f"  新训练集特征形状: {X_train.shape}, 新训练集目标形状: {y_train.shape}")
    logger.info(f"  验证集特征形状: {X_val.shape}, 验证集目标形状: {y_val.shape}")
    return X_train, X_val, y_train, y_val

def print_data_sample(features: np.ndarray, targets: np.ndarray, info: Dict[str, Any], n_samples_to_show: int = 1):
    """打印加载数据的样本信息，模仿方案中的示例。"""
    logger.info("\n打印数据样本信息:")
    logger.info("------------------------------")
    logger.info(f"特征数组形状 (Shape): {features.shape}")
    logger.info(f"特征数据类型 (dtype): {features.dtype}")
    if features.ndim == 3 and features.shape[0] >= n_samples_to_show:
        for i in range(n_samples_to_show):
            logger.info(f"特征样本 #{i+1} (前5行, 前5列):\n{features[i, :5, :5]}")
    elif features.ndim == 2 and features.shape[0] >= n_samples_to_show:
        for i in range(n_samples_to_show):
            logger.info(f"特征样本 #{i+1} (前5个元素):\n{features[i, :5]}")
    
    logger.info("------------------------------")
    logger.info(f"目标数组形状 (Shape): {targets.shape}")
    logger.info(f"目标数据类型 (dtype): {targets.dtype}")
    if targets.shape[0] >= n_samples_to_show:
         for i in range(n_samples_to_show):
            logger.info(f"目标样本 #{i+1} (前5个元素): {targets[i:i+5] if targets.ndim == 1 else targets[i, :5]}")
            
    logger.info("------------------------------")
    logger.info(f"元信息 (info):")
    if isinstance(info, dict):
        for key, value in info.items():
            if isinstance(value, list) and len(value) > 5: # 避免打印过长的列表
                 logger.info(f"  {key}: {value[:5]}... (共 {len(value)} 项)")
            else:
                 logger.info(f"  {key}: {value}")
    else:
        logger.info(f"  {info}")
    logger.info("------------------------------\n")

