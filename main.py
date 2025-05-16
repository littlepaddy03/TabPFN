# main.py
import argparse
import logging
import os
from load_data import load_npz_data, validate_data, split_data, print_data_sample # 确保 load_data.py 在同一目录或 PYTHONPATH 中

# 配置日志
logger = logging.getLogger(__name__) # 使用在 load_data.py 中已配置的 root logger
logger.setLevel(logging.INFO) # 确保 main 脚本的日志级别

def main(args):
    logger.info("开始执行数据加载与准备流程...")

    # --- 1. 加载原始训练数据 ---
    logger.info("--- 正在加载原始训练数据 ---")
    X_orig_train, y_orig_train, info_train = None, None, None
    if args.train_file_path:
        if not os.path.exists(args.train_file_path):
            logger.error(f"训练文件路径不存在: {args.train_file_path}")
            return
        load_result_train = load_npz_data(args.train_file_path)
        if load_result_train:
            X_orig_train, y_orig_train, info_train = load_result_train
            # 打印训练数据样本
            print_data_sample(X_orig_train, y_orig_train, info_train, n_samples_to_show=1)
        else:
            logger.error("加载原始训练数据失败。")
            return # 关键数据加载失败，提前退出
    else:
        logger.warning("未提供训练文件路径，跳过训练数据加载。")

    # --- 2. 加载测试数据 ---
    logger.info("--- 正在加载测试数据 ---")
    X_test, y_test, info_test = None, None, None
    if args.test_file_path:
        if not os.path.exists(args.test_file_path):
            logger.error(f"测试文件路径不存在: {args.test_file_path}")
            # 根据需求，这里可以选择是否 return
        else:
            load_result_test = load_npz_data(args.test_file_path)
            if load_result_test:
                X_test, y_test, info_test = load_result_test
                # 打印测试数据样本
                print_data_sample(X_test, y_test, info_test, n_samples_to_show=1)
            else:
                logger.error("加载测试数据失败。")
                # 根据需求，这里可以选择是否 return
    else:
        logger.warning("未提供测试文件路径，跳过测试数据加载。")

    # --- 3. 数据校验 ---
    if X_orig_train is not None and y_orig_train is not None and info_train is not None:
        logger.info("--- 正在校验原始训练数据 ---")
        if not validate_data(X_orig_train, y_orig_train, info_train):
            logger.error("原始训练数据校验失败，请检查日志。后续处理可能受影响。")
            # return # 根据严格程度决定是否退出
    
    if X_test is not None and y_test is not None and info_test is not None:
        logger.info("--- 正在校验测试数据 ---")
        if not validate_data(X_test, y_test, info_test):
            logger.error("测试数据校验失败，请检查日志。后续处理可能受影响。")
            # return

    # --- 4. 训练集/验证集划分 ---
    # 根据更新后的方案，此步骤现在是强制执行的（如果原始训练数据已加载）
    X_train, X_val, y_train, y_val = None, None, None, None
    if X_orig_train is not None and y_orig_train is not None:
        logger.info("--- 正在划分训练数据为新的训练集和验证集 ---")
        X_train, X_val, y_train, y_val = split_data(
            X_orig_train, 
            y_orig_train, 
            test_size=args.val_split_size, 
            random_state=args.random_seed
        )
        logger.info(f"数据划分后: X_train shape: {X_train.shape if X_train is not None else 'N/A'}, y_train shape: {y_train.shape if y_train is not None else 'N/A'}")
        logger.info(f"             X_val shape: {X_val.shape if X_val is not None else 'N/A'},   y_val shape: {y_val.shape if y_val is not None else 'N/A'}")
        # 后续模型训练应使用 X_train, y_train (用于训练) 和 X_val, y_val (用于验证)
    else:
        logger.warning("原始训练数据 (X_orig_train, y_orig_train) 未加载或加载失败，无法进行训练集/验证集划分。")
        # 在这种情况下，X_train, y_train, X_val, y_val 将保持为 None
        # 后续依赖这些数据的步骤需要进行检查

    logger.info("数据加载与准备流程结束。")

    # 后续步骤（如基线模型、AgriTabPFN训练）将在这里继续...
    # 确保后续步骤能正确处理 X_train, y_train, X_val, y_val 可能为 None 的情况
    if X_train is not None and y_train is not None:
        logger.info("准备将加载和划分的数据用于后续模型训练和评估。")
        # 示例:
        # train_baseline_models(X_train, y_train, X_val, y_val, X_test, y_test, info_train, args)
        # train_agri_tabpfn(X_train, y_train, X_val, y_val, X_test, y_test, info_train, args)
    else:
        logger.error("由于训练数据未准备好，无法继续进行模型训练步骤。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN农业产量预测：数据加载与准备脚本")
    
    parser.add_argument(
        "--train_file_path", 
        type=str, 
        help="预处理后的训练数据 .npz 文件路径 (e.g., train_processed.npz)"
    )
    parser.add_argument(
        "--test_file_path", 
        type=str, 
        help="预处理后的测试数据 .npz 文件路径 (e.g., test_processed.npz)"
    )
    # 移除了 --perform_split 参数，因为划分现在是默认行为
    parser.add_argument(
        "--val_split_size", 
        type=float, 
        default=0.2, 
        help="指定验证集所占的比例 (用于从原始训练数据中划分)。"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42, 
        help="用于数据划分和所有其他随机过程的随机种子。"
    )

    args = parser.parse_args()
    main(args)
