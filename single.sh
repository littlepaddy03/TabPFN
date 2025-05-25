python run_baselines_single.py \
    --train_data_dir /root/lanyun-tmp/datasets/us_single/train \
    --test_data_dir /root/lanyun-tmp/datasets/us_single/test \
    --random_seed 42 \
    # --enable_optuna_rfr \ # 如果需要为随机森林进行 HPO
    # --n_trials_rfr_optuna 30 \ # Optuna 试验次数
    # ... 其他 BaselineFeaturizer 和 RandomForest 参数 ...