python run_baselines_single.py \
    --train_crop_data_dir /path/to/your/per_crop_train_data/ \
    --test_crop_data_dir /path/to/your/per_crop_test_data/ \
    --exp_name "PerCrop_Baselines_Detailed" \
    --run_name_prefix "SingleCropRun" \
    --model_names MeanPredictor LinearRegression RandomForestRegressor_WithCropFeature \
    --random_state 42 \
    # --enable_optuna_rfr \ # 如果需要为随机森林进行 HPO
    # --n_trials_rfr_optuna 30 \ # Optuna 试验次数
    # ... 其他 BaselineFeaturizer 和 RandomForest 参数 ...