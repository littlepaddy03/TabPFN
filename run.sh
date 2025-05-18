# 在终端中先执行这个 (如果之前没执行过)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
    --run_phase "d1b_tabpfn_benchmark" \
    --train_file_path "/root/lanyun-tmp/datasets/us_processed_cp/train_processed.npz" \
    --test_file_path "/root/lanyun-tmp/datasets/us_processed_cp/test_processed.npz" \
    --val_split_size 0.2 \
    --random_seed 42 \
    --standardize_agri_features True \
    --handle_missing_agri_values "mean" \
    --agri_temporal_hidden_dim 128 \
    --debug_max_train_samples 9000 \
    --agri_static_hidden_dims_str "64" \
    --agri_fusion_strategy "gated" \
    --agri_encoded_embedding_dim 128 \
    --agri_encoder_kwargs "{'temporal_encoder_kwargs': {'num_heads': 2}}" \
    --tabpfn_n_estimators 16 \
    --tabpfn_ignore_pretraining_limits True \
    # --tabpfn_n_quantiles 1000 \
    --mlflow_experiment_name "TabPFN_Agri_Yield_D1b_Benchmark" \
    --mlflow_run_name "D1b_OOM_Attempt1_$(date +%Y%m%d_%H%M%S)" \
    --require_core_classes