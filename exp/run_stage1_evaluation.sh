python run_stage1_evaluation.py \
    --train_file_path /root/lanyun-tmp/datasets/us_processed/train_processed.npz \
    --test_file_path /root/lanyun-tmp/datasets/us_processed/test_processed.npz \
    --output_dir /root/lanyun-tmp/TabPFN/exp_output \
    --max_temporal_length 128 
    # --static_row_index -1  # Optional: if not in train_info.npz or to override
    # --max_temporal_length 100 # Optional: if not in train_info.npz or to override
    # --device cpu # Optional: to force CPU