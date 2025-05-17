# main.py
import argparse
import logging
import os
import time 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import mlflow 
import sys
from typing import List, Optional, Any, Dict 

# --- PYTHONPATH/sys.path Management ---
# (Ensure tabpfn package is findable - adjust paths as per your project structure)
# Example:
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) 
# SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
# if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
#     sys.path.insert(0, SRC_DIR)
#     logger.info(f"Added '{SRC_DIR}' to sys.path for tabpfn module resolution.")
# else: # Fallback
#     if PROJECT_ROOT not in sys.path and os.path.exists(os.path.join(PROJECT_ROOT, "tabpfn")):
#          sys.path.insert(0, PROJECT_ROOT)
#          logger.info(f"Added project root '{PROJECT_ROOT}' to sys.path (fallback).")


from load_data import load_npz_data, validate_data, split_data, print_data_sample
from evaluate import calculate_regression_metrics, log_metrics_to_mlflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

CORE_CLASSES_IMPORTED = False
try:
    # Corrected imports based on user-provided file structure
    from tabpfn.encoders.agri_encoders import AgriDataPreprocessor, AgriDataEncoder
    logger.info("Successfully imported AgriDataPreprocessor and AgriDataEncoder.")
    CORE_CLASSES_IMPORTED = True
except ImportError as e:
    logger.error(f"CRITICAL ERROR: Failed to import AgriDataPreprocessor or AgriDataEncoder.")
    logger.error(f"Please ensure 'tabpfn.encoders.agri_encoders' is accessible via PYTHONPATH.")
    logger.error(f"Detailed import error: {e}", exc_info=True)

def get_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = "cuda"
        logger.info(f"CUDA detected. Using GPU: {device_name}")
    else:
        device = "cpu"
        logger.info("CUDA not detected. Using CPU.")
    return device

def parse_int_list(string_list: Optional[str]) -> Optional[List[int]]:
    if string_list is None or string_list.lower() == 'none':
        return None
    try:
        return [int(item.strip()) for item in string_list.split(',')]
    except ValueError:
        logger.error(f"Cannot parse '{string_list}' into a list of integers. Use comma-separated integers.")
        return None

def run_d1_train_agri_encoder(
    X_train_orig: np.ndarray, y_train_orig: np.ndarray,
    X_val_orig: np.ndarray, y_val_orig: np.ndarray,
    info_train: dict,
    args: argparse.Namespace
):
    if not CORE_CLASSES_IMPORTED:
        logger.error("Core classes (AgriDataPreprocessor/AgriDataEncoder) not imported. Skipping D1 training.")
        if mlflow.active_run():
            mlflow.log_param("d1_encoder_training_status", "skipped_import_error")
        return

    logger.info("--- Starting D1: Standalone AgriDataEncoder Training ---")
    device = get_device()

    # 1. PREPROCESS DATA
    logger.info("Step 1: Preprocessing data...")
    # AgriDataPreprocessor.__init__ takes: static_row_index, max_temporal_length, standardize_features, handle_missing_values
    # It also has optional temporal_features_cols, static_features_cols which we are not using here.
    preprocessor = AgriDataPreprocessor(
        static_row_index=int(info_train['static_feature_row_index']),
        max_temporal_length=int(info_train['max_len']),
        standardize_features=args.standardize_agri_features,
        handle_missing_values=args.handle_missing_agri_values
    )
    X_train_processed_np, train_temporal_mask_np = preprocessor.fit_transform(X_train_orig)
    X_val_processed_np, val_temporal_mask_np = preprocessor.transform(X_val_orig)

    # Convert to PyTorch Tensors
    X_train_processed = torch.tensor(X_train_processed_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_orig, dtype=torch.float32).unsqueeze(1).to(device) # Ensure y is (batch, 1) for MSELoss
    
    # Handle None mask: create a dummy mask of all Trues if None
    if train_temporal_mask_np is not None:
        train_temporal_mask = torch.tensor(train_temporal_mask_np, dtype=torch.bool).to(device)
    else: # This case should ideally not happen if max_temporal_length_ > 0
        logger.warning("Training temporal mask is None. Creating a dummy mask of all Trues.")
        # Shape of mask should be (n_samples, max_temporal_length_)
        # preprocessor.max_temporal_length_ is the length of the temporal part in X_train_processed
        # X_train_processed has shape (n_samples, preprocessor.max_temporal_length_ + 1, n_features_cols)
        # The temporal part has preprocessor.max_temporal_length_ steps.
        mask_shape_train = (X_train_processed.shape[0], preprocessor.max_temporal_length_)
        train_temporal_mask = torch.ones(mask_shape_train, dtype=torch.bool).to(device)


    X_val_processed = torch.tensor(X_val_processed_np, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val_orig, dtype=torch.float32).unsqueeze(1).to(device)
    
    if val_temporal_mask_np is not None:
        val_temporal_mask = torch.tensor(val_temporal_mask_np, dtype=torch.bool).to(device)
    else:
        logger.warning("Validation temporal mask is None. Creating a dummy mask of all Trues.")
        mask_shape_val = (X_val_processed.shape[0], preprocessor.max_temporal_length_)
        val_temporal_mask = torch.ones(mask_shape_val, dtype=torch.bool).to(device)
    
    logger.info(f"  X_train_processed shape: {X_train_processed.shape}, y_train shape: {y_train.shape}, train_temporal_mask shape: {train_temporal_mask.shape if train_temporal_mask is not None else 'None'}")
    logger.info(f"  X_val_processed shape: {X_val_processed.shape}, y_val shape: {y_val.shape}, val_temporal_mask shape: {val_temporal_mask.shape if val_temporal_mask is not None else 'None'}")

    if mlflow.active_run():
        mlflow.log_param("prep_static_row_index", preprocessor.static_row_index)
        mlflow.log_param("prep_max_temporal_length_input", preprocessor.max_temporal_length) # User input
        mlflow.log_param("prep_max_temporal_length_fitted", preprocessor.max_temporal_length_) # Actual used
        mlflow.log_param("prep_standardize", preprocessor.standardize_features)
        mlflow.log_param("prep_missing_values", preprocessor.handle_missing_values)

    # 2. INITIALIZE AgriDataEncoder MODEL
    logger.info("Step 2: Initializing AgriDataEncoder...")
    n_features_cols = X_train_processed.shape[2] # This is the input_dim for sub-encoders
    parsed_static_hidden_dims = parse_int_list(args.encoder_static_hidden_dims_str)

    # Default kwargs for sub-encoders can be empty or specified if needed
    # For TimeFeaturesEncoder, 'num_heads' is a param. Let's use a default or make it CLI.
    default_temporal_encoder_kwargs = {'num_heads': args.encoder_temporal_num_heads}

    agri_encoder = AgriDataEncoder(
        temporal_input_dim=n_features_cols,
        static_input_dim=n_features_cols,
        output_dim=1, # Outputting a single value for regression
        temporal_hidden_dim=args.encoder_temporal_hidden_dim,
        static_hidden_dims=parsed_static_hidden_dims if parsed_static_hidden_dims is not None else [64], # Default from AgriDataEncoder
        fusion_strategy=args.encoder_fusion_strategy,
        temporal_encoder_kwargs=args.encoder_temporal_kwargs if args.encoder_temporal_kwargs else default_temporal_encoder_kwargs,
        static_encoder_kwargs=args.encoder_static_kwargs if args.encoder_static_kwargs else {}
    ).to(device)

    logger.info(f"AgriDataEncoder initialized. Model structure:\n{agri_encoder}")
    if mlflow.active_run():
        mlflow.log_params({
            "enc_temporal_input_dim_sub": n_features_cols, "enc_static_input_dim_sub": n_features_cols,
            "enc_output_dim_final": 1, "enc_temporal_hidden_dim": args.encoder_temporal_hidden_dim,
            "enc_static_hidden_dims": str(parsed_static_hidden_dims), 
            "enc_fusion_strategy": args.encoder_fusion_strategy,
            "enc_temporal_kwargs": str(args.encoder_temporal_kwargs if args.encoder_temporal_kwargs else default_temporal_encoder_kwargs),
            "enc_static_kwargs": str(args.encoder_static_kwargs if args.encoder_static_kwargs else {}),
        })

    # 3. DEFINE LOSS FUNCTION AND OPTIMIZER
    logger.info("Step 3: Defining loss function and optimizer...")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        agri_encoder.parameters(), lr=args.encoder_lr, weight_decay=args.encoder_weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.encoder_scheduler_step_size, gamma=args.encoder_scheduler_gamma
    )
    if mlflow.active_run():
        mlflow.log_params({
            "enc_train_optimizer": "AdamW", "enc_train_lr": args.encoder_lr,
            "enc_train_weight_decay": args.encoder_weight_decay, "enc_train_scheduler": "StepLR",
            "enc_train_scheduler_step": args.encoder_scheduler_step_size, 
            "enc_train_scheduler_gamma": args.encoder_scheduler_gamma,
            "enc_train_loss_func": "MSELoss"
        })

    # 4. TRAINING LOOP
    logger.info("Step 4: Starting AgriDataEncoder training loop...")
    train_dataset = TensorDataset(X_train_processed, y_train, train_temporal_mask)
    train_loader = DataLoader(train_dataset, batch_size=args.encoder_batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_processed, y_val, val_temporal_mask)
    val_loader = DataLoader(val_dataset, batch_size=args.encoder_batch_size, shuffle=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Determine the static_row_index for the AgriDataEncoder.forward method
    # This is the index *within the preprocessed X_final* that AgriDataPreprocessor outputs
    final_rows_in_preprocessed_data = preprocessor.max_temporal_length_ + 1 # max_temporal_length_ is fitted
    static_row_idx_for_encoder_fwd = preprocessor.static_row_index % final_rows_in_preprocessed_data
    logger.info(f"Static row index for AgriDataEncoder.forward (in preprocessed data): {static_row_idx_for_encoder_fwd}")


    os.makedirs(args.output_dir_d1_encoder, exist_ok=True)
    best_encoder_path = os.path.join(args.output_dir_d1_encoder, f"best_agri_encoder_run_{mlflow.active_run().info.run_id if mlflow.active_run() else 'local'}.pth")

    start_time_total_train = time.time()
    for epoch in range(args.encoder_num_epochs):
        agri_encoder.train()
        train_loss_epoch = 0.0
        for x_batch, y_batch, mask_batch in train_loader:
            optimizer.zero_grad()
            outputs = agri_encoder(x_batch, temporal_mask=mask_batch, static_row_index=static_row_idx_for_encoder_fwd)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        avg_train_loss = train_loss_epoch / len(train_loader)
        
        agri_encoder.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for x_batch_val, y_batch_val, mask_batch_val in val_loader:
                outputs_val = agri_encoder(x_batch_val, temporal_mask=mask_batch_val, static_row_index=static_row_idx_for_encoder_fwd)
                loss_val = criterion(outputs_val, y_batch_val)
                val_loss_epoch += loss_val.item()
        avg_val_loss = val_loss_epoch / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step() 

        logger.info(f"Epoch [{epoch+1}/{args.encoder_num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        if mlflow.active_run():
            mlflow.log_metrics({"enc_train_loss": avg_train_loss, "enc_val_loss": avg_val_loss, "enc_lr": current_lr}, step=epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(agri_encoder.state_dict(), best_encoder_path)
            logger.info(f"Val loss improved. Saved model to {best_encoder_path}")
            if mlflow.active_run(): mlflow.log_metric("enc_best_val_loss_epoch", best_val_loss, step=epoch)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.encoder_early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1} after {args.encoder_early_stopping_patience} epochs with no improvement.")
                if mlflow.active_run(): mlflow.log_param("enc_training_stopped_early_epoch", epoch+1)
                break
    
    total_training_time = time.time() - start_time_total_train
    logger.info(f"AgriDataEncoder training finished. Total time: {total_training_time:.2f}s. Best val_loss: {best_val_loss:.4f}")
    if mlflow.active_run():
        mlflow.log_metric("enc_total_training_time_seconds", total_training_time)
        mlflow.log_metric("enc_final_best_val_loss", best_val_loss) # Log the overall best
        mlflow.log_param("enc_training_status", "completed" if epochs_no_improve < args.encoder_early_stopping_patience else "early_stopped")
        if os.path.exists(best_encoder_path): # Log best model as artifact
            mlflow.log_artifact(best_encoder_path, artifact_path="d1_trained_encoder_model")


    # 5. EVALUATE THE BEST TRAINED ENCODER on validation set
    logger.info("Step 5: Evaluating the best AgriDataEncoder on validation set...")
    if os.path.exists(best_encoder_path):
        agri_encoder.load_state_dict(torch.load(best_encoder_path))
        logger.info(f"Loaded best encoder weights from {best_encoder_path} for final validation metrics.")
        
        agri_encoder.eval()
        all_y_val_pred_final = []
        all_y_val_true_final = []
        with torch.no_grad():
            for x_batch_val, y_batch_val, mask_batch_val in val_loader:
                outputs_val = agri_encoder(x_batch_val, temporal_mask=mask_batch_val, static_row_index=static_row_idx_for_encoder_fwd)
                all_y_val_pred_final.append(outputs_val.cpu().numpy())
                all_y_val_true_final.append(y_batch_val.cpu().numpy())
                
        y_pred_val_final = np.concatenate(all_y_val_pred_final).squeeze()
        y_true_val_final = np.concatenate(all_y_val_true_final).squeeze()

        final_val_metrics = calculate_regression_metrics(y_true_val_final, y_pred_val_final, prefix="enc_val_final_metrics")
        logger.info(f"Final Encoder Validation Metrics (on best model): {final_val_metrics}")
        if mlflow.active_run():
            log_metrics_to_mlflow(final_val_metrics)
            mlflow.log_param("d1_overall_status", "success_encoder_trained")
    else:
        logger.error(f"Best encoder model not found at {best_encoder_path}. Cannot evaluate.")
        if mlflow.active_run(): mlflow.log_param("d1_overall_status", "failed_best_encoder_not_found")


    logger.info("--- D1: Standalone AgriDataEncoder Training Completed ---")


def main(args):
    logger.info("Starting main execution flow...")
    
    # Setup sys.path to find tabpfn (adjust if your structure is different)
    # This attempts to find 'src' relative to the script's location or its parent.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_guesses = [
        os.path.abspath(os.path.join(script_dir, "..", "src")), # If main.py is in experiments/ TabPFN/src
        os.path.abspath(os.path.join(script_dir, "src")),      # If main.py is in TabPFN/ and src is TabPFN/src
        os.path.abspath(os.path.join(script_dir, ".."))        # If main.py is in TabPFN/src/scripts and TabPFN is root
    ]
    path_added = False
    for prg in project_root_guesses:
        if os.path.isdir(prg) and prg not in sys.path:
            # Check if 'tabpfn' subfolder exists, indicating this might be the correct 'src'
            if os.path.isdir(os.path.join(prg, "tabpfn")):
                sys.path.insert(0, prg)
                logger.info(f"Added to sys.path for module resolution: {prg}")
                path_added = True
                break
    if not path_added:
         # Fallback: add the script's own directory if tabpfn is directly there or a sibling
        if script_dir not in sys.path and (os.path.isdir(os.path.join(script_dir, "tabpfn")) or os.path.isdir(os.path.join(os.path.dirname(script_dir), "tabpfn"))):
            sys.path.insert(0, os.path.dirname(script_dir) if os.path.isdir(os.path.join(os.path.dirname(script_dir), "tabpfn")) else script_dir)
            logger.info(f"Added script's parent/self to sys.path: {sys.path[0]}")
        else:
            logger.warning(f"Could not automatically determine and add 'src' or project root to sys.path. Ensure 'tabpfn' is importable. Current sys.path: {sys.path}")


    if not CORE_CLASSES_IMPORTED and args.require_core_classes: 
        logger.critical("Core classes (AgriDataPreprocessor/AgriDataEncoder) import failed and are required. Exiting.")
        sys.exit(1) 
    elif not CORE_CLASSES_IMPORTED:
        logger.warning("Core classes (AgriDataPreprocessor/AgriDataEncoder) import failed. D1 training will be skipped.")

    mlflow.set_experiment(args.mlflow_experiment_name) 

    with mlflow.start_run(run_name=args.mlflow_run_name) as run: 
        if run is None:
            logger.error("Failed to start MLflow run. Check MLflow configuration.")
        else:
             logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        if mlflow.active_run():
            mlflow.log_param("script_name", "main.py_D1_EncoderTrain")
            mlflow.log_params(vars(args)) 

        logger.info("--- Phase 1: Data Loading & Preparation ---")
        # ... (data loading and validation logic remains the same as in main_py_v3)
        X_orig_train, y_orig_train, info_train = None, None, None
        if args.train_file_path:
            if not os.path.exists(args.train_file_path):
                logger.error(f"Train file path does not exist: {args.train_file_path}")
                if mlflow.active_run(): mlflow.log_param("data_loading_status", "train_file_not_found")
                return
            load_result_train = load_npz_data(args.train_file_path)
            if load_result_train:
                X_orig_train, y_orig_train, info_train = load_result_train
                logger.info("Original training data loaded successfully.")
            else:
                logger.error("Failed to load original training data.")
                if mlflow.active_run(): mlflow.log_param("data_loading_status", "train_load_failed")
                return
        else: 
            logger.error("Train file path not provided. This script requires training data.")
            if mlflow.active_run(): mlflow.log_param("data_loading_status", "train_file_missing")
            return

        X_test, y_test, info_test = None, None, None # Test set not used in D1 encoder training
        # ... (validation logic for train data)
        logger.info("Validating loaded training data...")
        if not validate_data(X_orig_train, y_orig_train, info_train):
            logger.error("Original training data validation failed.")
            if mlflow.active_run(): mlflow.log_param("data_validation_status", "train_validation_failed")
            return
        if mlflow.active_run(): mlflow.log_param("data_validation_status", "success")
        
        logger.info("Splitting data into new train and validation sets...")
        X_train, X_val, y_train, y_val = split_data(
            X_orig_train, y_orig_train, 
            test_size=args.val_split_size, random_state=args.random_seed
        )
        logger.info(f"Data split: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"            X_val shape: {X_val.shape},   y_val shape: {y_val.shape}")
        
        if CORE_CLASSES_IMPORTED:
            run_d1_train_agri_encoder(X_train, y_train, X_val, y_val, info_train, args)
        else:
            logger.error("Skipping D1 AgriDataEncoder training due to import failures.")
            if mlflow.active_run(): mlflow.log_param("d1_encoder_training_overall_status", "skipped_due_to_import_error_at_start")

    logger.info("Main execution flow completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN Agri Yield: D1 - Standalone AgriDataEncoder Training")
    
    # Data paths
    parser.add_argument("--train_file_path", type=str, required=True, help="Path to train_processed.npz")
    parser.add_argument("--test_file_path", type=str, help="Path to test_processed.npz (not used in D1 encoder training but good to have)")
    parser.add_argument("--output_dir_d1_encoder", type=str, default="./d1_encoder_output", help="Directory to save the trained D1 encoder model and artifacts.")

    # Data splitting
    parser.add_argument("--val_split_size", type=float, default=0.2, help="Validation set proportion.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for all operations.")

    # AgriDataPreprocessor params (passed to AgriDataPreprocessor directly)
    parser.add_argument("--standardize_agri_features", type=lambda x: (str(x).lower() == 'true'), default=True, help="Standardize 3D input features (True/False).")
    parser.add_argument("--handle_missing_agri_values", type=str, default="mean", choices=["mean", "zero", "none"], help="Handle NaNs in 3D input.")

    # AgriDataEncoder structure params (passed to AgriDataEncoder __init__)
    parser.add_argument("--encoder_temporal_hidden_dim", type=int, default=128, help="Hidden dim for temporal encoder part.")
    parser.add_argument("--encoder_static_hidden_dims_str", type=str, default="64", help="Static encoder hidden dims (comma-sep string, e.g., '64,32').")
    parser.add_argument("--encoder_fusion_strategy", type=str, default="gated", choices=["concat", "attention", "gated"], help="Fusion strategy for temporal and static embeddings.")
    # output_dim for AgriDataEncoder will be 1 (for regression training)
    # temporal_input_dim and static_input_dim will be derived from data (n_features_cols)
    parser.add_argument("--encoder_temporal_num_heads", type=int, default=4, help="Number of attention heads for TimeFeaturesEncoder if use_attention is True in its kwargs.")
    # For simplicity, temporal_encoder_kwargs and static_encoder_kwargs are not CLI args for D1
    # We can pass simple defaults or empty dicts.
    # Example: --encoder_temporal_kwargs "{'use_attention': True, 'num_heads': 4}" (would need json.loads)
    # For now, we'll keep it simple.
    parser.add_argument("--encoder_temporal_kwargs", type=eval, default=None, help="Dict string for TimeFeaturesEncoder kwargs, e.g., \"{'use_attention':True}\"")
    parser.add_argument("--encoder_static_kwargs", type=eval, default=None, help="Dict string for StaticFeaturesEncoder kwargs")


    # AgriDataEncoder training params
    parser.add_argument("--encoder_lr", type=float, default=1e-3, help="Learning rate for encoder training.")
    parser.add_argument("--encoder_weight_decay", type=float, default=1e-5, help="Weight decay for encoder optimizer.")
    parser.add_argument("--encoder_batch_size", type=int, default=128, help="Batch size for encoder training.")
    parser.add_argument("--encoder_num_epochs", type=int, default=50, help="Number of epochs for encoder training.") # Increased default
    parser.add_argument("--encoder_scheduler_step_size", type=int, default=10, help="Step size for LR scheduler.")
    parser.add_argument("--encoder_scheduler_gamma", type=float, default=0.7, help="Gamma for LR scheduler.")
    parser.add_argument("--encoder_early_stopping_patience", type=int, default=5, help="Patience for early stopping encoder training.") # Increased default

    # MLflow params
    parser.add_argument("--mlflow_experiment_name", type=str, default="TabPFN_Agri_Encoder_Training", help="MLflow experiment name.")
    parser.add_argument("--mlflow_run_name", type=str, default="D1_Encoder_Run", help="MLflow run name.")
    parser.add_argument("--require_core_classes", action='store_true', help="If set, script exits if core classes fail to import.")

    args = parser.parse_args()
    
    # Setup sys.path for local imports (adjust if needed)
    # This is a common pattern: add the parent directory of 'src' if 'src/tabpfn' exists
    # Or add 'src' if 'tabpfn' is directly under 'src'
    # The goal is to make 'from tabpfn.some_module import Something' work.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_guesses = [
        os.path.abspath(os.path.join(script_dir, "..")),      # If main.py is in 'scripts' or 'experiments'
        script_dir                                            # If main.py is in project root
    ]
    path_to_add = None
    for prg in project_root_guesses:
        potential_src_path = os.path.join(prg, "src")
        if os.path.isdir(potential_src_path) and os.path.isdir(os.path.join(potential_src_path, "tabpfn")):
            path_to_add = potential_src_path
            break
        elif os.path.isdir(os.path.join(prg, "tabpfn")): # If tabpfn is directly under project_root
            path_to_add = prg
            break
    
    if path_to_add and path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)
        logger.info(f"Added to sys.path for module resolution: {path_to_add}")
    else:
        logger.warning(f"Could not automatically determine and add 'src' or project root to sys.path. Ensure 'tabpfn' is importable. Current sys.path: {sys.path}")


    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    main(args)
