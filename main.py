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
# (sys.path management code from main_py_v4 can be kept here)


from load_data import load_npz_data, validate_data, split_data, print_data_sample
from evaluate import calculate_regression_metrics, log_metrics_to_mlflow

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

# Global flags for imported classes
AGRI_ENCODERS_IMPORTED = False
AGRI_TABPFN_REGRESSOR_IMPORTED = False
TABPFN_MODEL_CONFIG_IMPORTED = False # Remains False as main.py cannot reliably import these specific classes

try:
    from tabpfn.encoders.agri_encoders import AgriDataPreprocessor, AgriDataEncoder
    logger.info("Successfully imported AgriDataPreprocessor and AgriDataEncoder from tabpfn.encoders.agri_encoders.")
    AGRI_ENCODERS_IMPORTED = True
except ImportError as e:
    logger.error(f"CRITICAL ERROR: Failed to import AgriDataPreprocessor or AgriDataEncoder from tabpfn.encoders.agri_encoders.")
    logger.error(f"Detailed import error: {e}", exc_info=True)

try:
    from tabpfn.agri_tabpfn import AgriTabPFNRegressor
    logger.info("Successfully imported AgriTabPFNRegressor from tabpfn.agri_tabpfn.")
    AGRI_TABPFN_REGRESSOR_IMPORTED = True
except ImportError as e:
    logger.error(f"CRITICAL ERROR: Failed to import AgriTabPFNRegressor from tabpfn.agri_tabpfn.")
    logger.error(f"Detailed import error: {e}", exc_info=True)

# Attempt to import TabPFN model config classes is removed as they are not found in user's env
# TABPFN_MODEL_CONFIG_IMPORTED will remain False.
logger.warning("Skipping import of ModelInterfaceConfig and FeatureProcessingConfig from tabpfn.model.config as they were reported missing or caused issues.")


def get_device():
    """Detects available device (GPU or CPU)"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = "cuda"
        logger.info(f"CUDA detected. Using GPU: {device_name}")
    else:
        device = "cpu"
        logger.info("CUDA not detected. Using CPU.")
    return device

def parse_int_list(string_list: Optional[str]) -> Optional[List[int]]:
    """Helper to parse comma-separated string of ints to list of ints."""
    if string_list is None or string_list.lower() == 'none' or string_list == "":
        return None
    try:
        return [int(item.strip()) for item in string_list.split(',')]
    except ValueError:
        logger.error(f"Cannot parse '{string_list}' into a list of integers. Use comma-separated integers.")
        return None

# --- D1a: Standalone AgriDataEncoder Training ---
def run_d1a_train_agri_encoder( 
    X_train_orig: np.ndarray, y_train_orig: np.ndarray,
    X_val_orig: np.ndarray, y_val_orig: np.ndarray,
    info_train: dict,
    args: argparse.Namespace
):
    """
    Executes D1a phase: Standalone training of AgriDataEncoder.
    This function trains the AgriDataEncoder as a standalone regression model.
    """
    if not AGRI_ENCODERS_IMPORTED: 
        logger.error("Core encoder classes (AgriDataPreprocessor/AgriDataEncoder) not imported. Skipping D1a encoder training.")
        if mlflow.active_run():
            mlflow.log_param("d1a_encoder_training_status", "skipped_import_error")
        return

    logger.info("--- Starting D1a: Standalone AgriDataEncoder Training ---")
    device = get_device()

    # 1. PREPROCESS DATA
    logger.info("Step 1: Preprocessing data for D1a encoder training...")
    preprocessor = AgriDataPreprocessor(
        static_row_index=int(info_train['static_feature_row_index']),
        max_temporal_length=int(info_train['max_len']), 
        standardize_features=args.standardize_agri_features,
        handle_missing_values=args.handle_missing_agri_values
    )
    X_train_processed_np, train_temporal_mask_np = preprocessor.fit_transform(X_train_orig)
    X_val_processed_np, val_temporal_mask_np = preprocessor.transform(X_val_orig)

    X_train_processed = torch.tensor(X_train_processed_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_orig, dtype=torch.float32).unsqueeze(1).to(device) 
    
    if train_temporal_mask_np is not None:
        train_temporal_mask = torch.tensor(train_temporal_mask_np, dtype=torch.bool).to(device)
    else: 
        logger.warning("D1a: Training temporal mask is None. Creating a dummy mask of all Trues.")
        mask_shape_train = (X_train_processed.shape[0], preprocessor.max_temporal_length_)
        train_temporal_mask = torch.ones(mask_shape_train, dtype=torch.bool).to(device)

    X_val_processed = torch.tensor(X_val_processed_np, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val_orig, dtype=torch.float32).unsqueeze(1).to(device)
    
    if val_temporal_mask_np is not None:
        val_temporal_mask = torch.tensor(val_temporal_mask_np, dtype=torch.bool).to(device)
    else:
        logger.warning("D1a: Validation temporal mask is None. Creating a dummy mask of all Trues.")
        mask_shape_val = (X_val_processed.shape[0], preprocessor.max_temporal_length_)
        val_temporal_mask = torch.ones(mask_shape_val, dtype=torch.bool).to(device)
    
    logger.info(f"  D1a: X_train_processed shape: {X_train_processed.shape}, y_train shape: {y_train.shape}")
    logger.info(f"  D1a: X_val_processed shape: {X_val_processed.shape}, y_val shape: {y_val.shape}")

    if mlflow.active_run():
        mlflow.log_params({
            "d1a_prep_static_row_index": preprocessor.static_row_index,
            "d1a_prep_max_temporal_length_input": preprocessor.max_temporal_length,
            "d1a_prep_max_temporal_length_fitted": preprocessor.max_temporal_length_,
            "d1a_prep_standardize": preprocessor.standardize_features,
            "d1a_prep_missing_values": preprocessor.handle_missing_values
        })

    # 2. INITIALIZE AgriDataEncoder MODEL for D1a
    logger.info("Step 2: Initializing AgriDataEncoder for D1a training...")
    n_features_cols = X_train_processed.shape[2] 
    parsed_static_hidden_dims_d1a = parse_int_list(args.d1a_encoder_static_hidden_dims_str)
    default_temporal_encoder_kwargs_d1a = {'num_heads': args.d1a_encoder_temporal_num_heads}
    
    initial_encoder_output_dim_d1a = 1 
    
    agri_encoder_d1a = AgriDataEncoder(
        temporal_input_dim=n_features_cols,
        static_input_dim=n_features_cols,
        output_dim=initial_encoder_output_dim_d1a, 
        temporal_hidden_dim=args.d1a_encoder_temporal_hidden_dim,
        static_hidden_dims=parsed_static_hidden_dims_d1a if parsed_static_hidden_dims_d1a is not None else [64], 
        fusion_strategy=args.d1a_encoder_fusion_strategy,
        temporal_encoder_kwargs=args.d1a_encoder_temporal_kwargs if args.d1a_encoder_temporal_kwargs else default_temporal_encoder_kwargs_d1a,
        static_encoder_kwargs=args.d1a_encoder_static_kwargs if args.d1a_encoder_static_kwargs else {}
    ).to(device)

    logger.info(f"D1a AgriDataEncoder initialized. Requested output_dim: {initial_encoder_output_dim_d1a}, Actual internal output_dim (due to fusion logic): {agri_encoder_d1a.output_dim}.")
    if mlflow.active_run():
        mlflow.log_params({
            "d1a_enc_temporal_input_dim_sub": n_features_cols, "d1a_enc_static_input_dim_sub": n_features_cols,
            "d1a_enc_requested_output_dim": initial_encoder_output_dim_d1a,
            "d1a_enc_actual_internal_output_dim": agri_encoder_d1a.output_dim,
            "d1a_enc_temporal_hidden_dim": args.d1a_encoder_temporal_hidden_dim,
            "d1a_enc_static_hidden_dims": str(parsed_static_hidden_dims_d1a), 
            "d1a_enc_fusion_strategy": args.d1a_encoder_fusion_strategy,
            "d1a_enc_temporal_kwargs": str(args.d1a_encoder_temporal_kwargs if args.d1a_encoder_temporal_kwargs else default_temporal_encoder_kwargs_d1a),
            "d1a_enc_static_kwargs": str(args.d1a_encoder_static_kwargs if args.d1a_encoder_static_kwargs else {}),
        })

    # 3. DEFINE LOSS FUNCTION AND OPTIMIZER for D1a
    logger.info("Step 3: Defining loss function and optimizer for D1a encoder training...")
    criterion_d1a = nn.MSELoss()
    optimizer_d1a = optim.AdamW(
        agri_encoder_d1a.parameters(), lr=args.d1a_encoder_lr, weight_decay=args.d1a_encoder_weight_decay
    )
    scheduler_d1a = optim.lr_scheduler.StepLR(
        optimizer_d1a, step_size=args.d1a_encoder_scheduler_step_size, gamma=args.d1a_encoder_scheduler_gamma
    )
    if mlflow.active_run():
        mlflow.log_params({
            "d1a_enc_train_optimizer": "AdamW", "d1a_enc_train_lr": args.d1a_encoder_lr,
            "d1a_enc_train_weight_decay": args.d1a_encoder_weight_decay, "d1a_enc_train_scheduler": "StepLR",
            "d1a_enc_train_scheduler_step": args.d1a_encoder_scheduler_step_size, 
            "d1a_enc_train_scheduler_gamma": args.d1a_encoder_scheduler_gamma,
            "d1a_enc_train_loss_func": "MSELoss"
        })

    # 4. TRAINING LOOP for D1a
    logger.info("Step 4: Starting D1a AgriDataEncoder training loop...")
    train_dataset_d1a = TensorDataset(X_train_processed, y_train, train_temporal_mask)
    train_loader_d1a = DataLoader(train_dataset_d1a, batch_size=args.d1a_encoder_batch_size, shuffle=True)
    val_dataset_d1a = TensorDataset(X_val_processed, y_val, val_temporal_mask)
    val_loader_d1a = DataLoader(val_dataset_d1a, batch_size=args.d1a_encoder_batch_size, shuffle=False)

    best_val_loss_d1a = float('inf')
    epochs_no_improve_d1a = 0
    
    final_rows_in_preprocessed_data = preprocessor.max_temporal_length_ + 1 
    static_row_idx_for_encoder_fwd = preprocessor.static_row_index % final_rows_in_preprocessed_data
    
    os.makedirs(args.output_dir_d1_encoder, exist_ok=True)
    best_encoder_path_d1a = os.path.join(args.output_dir_d1_encoder, f"d1a_best_agri_encoder_run_{mlflow.active_run().info.run_id if mlflow.active_run() else 'local'}.pth")

    start_time_total_train_d1a = time.time()
    for epoch in range(args.d1a_encoder_num_epochs):
        agri_encoder_d1a.train()
        train_loss_epoch_d1a = 0.0
        for x_batch, y_batch, mask_batch in train_loader_d1a:
            optimizer_d1a.zero_grad()
            outputs = agri_encoder_d1a(x_batch, temporal_mask=mask_batch, static_row_index=static_row_idx_for_encoder_fwd)
            
            outputs_for_loss = outputs
            if outputs.shape[1] > 1 and y_batch.shape[1] == 1: 
                outputs_for_loss = outputs[:, 0:1]
                
            loss = criterion_d1a(outputs_for_loss, y_batch)
            loss.backward()
            optimizer_d1a.step()
            train_loss_epoch_d1a += loss.item()
        avg_train_loss_d1a = train_loss_epoch_d1a / len(train_loader_d1a)
        
        agri_encoder_d1a.eval()
        val_loss_epoch_d1a = 0.0
        with torch.no_grad():
            for x_batch_val, y_batch_val, mask_batch_val in val_loader_d1a:
                outputs_val = agri_encoder_d1a(x_batch_val, temporal_mask=mask_batch_val, static_row_index=static_row_idx_for_encoder_fwd)
                outputs_val_for_loss = outputs_val
                if outputs_val.shape[1] > 1 and y_batch_val.shape[1] == 1:
                    outputs_val_for_loss = outputs_val[:, 0:1]
                loss_val = criterion_d1a(outputs_val_for_loss, y_batch_val)
                val_loss_epoch_d1a += loss_val.item()
        avg_val_loss_d1a = val_loss_epoch_d1a / len(val_loader_d1a)
        current_lr_d1a = scheduler_d1a.get_last_lr()[0]
        scheduler_d1a.step() 

        logger.info(f"D1a Epoch [{epoch+1}/{args.d1a_encoder_num_epochs}], Train Loss: {avg_train_loss_d1a:.4f}, Val Loss: {avg_val_loss_d1a:.4f}, LR: {current_lr_d1a:.6f}")
        if mlflow.active_run():
            mlflow.log_metrics({"d1a_enc_train_loss": avg_train_loss_d1a, "d1a_enc_val_loss": avg_val_loss_d1a, "d1a_enc_lr": current_lr_d1a}, step=epoch)

        if avg_val_loss_d1a < best_val_loss_d1a:
            best_val_loss_d1a = avg_val_loss_d1a
            epochs_no_improve_d1a = 0
            torch.save(agri_encoder_d1a.state_dict(), best_encoder_path_d1a)
            logger.info(f"D1a Val loss improved. Saved model to {best_encoder_path_d1a}")
            if mlflow.active_run(): mlflow.log_metric("d1a_enc_best_val_loss_epoch", best_val_loss_d1a, step=epoch)
        else:
            epochs_no_improve_d1a += 1
            if epochs_no_improve_d1a >= args.d1a_encoder_early_stopping_patience:
                logger.info(f"D1a Early stopping at epoch {epoch+1}.")
                if mlflow.active_run(): mlflow.log_param("d1a_enc_training_stopped_early_epoch", epoch+1)
                break
    
    total_training_time_d1a = time.time() - start_time_total_train_d1a
    logger.info(f"D1a AgriDataEncoder training finished. Total time: {total_training_time_d1a:.2f}s. Best val_loss: {best_val_loss_d1a:.4f}")
    if mlflow.active_run():
        mlflow.log_metric("d1a_enc_total_training_time_seconds", total_training_time_d1a)
        mlflow.log_metric("d1a_enc_final_best_val_loss", best_val_loss_d1a) 
        mlflow.log_param("d1a_enc_training_status", "completed" if epochs_no_improve_d1a < args.d1a_encoder_early_stopping_patience else "early_stopped")
        if os.path.exists(best_encoder_path_d1a): 
            mlflow.log_artifact(best_encoder_path_d1a, artifact_path="d1a_trained_encoder_model")

    # 5. EVALUATE THE BEST TRAINED D1a ENCODER
    logger.info("Step 5: Evaluating the best D1a AgriDataEncoder on validation set...")
    if os.path.exists(best_encoder_path_d1a):
        agri_encoder_d1a.load_state_dict(torch.load(best_encoder_path_d1a, map_location=device))
        agri_encoder_d1a.eval()
        all_y_val_pred_final_list_d1a = []
        all_y_val_true_final_list_d1a = []
        with torch.no_grad():
            for x_batch_val, y_batch_val, mask_batch_val in val_loader_d1a:
                outputs_val = agri_encoder_d1a(x_batch_val, temporal_mask=mask_batch_val, static_row_index=static_row_idx_for_encoder_fwd)
                pred_for_metrics = outputs_val.cpu().numpy()
                if pred_for_metrics.ndim > 1 and pred_for_metrics.shape[1] > 1 and y_batch_val.shape[1] == 1: 
                    logger.debug(f"D1a Final Eval: Encoder output {pred_for_metrics.shape[1]}D. Using output[:, 0] for metrics.")
                    pred_for_metrics = pred_for_metrics[:, 0] 
                all_y_val_pred_final_list_d1a.append(pred_for_metrics)
                all_y_val_true_final_list_d1a.append(y_batch_val.cpu().numpy().squeeze())
        y_pred_val_final_d1a = np.concatenate(all_y_val_pred_final_list_d1a)
        y_true_val_final_d1a = np.concatenate(all_y_val_true_final_list_d1a)
        if y_pred_val_final_d1a.ndim > 1 and y_pred_val_final_d1a.shape[1] == 1:
            y_pred_val_final_d1a = y_pred_val_final_d1a.squeeze(axis=1)
        
        final_val_metrics_d1a = calculate_regression_metrics(y_true_val_final_d1a, y_pred_val_final_d1a, prefix="d1a_enc_val_final_metrics")
        logger.info(f"D1a Final Encoder Validation Metrics: {final_val_metrics_d1a}")
        if mlflow.active_run():
            log_metrics_to_mlflow(final_val_metrics_d1a)
            mlflow.log_param("d1a_overall_status", "success_encoder_trained_and_evaluated")
    else:
        logger.error(f"D1a Best encoder model not found at {best_encoder_path_d1a}. Cannot evaluate.")
        if mlflow.active_run(): mlflow.log_param("d1a_overall_status", "failed_best_encoder_not_found")
    logger.info("--- D1a: Standalone AgriDataEncoder Training Completed ---")


# --- D1b: AgriTabPFNRegressor Benchmark ---
def run_d1b_agri_tabpfn_benchmark(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    info_train: dict,
    args: argparse.Namespace
):
    """
    Executes D1b phase: Benchmark training of AgriTabPFNRegressor.
    The internal AgriDataEncoder is NOT pre-trained for this phase; its weights are
    randomly initialized based on its structure defined by __init__ params.
    """
    if not AGRI_TABPFN_REGRESSOR_IMPORTED:
        logger.error("AgriTabPFNRegressor class not imported. Skipping D1b benchmark.")
        if mlflow.active_run():
            mlflow.log_param("d1b_benchmark_status", "skipped_import_error")
        return
    
    logger.info("--- Starting D1b: AgriTabPFNRegressor Benchmark ---")
    device = get_device()

    # 1. Prepare parameters for AgriTabPFNRegressor.__init__
    logger.info("Step 1: Preparing AgriTabPFNRegressor parameters for D1b...")
    
    static_row_index_from_data = int(info_train['static_feature_row_index'])
    max_temporal_length_from_data = int(info_train['max_len']) if info_train.get('max_len') is not None else None

    parsed_agri_static_hidden_dims = parse_int_list(args.agri_static_hidden_dims_str)

    # For inference_config, pass None. TabPFNRegressor will use its defaults.
    # This avoids issues if ModelInterfaceConfig/FeatureProcessingConfig are not
    # structured as expected or if main.py cannot import them.
    current_inference_config_dict = None
    logger.info("  Setting inference_config=None for AgriTabPFNRegressor. TabPFN will use its default n_quantiles.")
    if args.tabpfn_n_quantiles is not None:
        logger.warning(f"  User specified --tabpfn_n_quantiles={args.tabpfn_n_quantiles}, "
                       "but this will be IGNORED for this run to rely on TabPFN's internal defaults "
                       "due to previous errors with 'Unknown kwarg: n_quantiles'.")


    agri_tabpfn_init_params = {
        "temporal_hidden_dim": args.agri_temporal_hidden_dim,
        "static_hidden_dims": parsed_agri_static_hidden_dims, 
        "fusion_strategy": args.agri_fusion_strategy,
        "encoded_embedding_dim": args.agri_encoded_embedding_dim, 
        "static_row_index": static_row_index_from_data,
        "max_temporal_length": max_temporal_length_from_data,
        "standardize_agri_features": args.standardize_agri_features,
        "handle_missing_agri_values": args.handle_missing_agri_values, 
        "agri_encoder_kwargs": args.agri_encoder_kwargs if args.agri_encoder_kwargs else {},
        "n_estimators": args.tabpfn_n_estimators,
        "device": device,
        "random_state": args.random_seed,
        "model_path": "auto", 
        "fit_mode": "fit_preprocessors",
        "ignore_pretraining_limits": args.tabpfn_ignore_pretraining_limits,
        "inference_config": current_inference_config_dict # This will be None
    }
    logger.info(f"  AgriTabPFNRegressor __init__ params: {agri_tabpfn_init_params}")
    if mlflow.active_run():
        log_params_dict = {}
        for k, v_param in agri_tabpfn_init_params.items():
            if k == "inference_config" and v_param is None:
                 log_params_dict["d1b_inference_n_quantiles_used"] = "TabPFN_Default (inference_config=None)"
            elif isinstance(v_param, (dict, list)): 
                log_params_dict[f"d1b_{k}"] = str(v_param)
            else:
                 log_params_dict[f"d1b_{k}"] = v_param
        mlflow.log_params(log_params_dict)


    # 2. Instantiate AgriTabPFNRegressor
    logger.info("Step 2: Instantiating AgriTabPFNRegressor for D1b...")
    try:
        agri_model_d1b = AgriTabPFNRegressor(**agri_tabpfn_init_params)
        logger.info(f"AgriTabPFNRegressor for D1b instantiated successfully. Device: {agri_model_d1b.device}")
    except Exception as e:
        logger.error(f"Error instantiating AgriTabPFNRegressor for D1b: {e}", exc_info=True)
        if mlflow.active_run():
            mlflow.log_param("d1b_instantiation_status", "failed")
            mlflow.log_param("d1b_instantiation_error", str(e))
        return

    # 3. Fit AgriTabPFNRegressor
    logger.info("Step 3: Fitting AgriTabPFNRegressor for D1b...")
    start_time_fit_d1b = time.time()
    try:
        agri_model_d1b.fit(X_train, y_train)
        fit_time_d1b = time.time() - start_time_fit_d1b
        logger.info(f"AgriTabPFNRegressor D1b fitting completed. Time: {fit_time_d1b:.2f}s")
        if mlflow.active_run():
            mlflow.log_metric("d1b_model_fitting_time_seconds", fit_time_d1b)
            mlflow.log_param("d1b_fitting_status", "success")
    except Exception as e:
        logger.error(f"Error fitting AgriTabPFNRegressor for D1b: {e}", exc_info=True)
        if mlflow.active_run():
            mlflow.log_param("d1b_fitting_status", "failed")
            mlflow.log_param("d1b_fitting_error", str(e))
        return

    # 4. Evaluate on Validation Set
    logger.info("Step 4: Evaluating D1b AgriTabPFNRegressor on validation set...")
    if X_val is not None and y_val is not None:
        start_time_predict_d1b = time.time()
        try:
            y_pred_val_d1b = agri_model_d1b.predict(X_val)
            predict_time_d1b = time.time() - start_time_predict_d1b
            logger.info(f"D1b validation prediction completed. Time: {predict_time_d1b:.2f}s")
            if mlflow.active_run():
                mlflow.log_metric("d1b_validation_prediction_time_seconds", predict_time_d1b)

            val_metrics_d1b = calculate_regression_metrics(y_val, y_pred_val_d1b, prefix="d1b_val")
            logger.info(f"D1b Validation Metrics: {val_metrics_d1b}")
            if mlflow.active_run():
                log_metrics_to_mlflow(val_metrics_d1b)
                mlflow.log_param("d1b_validation_status", "success")
        except Exception as e:
            logger.error(f"Error during D1b validation set evaluation: {e}", exc_info=True)
            if mlflow.active_run():
                mlflow.log_param("d1b_validation_status", "failed")
                mlflow.log_param("d1b_validation_error", str(e))
    else:
        logger.warning("Validation data (X_val, y_val) not available for D1b evaluation.")
        if mlflow.active_run(): mlflow.log_param("d1b_validation_status", "skipped_no_data")
    
    if mlflow.active_run(): mlflow.log_param("d1b_overall_status", "completed")
    logger.info("--- D1b: AgriTabPFNRegressor Benchmark Completed ---")


def main(args):
    logger.info("Starting main execution flow...")
    
    # Setup sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_check = [
        os.path.join(script_dir, "src"), os.path.join(os.path.abspath(os.path.join(script_dir, "..")), "src"),
        script_dir, os.path.abspath(os.path.join(script_dir, ".."))
    ]
    path_added_for_tabpfn = False
    for p_candidate in paths_to_check:
        if os.path.isdir(os.path.join(p_candidate, "tabpfn")):
            if p_candidate not in sys.path:
                sys.path.insert(0, p_candidate)
                logger.info(f"Added to sys.path for 'tabpfn' package: {p_candidate}")
                path_added_for_tabpfn = True
                break
        potential_src_dir = os.path.join(p_candidate, "src")
        if os.path.isdir(potential_src_dir) and os.path.isdir(os.path.join(potential_src_dir, "tabpfn")):
            if potential_src_dir not in sys.path:
                sys.path.insert(0, potential_src_dir)
                logger.info(f"Added to sys.path for 'tabpfn' package (via src): {potential_src_dir}")
                path_added_for_tabpfn = True
                break
    if not path_added_for_tabpfn:
         logger.warning(f"Could not automatically determine and add path for 'tabpfn' package. Ensure it's importable. Current sys.path: {sys.path}")

    # Core Class Import Check
    if args.require_core_classes:
        if args.run_phase == "d1a_train_encoder" and not AGRI_ENCODERS_IMPORTED:
            logger.critical("AgriDataPreprocessor/AgriDataEncoder import failed and are required for D1a. Exiting.")
            sys.exit(1)
        if args.run_phase == "d1b_tabpfn_benchmark" and not AGRI_TABPFN_REGRESSOR_IMPORTED:
            logger.critical("AgriTabPFNRegressor import failed and is required for D1b. Exiting.")
            sys.exit(1)

    mlflow.set_experiment(args.mlflow_experiment_name) 

    with mlflow.start_run(run_name=f"{args.mlflow_run_name}_{args.run_phase}") as run: 
        if run is None:
            logger.error("Failed to start MLflow run. Check MLflow configuration.")
        else:
             logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        if mlflow.active_run():
            mlflow.log_param("script_name", f"main.py_phase_{args.run_phase}")
            mlflow.log_params(vars(args)) 

        logger.info("--- Phase 1: Data Loading & Preparation ---")
        X_orig_train, y_orig_train, info_train = None, None, None
        if args.train_file_path: 
            if not os.path.exists(args.train_file_path):
                logger.error(f"Train file path does not exist: {args.train_file_path}")
                if mlflow.active_run(): mlflow.log_param("data_loading_status", "train_file_not_found")
                return
            load_result_train = load_npz_data(args.train_file_path)
            if load_result_train:
                X_orig_train_loaded, y_orig_train_loaded, info_train = load_result_train
                logger.info("Original training data loaded successfully.")
                
                # --- Data Subsampling for Debugging ---
                if args.debug_max_train_samples is not None and args.debug_max_train_samples > 0:
                    if args.debug_max_train_samples < X_orig_train_loaded.shape[0]:
                        logger.warning(
                            f"DEBUG: Truncating original training data to {args.debug_max_train_samples} samples "
                            f"from {X_orig_train_loaded.shape[0]} samples."
                        )
                        # Shuffle before truncating to get a more representative subset
                        indices = np.arange(X_orig_train_loaded.shape[0])
                        np.random.seed(args.random_seed) # Ensure consistent shuffle for debugging
                        np.random.shuffle(indices)
                        X_orig_train = X_orig_train_loaded[indices[:args.debug_max_train_samples]]
                        y_orig_train = y_orig_train_loaded[indices[:args.debug_max_train_samples]]
                        # Update info_train if it contains sample count, though it's usually about features/cols
                        info_train['n_samples_in_split_original'] = X_orig_train_loaded.shape[0]
                        info_train['n_samples_in_split'] = X_orig_train.shape[0]
                        if mlflow.active_run(): mlflow.log_param("debug_data_truncated_to_samples", X_orig_train.shape[0])
                    else:
                        X_orig_train, y_orig_train = X_orig_train_loaded, y_orig_train_loaded
                        logger.info(f"Debug_max_train_samples ({args.debug_max_train_samples}) is >= actual samples. Using full loaded training data.")
                else:
                    X_orig_train, y_orig_train = X_orig_train_loaded, y_orig_train_loaded
                # --- End Data Subsampling ---

            else:
                logger.error("Failed to load original training data.")
                if mlflow.active_run(): mlflow.log_param("data_loading_status", "train_load_failed")
                return
        else: 
            logger.error("Train file path not provided. This script requires training data.")
            if mlflow.active_run(): mlflow.log_param("data_loading_status", "train_file_missing")
            return
        
        if args.test_file_path: 
            if not os.path.exists(args.test_file_path):
                logger.warning(f"Test file path does not exist: {args.test_file_path}")
            else:
                load_result_test = load_npz_data(args.test_file_path)
                if load_result_test:
                    logger.info("Test data loaded successfully (features and info only).")
                else:
                    logger.warning("Failed to load test data.")
        else:
            logger.info("Test file path not provided, skipping test data loading.")
        
        if mlflow.active_run(): mlflow.log_param("data_loading_status", "success")

        logger.info("Validating loaded training data...")
        if not validate_data(X_orig_train, y_orig_train, info_train): # Validate potentially subsampled data
            logger.error("Original training data validation failed.")
            if mlflow.active_run(): mlflow.log_param("data_validation_status", "train_validation_failed")
            return
        if mlflow.active_run(): mlflow.log_param("data_validation_status", "success")
        
        logger.info("Splitting data into new train and validation sets...")
        X_train, X_val, y_train_split, y_val_split = split_data(
            X_orig_train, y_orig_train, 
            test_size=args.val_split_size, random_state=args.random_seed
        )
        logger.info(f"Data split: X_train shape: {X_train.shape}, y_train shape: {y_train_split.shape}")
        logger.info(f"            X_val shape: {X_val.shape},   y_val shape: {y_val_split.shape}")
        
        # Execute selected D1 Phase
        if args.run_phase == "d1a_train_encoder":
            if AGRI_ENCODERS_IMPORTED:
                run_d1a_train_agri_encoder(X_train, y_train_split, X_val, y_val_split, info_train, args)
            else:
                logger.error("Skipping D1a AgriDataEncoder training due to import failures for AgriDataPreprocessor/AgriDataEncoder.")
                if mlflow.active_run(): mlflow.log_param("d1a_overall_status", "skipped_due_to_import_error_at_start")
        
        elif args.run_phase == "d1b_tabpfn_benchmark":
            if AGRI_TABPFN_REGRESSOR_IMPORTED: 
                run_d1b_agri_tabpfn_benchmark(X_train, y_train_split, X_val, y_val_split, info_train, args)
            else:
                logger.error("Skipping D1b AgriTabPFNRegressor benchmark due to import failure for AgriTabPFNRegressor.")
                if mlflow.active_run(): mlflow.log_param("d1b_overall_status", "skipped_due_to_import_error_at_start")
        else:
            logger.error(f"Unknown run_phase: {args.run_phase}. Choose 'd1a_train_encoder' or 'd1b_tabpfn_benchmark'.")

    logger.info("Main execution flow completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN Agri Yield: D1 Experimental Phases")
    
    # Common arguments
    parser.add_argument("--train_file_path", type=str, required=True, help="Path to train_processed.npz")
    parser.add_argument("--test_file_path", type=str, help="Path to test_processed.npz (optional)")
    parser.add_argument("--val_split_size", type=float, default=0.2, help="Validation set proportion.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--standardize_agri_features", type=lambda x: (str(x).lower() == 'true'), default=True, help="Standardize input (True/False).")
    parser.add_argument("--handle_missing_agri_values", type=str, default="mean", choices=["mean", "zero", "none"], help="Handle NaNs in input.")
    parser.add_argument("--mlflow_experiment_name", type=str, default="TabPFN_Agri_Yield", help="MLflow experiment name.")
    parser.add_argument("--mlflow_run_name", type=str, default="D1_Run", help="Base MLflow run name (phase will be appended).")
    parser.add_argument("--require_core_classes", action='store_true', help="Exit if core classes for the selected phase fail to import.")
    parser.add_argument("--run_phase", type=str, required=True, choices=["d1a_train_encoder", "d1b_tabpfn_benchmark"], help="Which D1 phase to run.")
    parser.add_argument("--debug_max_train_samples", type=int, default=None, help="DEBUG: Max number of samples to load from original training data for quick testing. Default: None (use all).")


    # Arguments for D1a: Standalone AgriDataEncoder Training
    d1a_group = parser.add_argument_group('D1a: Standalone AgriDataEncoder Training Parameters')
    d1a_group.add_argument("--output_dir_d1_encoder", type=str, default="./d1_encoder_output", help="[D1a] Directory to save trained D1 encoder.")
    d1a_group.add_argument("--d1a_encoder_temporal_hidden_dim", type=int, default=128, help="[D1a] Temporal encoder hidden dim.")
    d1a_group.add_argument("--d1a_encoder_static_hidden_dims_str", type=str, default="64", help="[D1a] Static encoder hidden dims (comma-sep string).")
    d1a_group.add_argument("--d1a_encoder_fusion_strategy", type=str, default="gated", choices=["concat", "attention", "gated"], help="[D1a] Fusion strategy.")
    d1a_group.add_argument("--d1a_encoder_temporal_num_heads", type=int, default=4, help="[D1a] Num heads for TimeFeaturesEncoder attention.")
    d1a_group.add_argument("--d1a_encoder_temporal_kwargs", type=eval, default=None, help="[D1a] Dict string for TimeFeaturesEncoder kwargs.")
    d1a_group.add_argument("--d1a_encoder_static_kwargs", type=eval, default=None, help="[D1a] Dict string for StaticFeaturesEncoder kwargs.")
    d1a_group.add_argument("--d1a_encoder_lr", type=float, default=1e-3, help="[D1a] Encoder training LR.")
    d1a_group.add_argument("--d1a_encoder_weight_decay", type=float, default=1e-5, help="[D1a] Encoder training weight decay.")
    d1a_group.add_argument("--d1a_encoder_batch_size", type=int, default=64, help="[D1a] Encoder training batch size.")
    d1a_group.add_argument("--d1a_encoder_num_epochs", type=int, default=50, help="[D1a] Encoder training epochs.")
    d1a_group.add_argument("--d1a_encoder_scheduler_step_size", type=int, default=15, help="[D1a] LR scheduler step size.")
    d1a_group.add_argument("--d1a_encoder_scheduler_gamma", type=float, default=0.5, help="[D1a] LR scheduler gamma.")
    d1a_group.add_argument("--d1a_encoder_early_stopping_patience", type=int, default=7, help="[D1a] Early stopping patience.")

    # Arguments for D1b: AgriTabPFNRegressor Benchmark
    d1b_group = parser.add_argument_group('D1b: AgriTabPFNRegressor Benchmark Parameters')
    d1b_group.add_argument("--agri_temporal_hidden_dim", type=int, default=32, help="[D1b] AgriTabPFNRegressor: temporal_hidden_dim for internal encoder.") 
    d1b_group.add_argument("--agri_static_hidden_dims_str", type=str, default="32", help="[D1b] AgriTabPFNRegressor: static_hidden_dims for internal encoder (comma-sep string).") 
    d1b_group.add_argument("--agri_fusion_strategy", type=str, default="gated", choices=["concat", "attention", "gated"], help="[D1b] AgriTabPFNRegressor: fusion_strategy for internal encoder.")
    d1b_group.add_argument("--agri_encoded_embedding_dim", type=int, default=64, help="[D1b] AgriTabPFNRegressor: encoded_embedding_dim (output of internal encoder to TabPFN). Should be <=100.") 
    d1b_group.add_argument("--agri_encoder_kwargs", type=eval, default="{'temporal_encoder_kwargs': {'num_heads': 1}}", help="[D1b] Dict string for AgriTabPFNRegressor's agri_encoder_kwargs.") 
    d1b_group.add_argument("--tabpfn_n_estimators", type=int, default=16, help="[D1b] AgriTabPFNRegressor: n_estimators for internal TabPFN.")
    d1b_group.add_argument("--tabpfn_ignore_pretraining_limits", type=lambda x: (str(x).lower() == 'true'), default=True, help="[D1b] AgriTabPFNRegressor: ignore_pretraining_limits for internal TabPFN (True/False).") 
    d1b_group.add_argument("--tabpfn_n_quantiles", type=int, default=None, help="[D1b] n_quantiles for TabPFN's internal QuantileTransformer. If None, TabPFN uses its default. Max 10000 due to subsampling.")


    args = parser.parse_args()
    
    # Setup sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_check = [
        os.path.join(script_dir, "src"), os.path.join(os.path.abspath(os.path.join(script_dir, "..")), "src"),
        script_dir, os.path.abspath(os.path.join(script_dir, ".."))
    ]
    path_added_for_tabpfn = False
    for p_candidate in paths_to_check:
        if os.path.isdir(os.path.join(p_candidate, "tabpfn")):
            if p_candidate not in sys.path:
                sys.path.insert(0, p_candidate)
                logger.info(f"Added to sys.path for 'tabpfn' package: {p_candidate}")
                path_added_for_tabpfn = True
                break
        potential_src_dir = os.path.join(p_candidate, "src")
        if os.path.isdir(potential_src_dir) and os.path.isdir(os.path.join(potential_src_dir, "tabpfn")):
            if potential_src_dir not in sys.path:
                sys.path.insert(0, potential_src_dir)
                logger.info(f"Added to sys.path for 'tabpfn' package (via src): {potential_src_dir}")
                path_added_for_tabpfn = True
                break
    if not path_added_for_tabpfn:
         logger.warning(f"Could not automatically determine and add path for 'tabpfn' package. Ensure it's importable. Current sys.path: {sys.path}")

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    main(args)
