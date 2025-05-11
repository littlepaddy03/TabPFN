# File: TabPFN/scripts/preprocess_agri_datasets.py
# pylint: disable=line-too-long
"""
Processes pre-split agricultural train and test datasets.

This script loads data from 'train' and 'test' subdirectories within a base
data directory using the agri_utils.data_loader, performs the necessary
transformations, and optionally saves the processed datasets (features, targets,
and sample information) as .npz files.
"""
import argparse
import logging
from pathlib import Path
import sys
import numpy as np
from typing import List, Tuple, Dict, Any # Added import for typing

# Ensure the agri_utils module can be found
# This assumes the script is run from the TabPFN root or TabPFN/src is in PYTHONPATH
try:
    from tabpfn.agri_utils.data_loader import load_and_transform_agri_data
except ImportError:
    # Fallback for running the script directly if the package structure isn't fully set up
    # or if running from a different working directory.
    # Adjust path as necessary if TabPFN/src is not directly in sys.path
    # This adds the parent directory of 'scripts' (i.e., TabPFN root) to sys.path,
    # then 'src' to effectively find 'tabpfn.agri_utils.data_loader'
    SCRIPT_DIR = Path(__file__).resolve().parent
    TABPFN_ROOT = SCRIPT_DIR.parent
    SRC_DIR = TABPFN_ROOT / "src"
    if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if TABPFN_ROOT.exists() and str(TABPFN_ROOT) not in sys.path: # If src is not directly there
        sys.path.insert(0, str(TABPFN_ROOT))


    from tabpfn.agri_utils.data_loader import load_and_transform_agri_data


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_processed_data(
    output_dir: Path,
    dataset_name: str,
    features: List[np.ndarray],
    targets: np.ndarray,
    info: List[Tuple[Any, ...]] # Changed from List[tuple] to List[Tuple[Any, ...]] for more specific typing
) -> None:
    """
    Saves the processed features, targets, and info to an .npz file.

    Args:
        output_dir: The directory to save the .npz file.
        dataset_name: Name for the dataset (e.g., 'train' or 'test').
        features: List of 3D NumPy arrays for samples.
        targets: 1D NumPy array of target values.
        info: List of sample information tuples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # Note: np.savez can't directly save a list of arrays of varying shapes easily
    # into a single array in the .npz. We save them as separate items if needed,
    # or, more commonly for ragged arrays, save X as an object array if all
    # inner arrays are to be treated as one entity, or save pickled.
    # For lists of arrays, it's often better to save them individually or handle
    # them carefully. Here, we'll save 'features' as an object array,
    # which allows arrays of different shapes within the list.
    # However, for better compatibility and to avoid pickling issues,
    # it might be preferable to save X_0, X_1, ... if the number is small,
    # or to pad them to a common shape if feasible before saving.
    # Given X is a list of (n_time_steps_for_sample + 1, N_FEATURES_UNIFIED),
    # n_time_steps_for_sample can vary.
    # We will save X as an object array.
    # For sample_info, it's a list of tuples, which also needs careful handling.
    # np.savez can store it as an object array.

    # Convert list of tuples to a structured array or object array for saving
    # For info, if it's simple tuples of (year, crop_name, lon, lat)
    # an object array is fine.
    try:
        # Attempt to convert info to a structured array if elements are consistent
        # This is more robust if dtypes can be inferred or specified.
        # Example: info_dtype = [('year', 'i4'), ('crop', 'U20'), ('lon', 'f8'), ('lat', 'f8')]
        # For simplicity, we'll use object array for info as well.
        info_array = np.array(info, dtype=object)
    except Exception as e:
        logger.warning(f"Could not convert info to a standard NumPy array, saving as raw object: {e}")
        info_array = np.array(info, dtype=object) # Fallback

    file_path = output_dir / f"{dataset_name}_processed.npz"
    np.savez_compressed(
        file_path,
        features=np.array(features, dtype=object), # Save list of arrays as an object array
        targets=targets,
        info=info_array
    )
    logger.info(f"Saved processed {dataset_name} data to {file_path}")

def main() -> Tuple[List[np.ndarray], np.ndarray, List[Tuple[Any, ...]], List[np.ndarray], np.ndarray, List[Tuple[Any, ...]]]: # Added return type hint
    """
    Main function to load, process, and optionally save datasets.
    Returns the loaded and processed train and test data.
    """
    parser = argparse.ArgumentParser(
        description="Load and preprocess agricultural train/test datasets."
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Base directory containing 'train' and 'test' subdirectories."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional. Directory to save the processed .npz files. "
             "If not provided, data is only loaded and summarized."
    )
    args = parser.parse_args()

    base_data_path = Path(args.base_data_dir)
    train_data_path = base_data_path / "train"
    test_data_path = base_data_path / "test"

    output_path = Path(args.output_dir) if args.output_dir else None

    if not base_data_path.is_dir():
        logger.error(f"Base data directory not found: {base_data_path}")
        sys.exit(1)
    if not train_data_path.is_dir():
        logger.error(f"Train data directory not found: {train_data_path}")
        sys.exit(1)
    if not test_data_path.is_dir():
        logger.error(f"Test data directory not found: {test_data_path}")
        sys.exit(1)

    logger.info(f"Processing datasets from base directory: {base_data_path}")

    # Initialize with empty lists/arrays in case loading fails or data is empty
    X_train: List[np.ndarray] = []
    y_train: np.ndarray = np.array([])
    train_info: List[Tuple[Any, ...]] = []
    X_test: List[np.ndarray] = []
    y_test: np.ndarray = np.array([])
    test_info: List[Tuple[Any, ...]] = []


    # --- Process Training Data ---
    logger.info(f"Loading and transforming training data from: {train_data_path}")
    X_train, y_train, train_info = load_and_transform_agri_data(train_data_path)
    logger.info(f"Loaded {len(X_train)} training samples.")
    if X_train:
        logger.info(f"  Shape of first training sample X: {X_train[0].shape}")
        logger.info(f"  Number of training targets y: {len(y_train)}")
        logger.info(f"  Data type of first training sample X: {X_train[0].dtype}")
        logger.info(f"  Data type of training targets y: {y_train.dtype}")
    else:
        logger.warning("No training data loaded.")


    # --- Process Test Data ---
    logger.info(f"Loading and transforming test data from: {test_data_path}")
    X_test, y_test, test_info = load_and_transform_agri_data(test_data_path)
    logger.info(f"Loaded {len(X_test)} test samples.")
    if X_test:
        logger.info(f"  Shape of first test sample X: {X_test[0].shape}")
        logger.info(f"  Number of test targets y: {len(y_test)}")
        logger.info(f"  Data type of first test sample X: {X_test[0].dtype}")
        logger.info(f"  Data type of test targets y: {y_test.dtype}")
    else:
        logger.warning("No test data loaded.")

    # --- Save Processed Data (Optional) ---
    if output_path:
        logger.info(f"Saving processed data to: {output_path}")
        if X_train or y_train.size > 0 : # Check if there's anything to save
            save_processed_data(output_path, "train", X_train, y_train, train_info)
        else:
            logger.warning("Skipping save for empty training data.")

        if X_test or y_test.size > 0: # Check if there's anything to save
            save_processed_data(output_path, "test", X_test, y_test, test_info)
        else:
            logger.warning("Skipping save for empty test data.")

    logger.info("Dataset processing complete.")
    return X_train, y_train, train_info, X_test, y_test, test_info


if __name__ == "__main__":
    main()
