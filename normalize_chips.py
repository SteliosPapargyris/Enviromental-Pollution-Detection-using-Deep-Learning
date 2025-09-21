import pandas as pd
from pathlib import Path
from utils.apply_normalization import apply_normalization
from utils.config import *

def normalize_all_chips():
    """Normalize all chip CSV files using the current normalization method"""

    chip_files = []
    data_dir = Path("data/out")

    # Load all numbered chip files (1.csv through 5.csv)
    for chip_num in range(1, total_num_chips + 1):
        chip_file = data_dir / f"{chip_num}.csv"
        if chip_file.exists():
            df = pd.read_csv(chip_file)
            chip_files.append(df)

    if not chip_files:
        return

    # Apply normalization to all datasets
    normalized_datasets = apply_normalization(chip_files, CURRENT_NORMALIZATION)

    # Save normalized datasets
    output_dir = Path(f"data/out/{CURRENT_NORMALIZATION.replace('class_based_', '')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, normalized_df in enumerate(normalized_datasets, 1):
        output_file = output_dir / f"chip_{i}_{CURRENT_NORMALIZATION.replace('class_based_', '')}.csv"
        normalized_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    normalize_all_chips()