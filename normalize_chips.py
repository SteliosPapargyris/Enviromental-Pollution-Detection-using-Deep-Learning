import pandas as pd
from pathlib import Path
from utils.apply_normalization import apply_normalization
from utils.config import *

def normalize_all_chips():
    """Normalize all chip CSV files using the current normalization method"""

    chip_files = []
    data_dir = Path(f"data/out/{total_num_chips}chips")

    # Load all numbered chip files (1.csv through total_num_chips + 1.csv)
    for chip_num in range(1, total_num_chips + 1):
        chip_file = data_dir / f"{chip_num}.csv"
        if chip_file.exists():
            df = pd.read_csv(chip_file)
            chip_files.append(df)

    if not chip_files:
        return

    # Apply normalization to all datasets
    normalized_datasets = apply_normalization(chip_files, CURRENT_NORMALIZATION)

    # Save normalized datasets with chip count folder structure based on normalization method
    method_mapping = {
        'class_based_mean_std': 'mean_std',
        'class_based_minmax': 'minmax',
        'class_based_robust': 'robust'
    }

    method_suffix = method_mapping.get(CURRENT_NORMALIZATION, 'unknown')
    output_dir = Path(f"data/out/{method_suffix}/{total_num_chips}chips")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, normalized_df in enumerate(normalized_datasets, 1):
        output_file = output_dir / f"chip_{i}_{method_suffix}.csv"
        normalized_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    normalize_all_chips()