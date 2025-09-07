import pandas as pd
from pathlib import Path
from utils.data_utils import compute_mean_class_4_then_subtract
from utils.plot_utils import plot_raw_mean_feature_per_class
from utils.config import *

config = {
    "chip_exclude": chip_exclude,
    "file_path": "data/out/transformed_simulated_data.csv",
    "output_dir": Path("data/fts_mzi_dataset"),
    "plots_dir": Path("out"),
    "normalized_dir": Path("data/out/normalized"),
    "class_column": "Class",
    "chip_column": "Chip", 
    "target_class": target_class
}

try:
    print("Loading dataset...")
    data = pd.read_csv(config["file_path"])
    
    # Validate required columns
    required_cols = [config["class_column"], config["chip_column"]]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create directories
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config["plots_dir"].mkdir(parents=True, exist_ok=True)
    config["normalized_dir"].mkdir(parents=True, exist_ok=True)
    
    # Plot raw data
    plot_raw_mean_feature_per_class(
        data,
        class_column=config["class_column"],
        save_path=str(config["plots_dir"] / "raw_mean_feature_per_class.png"),
        title='Raw Mean Feature per Class'
    )
    
    # Find columns to normalize
    columns_to_normalize = [col for col in data.columns if col.startswith("Peak")]
    print(f"Found {len(columns_to_normalize)} feature columns")
    
    # Normalize dataset (simplified call)
    print("Applying normalization...")
    normalized_data, mean_stats, std_stats = compute_mean_class_4_then_subtract(
        data,
        chip_exclude=config["chip_exclude"],
        class_column=config["class_column"],
        chip_column=config["chip_column"],
        columns_to_normalize=columns_to_normalize,
        target_class=config["target_class"],
        save_stats_json=str(config["output_dir"] / "normalization_statistics.json")
    )

    # Save normalized dataset
    output_path = config["output_dir"] / "Normalized_FTS-MZI_Matrix.csv"
    normalized_data.to_csv(output_path, index=False)
    print(f"Normalized dataset saved to {output_path}")
    
    # Split normalized data by chip and save to data/out/normalized/
    print("\nSplitting normalized dataset by chip...")
    unique_chips = sorted(normalized_data[config["chip_column"]].unique())
    
    for chip in unique_chips:
        # Filter data for this chip
        chip_data = normalized_data[normalized_data[config["chip_column"]] == chip]
        
        # Save to data/out/normalized/{chip}.csv
        chip_output_path = config["normalized_dir"] / f"{chip}.csv"
        chip_data.to_csv(chip_output_path, index=False)
        
        print(f"  Chip {chip}: {len(chip_data)} samples → {chip_output_path}")
    
    print(f"\n✅ Processing completed successfully!")
    print(f"📁 Normalized chip files saved in: {config['normalized_dir']}")
    print(f"📊 Total chips processed: {len(unique_chips)}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    raise