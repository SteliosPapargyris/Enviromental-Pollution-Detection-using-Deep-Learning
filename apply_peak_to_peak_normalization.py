import pandas as pd
from pathlib import Path
from utils.data_utils import compute_peak_to_peak_class_4_then_normalize
from utils.plot_utils import plot_raw_mean_feature_per_class, plot_peak_to_peak_normalized_mean_feature_per_class
from utils.config import *

config = {
    "chip_exclude": chip_exclude,
    "file_path": "data/out/transformed_simulated_data.csv",
    "output_dir": Path("data/fts_mzi_dataset"),
    "plots_dir": Path("out"),
    "normalized_dir": Path("data/out/normalized_peak_to_peak"),
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
    
    # Plot raw data (if not already done)
    plot_raw_mean_feature_per_class(
        data,
        class_column=config["class_column"],
        save_path=str(config["plots_dir"] / "raw_mean_feature_per_class.png"),
        title='Raw Mean Feature per Class'
    )
    
    # Find columns to normalize (Peak columns + Temperature)
    peak_columns = [col for col in data.columns if col.startswith("Peak")]
    temp_columns = [col for col in data.columns if "Temperature" in col]
    columns_to_normalize = peak_columns + temp_columns
    print(f"Found {len(peak_columns)} Peak columns and {len(temp_columns)} Temperature columns")
    print(f"Total feature columns to normalize: {len(columns_to_normalize)}")
    
    # Apply peak-to-peak normalization
    print("Applying peak-to-peak normalization...")
    normalized_data, range_stats = compute_peak_to_peak_class_4_then_normalize(
        data,
        chip_exclude=config["chip_exclude"],
        class_column=config["class_column"],
        chip_column=config["chip_column"],
        columns_to_normalize=columns_to_normalize,
        target_class=config["target_class"],
        save_stats_json=str(config["output_dir"] / "peak_to_peak_normalization_statistics.json")
    )

    # Plot normalized data
    plot_peak_to_peak_normalized_mean_feature_per_class(
        normalized_data,
        class_column=config["class_column"],
        save_path=str(config["plots_dir"] / "peak_to_peak_normalized_train_mean_feature_per_class.png"),
        title='Peak-to-Peak Normalized Mean Feature per Class (Training Data)'
    )

    # Save normalized dataset
    output_path = config["output_dir"] / "Peak_to_Peak_Normalized_FTS-MZI_Matrix.csv"
    normalized_data.to_csv(output_path, index=False)
    print(f"Peak-to-peak normalized dataset saved to {output_path}")
    
    # Split normalized data by chip and save to data/out/normalized_peak_to_peak/
    print("\nSplitting peak-to-peak normalized dataset by chip...")
    unique_chips = sorted(normalized_data[config["chip_column"]].unique())
    
    for chip in unique_chips:
        # Filter data for this chip
        chip_data = normalized_data[normalized_data[config["chip_column"]] == chip]
        
        # Save to data/out/normalized_peak_to_peak/{chip}.csv
        chip_output_path = config["normalized_dir"] / f"{chip}.csv"
        chip_data.to_csv(chip_output_path, index=False)
        
        print(f"  Chip {chip}: {len(chip_data)} samples → {chip_output_path}")
    
    print(f"\n✅ Peak-to-peak normalization completed successfully!")
    print(f"📁 Peak-to-peak normalized chip files saved in: {config['normalized_dir']}")
    print(f"📊 Total chips processed: {len(unique_chips)}")
    
    # Display normalization statistics summary
    print(f"\n📈 Normalization Statistics Summary:")
    print(f"   • Method: Peak-to-peak scaling based on Class {config['target_class']} range")
    print(f"   • Features normalized: {len(columns_to_normalize)}")
    print(f"   • Range values: [{range_stats.min():.4f}, {range_stats.max():.4f}]")
    print(f"   • Peak-to-peak normalization preserves spectral shape while normalizing amplitude")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    raise