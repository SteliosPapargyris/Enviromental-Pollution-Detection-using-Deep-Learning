import pandas as pd
from pathlib import Path
from utils.normalization_techniques import (
    compute_mean_class_4_then_subtract,
    compute_minmax_class_4_then_normalize,
    compute_robust_class_4_then_normalize
)
from utils.plot_utils import (
    plot_raw_mean_feature_per_class,
    plot_minmax_normalized_mean_feature_per_class,
    plot_robust_normalized_mean_feature_per_class
)
from utils.config import *

def get_normalization_config(normalization_method):
    """Get configuration for the specified normalization method"""
    method_configs = {
        'class_based_mean_std': {
            'func': compute_mean_class_4_then_subtract,
            'plot_func': None,
            'output_dir': Path("data/out/normalized"),
            'output_file': "Normalized_FTS-MZI_Matrix.csv",
            'stats_file': "normalization_statistics.json",
            'plot_file': "normalized_train_mean_feature_per_class.png",
            'plot_title': 'Mean/Std Normalized Mean Feature per Class (Training Data)',
            'method_name': 'mean/std normalization'
        },
        'class_based_minmax': {
            'func': compute_minmax_class_4_then_normalize,
            'plot_func': plot_minmax_normalized_mean_feature_per_class,
            'output_dir': Path("data/out/normalized_minmax"),
            'output_file': "MinMax_Normalized_FTS-MZI_Matrix.csv",
            'stats_file': "minmax_normalization_statistics.json",
            'plot_file': "minmax_normalized_train_mean_feature_per_class.png",
            'plot_title': 'Min-Max Normalized Mean Feature per Class (Training Data)',
            'method_name': 'min-max normalization'
        },
        'class_based_robust': {
            'func': compute_robust_class_4_then_normalize,
            'plot_func': plot_robust_normalized_mean_feature_per_class,
            'output_dir': Path("data/out/normalized_robust"),
            'output_file': "Robust_Normalized_FTS-MZI_Matrix.csv",
            'stats_file': "robust_normalization_statistics.json",
            'plot_file': "robust_normalized_train_mean_feature_per_class.png",
            'plot_title': 'Robust Scaled Mean Feature per Class (Training Data)',
            'method_name': 'robust scaling normalization'
        }
    }

    if normalization_method not in method_configs:
        available_methods = list(method_configs.keys())
        raise ValueError(f"Unsupported normalization method: {normalization_method}. Available: {available_methods}")

    return method_configs[normalization_method]

def apply_normalization(normalization_method=None):
    """Apply normalization based on the specified method or config setting"""

    # Use provided method or fall back to config
    if normalization_method is None:
        normalization_method = CURRENT_NORMALIZATION

    # Get method configuration
    method_config = get_normalization_config(normalization_method)

    # Setup base configuration
    config = {
        "chip_exclude": chip_exclude,
        "file_path": "data/out/transformed_simulated_data.csv",
        "output_dir": Path("data/fts_mzi_dataset"),
        "plots_dir": Path("out"),
        "normalized_dir": method_config['output_dir'],
        "class_column": "Class",
        "chip_column": "Chip",
        "target_class": target_class
    }

    try:
        print(f"Applying {method_config['method_name']} (method: {normalization_method})...")
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

        # Apply the selected normalization method
        print(f"Applying {method_config['method_name']}...")
        normalization_func = method_config['func']
        stats_path = str(config["output_dir"] / method_config['stats_file'])

        normalized_data, stat1, stat2 = normalization_func(
            data,
            chip_exclude=config["chip_exclude"],
            class_column=config["class_column"],
            chip_column=config["chip_column"],
            columns_to_normalize=columns_to_normalize,
            target_class=config["target_class"],
            save_stats_json=stats_path
        )

        # Plot normalized data (if plot function is available)
        if method_config['plot_func'] is not None:
            method_config['plot_func'](
                normalized_data,
                class_column=config["class_column"],
                save_path=str(config["plots_dir"] / method_config['plot_file']),
                title=method_config['plot_title']
            )

        # Save normalized dataset
        output_path = config["output_dir"] / method_config['output_file']
        normalized_data.to_csv(output_path, index=False)
        print(f"{method_config['method_name'].capitalize()} dataset saved to {output_path}")

        # Split normalized data by chip and save to appropriate directory
        print(f"\nSplitting {method_config['method_name']} dataset by chip...")
        unique_chips = sorted(normalized_data[config["chip_column"]].unique())

        for chip in unique_chips:
            # Filter data for this chip
            chip_data = normalized_data[normalized_data[config["chip_column"]] == chip]

            # Save to normalized directory
            chip_output_path = config["normalized_dir"] / f"{chip}.csv"
            chip_data.to_csv(chip_output_path, index=False)

            print(f"  Chip {chip}: {len(chip_data)} samples → {chip_output_path}")

        print(f"\n✅ {method_config['method_name'].capitalize()} completed successfully!")
        print(f"📁 Normalized chip files saved in: {config['normalized_dir']}")
        print(f"📊 Total chips processed: {len(unique_chips)}")

        # Display normalization statistics summary based on method
        print(f"\n📈 Normalization Statistics Summary:")
        print(f"   • Method: {method_config['method_name']} based on Class {config['target_class']}")
        print(f"   • Features normalized: {len(columns_to_normalize)}")

        if normalization_method == 'class_based':
            print(f"   • Mean values range: [{stat1.min():.4f}, {stat1.max():.4f}]")
            print(f"   • Std values range: [{stat2.min():.4f}, {stat2.max():.4f}]")
        elif normalization_method == 'class_based_minmax':
            print(f"   • Min values range: [{stat1.min():.4f}, {stat1.max():.4f}]")
            print(f"   • Max values range: [{stat2.min():.4f}, {stat2.max():.4f}]")
        elif normalization_method == 'class_based_robust':
            print(f"   • Median values range: [{stat1.min():.4f}, {stat1.max():.4f}]")
            print(f"   • MAD values range: [{stat2.min():.4f}, {stat2.max():.4f}]")
            print(f"   • Robust scaling is less sensitive to outliers than standard scaling")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Run with the normalization method specified in config.py
    print(f"Using normalization method from config: {CURRENT_NORMALIZATION}")
    apply_normalization()