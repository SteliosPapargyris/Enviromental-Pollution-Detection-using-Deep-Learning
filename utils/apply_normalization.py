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
    plot_robust_normalized_mean_feature_per_class,
    plot_normalized_train_mean_feature_per_class
)
from utils.config import *

def get_normalization_config(normalization_method):
    """Get configuration for the specified normalization method"""
    method_configs = {
        'class_based_mean_std': {
            'func': compute_mean_class_4_then_subtract,
            'plot_func': plot_normalized_train_mean_feature_per_class,
            'output_dir': Path("data/out/normalized_meanstd"),
            'output_file': "MeanStd_Normalized_FTS-MZI_Matrix.csv",
            'stats_file': "meanstd_normalization_statistics.json",
            'plot_file': "meanstd_normalized_train_mean_feature_per_class.png",
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

    return method_configs[normalization_method]

def apply_normalization(datasets, normalization_method=None):
    """Apply normalization to a list of datasets and return normalized datasets

    Args:
        datasets: List of DataFrames to normalize
        normalization_method: Normalization method to use (default from config)

    Returns:
        List of normalized DataFrames
    """
    
    method_config = get_normalization_config(normalization_method)

    config = {
        "output_dir": method_config['output_dir'],
        "plots_dir": method_config['output_dir'],
        "normalized_dir": method_config['output_dir'],
        "class_column": "Class",
        "chip_column": "Chip",
        "target_class": target_class
    }

    normalized_datasets = []

    for data in datasets:
        config["output_dir"].mkdir(parents=True, exist_ok=True)
        config["plots_dir"].mkdir(parents=True, exist_ok=True)
        config["normalized_dir"].mkdir(parents=True, exist_ok=True)

        train_peak_columns = [col for col in data.columns if col.startswith("train_Peak")]
        match_peak_columns = [col for col in data.columns if col.startswith("match_Peak")]

        if train_peak_columns or match_peak_columns:
            normalized_data = data.copy()
            normalization_func = method_config['func']

            if train_peak_columns:
                train_peaks_1_to_32 = []
                for col in train_peak_columns:
                    peak_num = int(col.split('_')[2]) if col.count('_') >= 2 else int(col.replace('train_Peak', ''))
                    if 1 <= peak_num <= 32:
                        train_peaks_1_to_32.append(col)

                train_peak_stats_path = str(config["output_dir"] / f"train_peak_{method_config['stats_file']}")
                normalized_data, _, _ = normalization_func(
                    normalized_data,
                    class_column="train_Class",
                    chip_column="train_Chip",
                    columns_to_normalize=train_peaks_1_to_32,
                    target_class=config["target_class"],
                    save_stats_json=train_peak_stats_path
                )

            if match_peak_columns:
                match_peaks_1_to_32 = []
                for col in match_peak_columns:
                    peak_num = int(col.split('_')[2]) if col.count('_') >= 2 else int(col.replace('match_Peak', ''))
                    if 1 <= peak_num <= 32:
                        match_peaks_1_to_32.append(col)

                match_peak_stats_path = str(config["output_dir"] / f"match_peak_{method_config['stats_file']}")
                normalized_data, _, _ = normalization_func(
                    normalized_data,
                    class_column="match_Class",
                    chip_column="match_Chip",
                    columns_to_normalize=match_peaks_1_to_32,
                    target_class=config["target_class"],
                    save_stats_json=match_peak_stats_path
                )

        else:
            peak_columns = [col for col in data.columns if col.startswith("Peak")]

            peaks_1_to_32 = []
            for col in peak_columns:
                peak_num = int(col.split('_')[1]) if '_' in col else int(col.replace('Peak', ''))
                if 1 <= peak_num <= 32:
                    peaks_1_to_32.append(col)

            columns_to_normalize = peaks_1_to_32
            normalization_func = method_config['func']
            stats_path = str(config["output_dir"] / method_config['stats_file'])

            normalized_data, _, _ = normalization_func(
                data,
                class_column=config["class_column"],
                chip_column=config["chip_column"],
                columns_to_normalize=columns_to_normalize,
                target_class=config["target_class"],
                save_stats_json=stats_path
            )

        normalized_datasets.append(normalized_data)

    return normalized_datasets