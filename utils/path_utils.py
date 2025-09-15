def get_paths_for_normalization(norm_method):
    """Get base_path and stats_path for the specified normalization method"""
    path_configs = {
        'class_based_mean_std': {
            'base_path': 'data/out/normalized',
            'stats_file': 'normalization_statistics.json'
        },
        'class_based_minmax': {
            'base_path': 'data/out/normalized_minmax',
            'stats_file': 'minmax_normalization_statistics.json'
        },
        'class_based_robust': {
            'base_path': 'data/out/normalized_robust',
            'stats_file': 'robust_normalization_statistics.json'
        }
    }
    config = path_configs.get(norm_method, path_configs['class_based_robust'])
    return config['base_path'], f"data/fts_mzi_dataset/{config['stats_file']}"