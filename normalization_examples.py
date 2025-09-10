#!/usr/bin/env python3
"""
Example configurations for different normalization methods in test.py
Copy and paste the desired configuration into test.py lines 43-46
"""

print("=== Normalization Method Examples ===\n")

examples = {
    "Class-4 Min-Max Normalization (Current)": {
        'CURRENT_NORMALIZATION': 'class_based_minmax',
        'CURRENT_STATS_SOURCE': 'compute',
        'files_generated': ['minmax_normalized_test_*.png', 'confusion_matrix_classifier_minmax_normalized_test.*']
    },
    
    "Class-4 Mean/Std Normalization (Original)": {
        'CURRENT_NORMALIZATION': 'class_based',
        'CURRENT_STATS_SOURCE': 'compute', 
        'files_generated': ['normalized_test_*.png', 'confusion_matrix_classifier_normalized_test.*']
    },
    
    "Standard Z-Score Normalization": {
        'CURRENT_NORMALIZATION': 'standard',
        'CURRENT_STATS_SOURCE': 'compute',
        'files_generated': ['standard_normalized_test_*.png', 'confusion_matrix_classifier_standard_normalized_test.*']
    },
    
    "Sklearn Min-Max Normalization": {
        'CURRENT_NORMALIZATION': 'minmax', 
        'CURRENT_STATS_SOURCE': 'compute',
        'files_generated': ['sklearn_minmax_normalized_test_*.png', 'confusion_matrix_classifier_sklearn_minmax_normalized_test.*']
    },
    
    "No Normalization (Raw Data)": {
        'CURRENT_NORMALIZATION': 'none',
        'CURRENT_STATS_SOURCE': 'compute',
        'files_generated': ['raw_test_*.png', 'confusion_matrix_classifier_raw_test.*']
    },
    
    "Load from JSON Stats (Min-Max)": {
        'CURRENT_NORMALIZATION': 'class_based_minmax',
        'CURRENT_STATS_SOURCE': 'json',
        'CURRENT_STATS_PATH': 'data/fts_mzi_dataset/minmax_normalization_statistics.json',
        'files_generated': ['minmax_normalized_test_*.png', 'confusion_matrix_classifier_minmax_normalized_test.*']
    }
}

for name, config in examples.items():
    print(f"🔧 {name}:")
    print(f"   CURRENT_NORMALIZATION = '{config['CURRENT_NORMALIZATION']}'")
    print(f"   CURRENT_STATS_SOURCE = '{config['CURRENT_STATS_SOURCE']}'")
    if 'CURRENT_STATS_PATH' in config:
        print(f"   CURRENT_STATS_PATH = '{config['CURRENT_STATS_PATH']}'")
    print(f"   📁 Files: {', '.join(config['files_generated'])}")
    print()

print("=== Usage Instructions ===")
print("📋 For Training:")
print("1. Change CURRENT_NORMALIZATION in train_autoencoder.py (line 40)")
print("2. Change CURRENT_NORMALIZATION in train_classifier.py (line 41)")  
print("3. Run: python train_autoencoder.py && python train_classifier.py")
print()
print("📋 For Testing:")
print("1. Copy the desired configuration variables")
print("2. Paste them into test.py (lines 43-46)")
print("3. Run: python test.py")
print()
print("✅ All plots and outputs will be automatically named based on the method")
print("🔄 Model files will automatically match between training and testing")

print("\n=== Complete Pipeline Example ===")
print("# For Min-Max Normalization:")
print("1. Set 'class_based_minmax' in both training scripts")
print("2. Run: python apply_minmax_normalization.py  # Prepare data")
print("3. Run: python train_autoencoder.py           # Train denoiser")
print("4. Run: python train_classifier.py            # Train classifier")  
print("5. Set 'class_based_minmax' in test.py and run it  # Test models")
print()
print("📁 Generated files will include:")
print("   • minmax_normalized_*.png (plots)")
print("   • *_minmax_normalized_*.pth (models)")
print("   • *_minmax_normalized_*.csv (reports)")