"""
Automated Data Preprocessing for Iris Dataset
Kriteria 1 - Skilled Level

Author: David kristian silalahi
Date: 2025-11-28
Description: Script otomatis untuk preprocessing dataset Iris berdasarkan hasil eksperimen
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_raw_data():
    """
    Load iris dataset dari scikit-learn
    
    Returns:
        tuple: (X_features, y_target, feature_names, target_names)
    """
    logger.info("Loading Iris dataset from scikit-learn...")
    
    iris_data = load_iris()
    X = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    y = iris_data.target
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, iris_data.feature_names, iris_data.target_names


def perform_eda(X, y):
    """
    Perform basic EDA checks
    
    Args:
        X (DataFrame): Features
        y (array): Target
        
    Returns:
        dict: EDA results
    """
    logger.info("Performing EDA checks...")
    
    eda_results = {
        'missing_values': X.isnull().sum().sum(),
        'duplicate_rows': X.duplicated().sum(),
        'feature_stats': X.describe(),
        'target_distribution': np.bincount(y),
        'data_types': X.dtypes
    }
    
    logger.info(f"Missing values: {eda_results['missing_values']}")
    logger.info(f"Duplicate rows: {eda_results['duplicate_rows']}")
    logger.info(f"Target distribution: {eda_results['target_distribution']}")
    
    return eda_results


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocess data dengan train-test split dan scaling
    
    Args:
        X (DataFrame): Features
        y (array): Target
        test_size (float): Size of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    logger.info("Starting data preprocessing...")
    
    # Train-test split dengan stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train-test split completed:")
    logger.info(f"  Train set: {X_train.shape[0]} samples")
    logger.info(f"  Test set: {X_test.shape[0]} samples")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    logger.info("Feature scaling completed using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Save processed data ke files
    
    Args:
        X_train, X_test (DataFrame): Scaled features
        y_train, y_test (array): Targets
        output_dir (str): Output directory path
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processed data to {output_dir}...")
    
    # Save training data
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    
    # Save target data
    pd.Series(y_train).to_csv(f"{output_dir}/y_train.csv", index=False, header=['target'])
    pd.Series(y_test).to_csv(f"{output_dir}/y_test.csv", index=False, header=['target'])
    
    logger.info("✅ Processed data saved successfully!")
    logger.info("Files created:")
    logger.info("  - X_train.csv")
    logger.info("  - X_test.csv")
    logger.info("  - y_train.csv")
    logger.info("  - y_test.csv")


def validate_preprocessing(X_train, X_test, y_train, y_test):
    """
    Validate preprocessing results
    
    Args:
        X_train, X_test (DataFrame): Processed features
        y_train, y_test (array): Target data
    """
    logger.info("Validating preprocessing results...")
    
    # Check shapes
    assert X_train.shape[1] == X_test.shape[1], "Feature dimensions mismatch!"
    assert len(y_train) == X_train.shape[0], "Train feature-target length mismatch!"
    assert len(y_test) == X_test.shape[0], "Test feature-target length mismatch!"
    
    # Check scaling (features should have mean~0, std~1 for train set)
    train_means = X_train.mean().values
    train_stds = X_train.std().values
    
    assert np.allclose(train_means, 0, atol=1e-8), "Features not properly centered!"
    assert np.allclose(train_stds, 1, atol=1e-1), "Features not properly scaled!"
    
    logger.info("✅ Preprocessing validation passed!")
    logger.info(f"Final data shapes:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    logger.info(f"  y_train: {len(y_train)}")
    logger.info(f"  y_test: {len(y_test)}")


def create_metadata_file(output_dir, eda_results, feature_names, target_names):
    """
    Create metadata file dengan informasi preprocessing
    
    Args:
        output_dir (str): Output directory
        eda_results (dict): EDA results
        feature_names (list): Feature names
        target_names (list): Target class names
    """
    metadata = {
        'preprocessing_info': {
            'dataset': 'Iris',
            'preprocessing_method': 'StandardScaler',
            'train_test_split': '80-20',
            'stratified': True,
            'random_state': 42
        },
        'dataset_info': {
            'features': list(feature_names),
            'target_classes': list(target_names),
            'missing_values': int(eda_results['missing_values']),
            'duplicate_rows': int(eda_results['duplicate_rows'])
        },
        'files_created': [
            'X_train.csv',
            'X_test.csv', 
            'y_train.csv',
            'y_test.csv'
        ]
    }
    
    # Save metadata
    import json
    with open(f"{output_dir}/preprocessing_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info("📄 Metadata file created: preprocessing_metadata.json")


def main():
    """
    Main function untuk menjalankan automated preprocessing
    """
    print("="*60)
    print("AUTOMATED DATA PREPROCESSING - IRIS DATASET")
    print("="*60)
    
    try:
        # 1. Load raw data
        X, y, feature_names, target_names = load_raw_data()
        
        # 2. Perform EDA
        eda_results = perform_eda(X, y)
        
        # 3. Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        
        # 4. Validate preprocessing
        validate_preprocessing(X_train, X_test, y_train, y_test)
        
        # 5. Save processed data
        output_dir = "namadataset_preprocessing"
        save_processed_data(X_train, X_test, y_train, y_test, output_dir)
        
        # 6. Create metadata
        create_metadata_file(output_dir, eda_results, feature_names, target_names)
        
        print("="*60)
        print("✅ AUTOMATED PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'output_dir': output_dir
        }
        
    except Exception as e:
        logger.error(f"❌ Error during preprocessing: {str(e)}")
        raise e


if __name__ == "__main__":
    # Jalankan automated preprocessing
    results = main()
    
    # Optional: Print summary
    print(f"\\n📊 Preprocessing Summary:")
    print(f"   Training samples: {results['X_train'].shape[0]}")
    print(f"   Test samples: {results['X_test'].shape[0]}")
    print(f"   Features: {results['X_train'].shape[1]}")
    print(f"   Output directory: {results['output_dir']}")