import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
import os


def load_covertype(save_path="data/covertype.npy", save_labels_path="data/labels.npy"):
    """
    Load and preprocess the Covertype dataset.

    The Covertype dataset contains 581,012 samples and 54 features
    describing forest cover type from cartographic variables. Features
    include elevation, slope, distances to water and roads, hillshade
    indices, and binary wilderness/soil type indicators.

    Preprocessing steps:
        - Features are standardized to zero mean and unit variance
          using StandardScaler to ensure equal contribution of all
          features to distance computations in K-Means.
        - Labels are preserved separately for optional quality evaluation.

    Args:
        save_path (str): Output path for preprocessed feature array.
        save_labels_path (str): Output path for label array.

    Returns:
        tuple:
            - X (numpy.ndarray): Standardized feature matrix of shape (581012, 54).
            - y (numpy.ndarray): Class labels of shape (581012,).
    """
    print("Loading Covertype dataset...")
    data = fetch_covtype()
    X, y = data.data, data.target

    print(f"  Samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")

    # standardize features — K-Means uses Euclidean distance so scale matters
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    os.makedirs("data", exist_ok=True)
    np.save(save_path, X)
    np.save(save_labels_path, y)

    print(f"  Dataset saved to {save_path}")
    return X, y


if __name__ == "__main__":
    X, y = load_covertype()
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Label array shape: {y.shape}")
    print(f"Feature mean (should be ~0): {X.mean():.6f}")
    print(f"Feature std  (should be ~1): {X.std():.6f}")