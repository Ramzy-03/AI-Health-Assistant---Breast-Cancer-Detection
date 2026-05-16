import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data():
    """
    Load the breast cancer dataset from sklearn
    Returns:
        X: DataFrame with features
        y: Series with target labels
    """
    print("   Loading breast cancer dataset...")
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    print(f"   Dataset shape: {X.shape}")
    print(f"   Features: {len(data.feature_names)}")
    print(f"   Classes: {data.target_names}")

    return X, y

def preprocess(test_size=0.2, random_state=42):
    """
    Preprocess the breast cancer data for model training

    Args:
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        X_train, X_test: Training and test features (scaled)
        y_train, y_test: Training and test labels
        scaler: Fitted StandardScaler for future use
    """
    # Load data
    X, y = load_data()

    # Split data
    print(f"   Splitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class balance
    )

    # Scale features
    print("   Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("✅ Preprocessing completed!")
    print(f"   Training set: {X_train_scaled.shape}")
    print(f"   Test set: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    print("🧪 Testing data preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess()
    print(f"✅ Test successful! Data shapes: {X_train.shape}, {X_test.shape}")