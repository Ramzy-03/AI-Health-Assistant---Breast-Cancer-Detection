import pickle
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.ensemble import RandomForestClassifier
from src.core.data_processor import preprocess

def main():
    print("🤖 Training AI Health Assistant Model...")
    print("📊 Loading and preprocessing data...")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess()

    print(f"✅ Data loaded successfully!")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")

    # Train model
    print("🎯 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    model.fit(X_train, y_train)
    print("✅ Model trained successfully!")

    # Evaluate on test set
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"📈 Training accuracy: {train_score:.3f}")
    print(f"📈 Test accuracy: {test_score:.3f}")

    # Save model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"💾 Model saved to {model_path}")
    print("🚀 AI Health Assistant is ready!")
    print("🌐 Run 'python app.py' to start the web application")

if __name__ == "__main__":
    main()