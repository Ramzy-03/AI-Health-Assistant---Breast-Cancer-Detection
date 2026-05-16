import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from src.core.data_processor import preprocess
import time

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)

X_train, X_test, y_train, y_test, scaler = preprocess()

mlflow.set_experiment("Breast_Cancer_Classification")

def evaluate_and_log(model, name, params):
    with mlflow.start_run():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_test, probs)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
        mlflow.log_metric("training_time", training_time)

        roc_path = os.path.join(REPORTS_DIR, "roc_curve.png")
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()

        pr_path = os.path.join(REPORTS_DIR, "pr_curve.png")
        plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(pr_path)
        mlflow.log_artifact(pr_path)
        plt.close()

        cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        mlflow.sklearn.log_model(model, name)

models = [
    (LogisticRegression(max_iter=1000), "LogisticRegression", {"model": "LogisticRegression", "max_iter": 1000}),
    (RandomForestClassifier(n_estimators=100, max_depth=5), "RandomForest", {"model": "RandomForest", "n_estimators": 100, "max_depth": 5})
]

for model, name, params in models:
    evaluate_and_log(model, name, params)