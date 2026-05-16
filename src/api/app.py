from flask import Flask, request, jsonify, render_template, send_file
import pickle
import numpy as np
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import io

app = Flask(__name__)

# Load the model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    print("Please run save_model.py first to train and save the model")
    exit(1)

model = pickle.load(open(model_path, "rb"))
print("Model loaded successfully!")

@app.route("/")
def home():
    """Serve the main web interface"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests from both API and web interface"""
    try:
        data = request.json["features"]
        data = np.array(data).reshape(1, -1)
        prediction = model.predict(data)
        probability = model.predict_proba(data)[0][1]

        return jsonify({
            "prediction": int(prediction[0]),
            "probability": float(probability),
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 400

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route("/api/docs")
def api_docs():
    """Simple API documentation"""
    return render_template("api_docs.html")

@app.route("/mlflow")
def mlflow_dashboard():
    """MLflow dashboard showing experiments and runs"""
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Breast_Cancer_Classification")
    if experiment is None:
        return "Experiment not found. Please run train_model.py first."
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    run_data = []
    for run in runs:
        run_info = {
            'run_id': run.info.run_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'metrics': run.data.metrics,
            'params': run.data.params,
            'artifacts': []
        }
        # Get artifacts
        artifacts = client.list_artifacts(run.info.run_id)
        for artifact in artifacts:
            if artifact.is_dir:
                continue
            run_info['artifacts'].append(artifact.path)
        run_data.append(run_info)
    
    return render_template("mlflow.html", runs=run_data)

@app.route("/artifact/<run_id>/<filename>")
def artifact(run_id, filename):
    """Serve MLflow artifacts"""
    client = MlflowClient()
    try:
        artifact_path = client.download_artifacts(run_id, filename)
        return send_file(artifact_path, mimetype='image/png')
    except Exception as e:
        return str(e), 404

if __name__ == "__main__":
    print("🚀 Starting AI Health Assistant...")
    print("📊 Model loaded and ready for predictions")
    print("🌐 Web interface available at http://localhost:5000")
    print("📚 API documentation at http://localhost:5000/api/docs")
    app.run(host="0.0.0.0", port=5000, debug=True)