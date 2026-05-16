# 🏥 AI Health Assistant - Breast Cancer Detection

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blue.svg)](https://mlflow.org/)

An end-to-end Machine Learning web application for real-time breast cancer risk assessment. This project combines advanced predictive modeling with a modern, user-friendly interface to provide instant diagnostic insights.

---

## 🚀 Key Features

- **High-Accuracy Prediction**: Utilizes a Random Forest Classifier trained on the Wisconsin Breast Cancer dataset.
- **Modern Web Interface**: Clean, responsive UI with interactive animations for seamless user experience.
- **Experiment Tracking**: Integrated with **MLflow** to track model versions, hyperparameters, and performance metrics.
- **RESTful API**: Scalable API endpoints for integration with other health systems.
- **Comprehensive Visualization**: Generates ROC curves, Precision-Recall curves, and Confusion Matrices for model transparency.
- **Real-time Diagnostics**: Get instant probability scores and classification results.

---

## 📂 Project Architecture

The project follows a modular, real-world organization for scalability and maintainability:

```text
ai-health-assistant/
├── data/                   # Dataset storage
├── models/                 # Saved model artifacts (.pkl)
├── notebooks/              # Research and exploration notebooks
├── reports/
│   └── figures/            # Generated performance plots
├── src/                    # Main source code
│   ├── api/                # Web Application Layer
│   │   ├── static/         # CSS, JS, and Images
│   │   ├── templates/      # HTML Templates
│   │   └── app.py          # Flask Application Server
│   ├── core/               # Core Logic Layer
│   │   └── data_processor.py # Preprocessing & Scaling
│   └── scripts/            # Automation Scripts
│       ├── build_production_model.py # Production training
│       └── train_experiments.py      # MLflow experiment tracking
├── tests/                  # Unit and integration tests
├── .gitignore              # Git exclusion rules
├── main.py                 # Application entry point
└── requirements.txt        # Project dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ai-health-assistant.git
cd ai-health-assistant
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚦 Usage

### 1. Training the Model

**To run experiments with MLflow tracking:**
```bash
python src/scripts/train_experiments.py
```
This will log parameters, metrics, and plots to the `mlruns` directory.

**To build the production-ready model:**
```bash
python src/scripts/build_production_model.py
```
This saves the final `model.pkl` to the `models/` folder.

### 2. Running the Web Application
```bash
python main.py
```
The application will be available at: `http://localhost:5000`

---

## 📊 MLflow Integration

Track your model's evolution by viewing the MLflow UI:
1. Run `mlflow ui` in your terminal.
2. Navigate to `http://localhost:5000/mlflow` within the app to see a built-in dashboard of recent runs.

---

## 📚 API Documentation

The AI Health Assistant provides a RESTful API for remote predictions.

### **Endpoint: `/predict`**
- **Method**: `POST`
- **Payload**:
```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, ...]
}
```
- **Response**:
```json
{
  "prediction": 1,
  "probability": 0.98,
  "status": "success"
}
```

Visit `http://localhost:5000/api/docs` for full documentation.

---

## 🧠 Model Performance

Our current Random Forest model achieves:
- **Accuracy**: ~97%
- **ROC AUC**: ~0.99
- **F1-Score**: ~0.97

*Visualizations like Confusion Matrix and ROC Curves can be found in the `reports/figures/` directory.*

---

## 💻 Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, NumPy, Pandas
- **Experiment Tracking**: MLflow
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Disclaimer**: *This tool is for educational purposes and should not be used as a replacement for professional medical advice.*
"# AI-Health-Assistant---Breast-Cancer-Detection" 
