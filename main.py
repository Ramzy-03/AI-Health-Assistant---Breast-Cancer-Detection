#!/usr/bin/env python3
"""
AI Health Assistant - Breast Cancer Detection
===========================================

This script starts the Flask web application for AI-powered breast cancer risk assessment.

Usage:
    python run.py

Or directly:
    python app.py

The application will be available at:
    http://localhost:5000

Features:
- Modern web interface with animations
- Real-time ML predictions
- REST API for integrations
- Mobile-responsive design

For API documentation, visit:
    http://localhost:5000/api/docs
"""

import os
import sys
from src.api.app import app

def main():
    print("🏥 AI Health Assistant - Breast Cancer Detection")
    print("=" * 50)
    print("🤖 Powered by Machine Learning")
    print("🌐 Flask Web Application")
    print("📊 Real-time Risk Assessment")
    print()

    # Check if model exists
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "model.pkl")
    if not os.path.exists(model_path):
        print("❌ Model not found! Please run 'python save_model.py' first.")
        sys.exit(1)

    print("✅ Model loaded successfully")
    print("🚀 Starting web server...")
    print()
    print("🌐 Web Interface: http://localhost:5000")
    print("📚 API Docs: http://localhost:5000/api/docs")
    print("🛑 Press Ctrl+C to stop")
    print()

    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except KeyboardInterrupt:
        print()
        print("👋 Application stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()