# api/index.py

from flask import Flask, request, render_template
import os
import pickle
import numpy as np

# Load model
model, scaler = pickle.load(open("models/best_model.pkl", "rb"))

app = Flask(__name__, template_folder="../templates")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None
    if request.method == "POST":
        try:
            # Example fields; extend as per your dataset
            age = float(request.form["age"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])

            data = np.array([[age, trestbps, chol]])
            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)[0]

            prediction_text = "Likely to have heart disease." if prediction == 1 else "Unlikely to have heart disease."
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template("index.html", prediction_text=prediction_text)

# Vercel expects this name
handler = app