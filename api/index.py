from flask import Flask, request, render_template
import os
import pickle
import numpy as np
import pandas as pd

# Load the trained model, preprocessor, and scaler
model, preprocessor, scaler = pickle.load(open("models/best_model.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__, template_folder="../templates")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = None

    if request.method == "POST":
        try:
            # Read all 13 inputs from the form
            input_values = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["cp"]),
                float(request.form["trestbps"]),
                float(request.form["chol"]),
                float(request.form["fbs"]),
                float(request.form["restecg"]),
                float(request.form["thalach"]),
                float(request.form["exang"]),
                float(request.form["oldpeak"]),
                float(request.form["slope"]),
                float(request.form["ca"]),
                float(request.form["thal"])
            ]

            # Define expected column names
            columns = [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal"
            ]

            # Wrap the input values into a DataFrame
            input_df = pd.DataFrame([input_values], columns=columns)

            # Apply preprocessing and scaling
            input_encoded = preprocessor.transform(input_df)
            input_scaled = scaler.transform(input_encoded)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_text = (
                "✅ Likely to have heart disease." if prediction == 1
                else "🟢 Unlikely to have heart disease."
            )

        except Exception as e:
            prediction_text = f"❌ Error: {str(e)}"

    return render_template("index.html", prediction_text=prediction_text)

# Vercel requires the handler name
handler = app