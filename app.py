from flask import Flask, render_template, request
import os
import sys

# Ensure src is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict import predict_electricity_usage

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read appliance quantities (counts) and daily hours
        fans_count = float(request.form.get("fans_count", 0))
        fans_hours = float(request.form.get("fans_hours", 0))

        fridge_count = float(request.form.get("refrigerator_count", 0))
        fridge_hours = float(request.form.get("refrigerator_hours", 0))

        ac_count = float(request.form.get("ac_count", 0))
        ac_hours = float(request.form.get("ac_hours", 0))

        tv_count = float(request.form.get("tv_count", 0))
        tv_hours = float(request.form.get("tv_hours", 0))

        monitor_count = float(request.form.get("monitor_count", 0))
        monitor_hours = float(request.form.get("monitor_hours", 0))

        pump_count = float(request.form.get("pump_count", 0))
        pump_hours = float(request.form.get("pump_hours", 0))
    except (TypeError, ValueError):
        return render_template(
            "index.html",
            error="Please enter valid numbers for appliance counts and usage hours."
        ), 400

    # Total effective monthly load hours = Count * Daily Hours * 30 days
    monthly_inputs = {
        "Fan": max(0.0, fans_count * fans_hours * 30),
        "Fans": max(0.0, fans_count * fans_hours * 30),
        "Refrigerator": max(0.0, fridge_count * fridge_hours * 30),
        "AirConditioner": max(0.0, ac_count * ac_hours * 30),
        "Television": max(0.0, tv_count * tv_hours * 30),
        "Monitor": max(0.0, monitor_count * monitor_hours * 30),
        "MotorPump": max(0.0, pump_count * pump_hours * 30)
    }

    try:
        results = predict_electricity_usage(monthly_inputs)
        return render_template(
            "result.html",
            units=results["predicted_units"],
            bill=results["predicted_bill"]
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Prediction error: {str(e)}"
        ), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
