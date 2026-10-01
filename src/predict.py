"""
Inference module for Electricity Usage & Bill Predictor.
Loads the trained model and predicts:
- Units Consumed (kWh)
- Estimated Monthly Bill (₹)
"""

import os
import warnings
import joblib
import pandas as pd
import numpy as np

# Suppress unpickling version warnings
warnings.filterwarnings("ignore")

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "model.pkl"
)

FEATURE_COLUMNS = [
    "Fan",
    "Refrigerator",
    "AirConditioner",
    "Television",
    "Monitor",
    "MotorPump"
]


def load_trained_model(model_path: str = DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model file not found at: '{model_path}'.\n"
        )
    return joblib.load(model_path)


def resolve_feature_value(inputs: dict, target_feature: str) -> float:
    """
    Intelligently extracts the numeric value for a feature from inputs,
    handling plural/singular variations (e.g., Fan vs Fans), case differences,
    and appliance aliases.
    """
    # 1. Exact match
    if target_feature in inputs:
        return float(inputs[target_feature])

    t_lower = target_feature.lower()

    # 2. Check for singular/plural/case-insensitive match
    for k, v in inputs.items():
        k_lower = str(k).lower()
        if k_lower == t_lower or k_lower == t_lower + "s" or k_lower + "s" == t_lower:
            return float(v)

    # 3. Keyword/alias fallback
    for k, v in inputs.items():
        k_lower = str(k).lower()
        if "fan" in t_lower and "fan" in k_lower:
            return float(v)
        if ("refrigerat" in t_lower or "fridge" in t_lower) and ("refrigerat" in k_lower or "fridge" in k_lower):
            return float(v)
        if ("air" in t_lower or "ac" in t_lower) and ("air" in k_lower or "ac" in k_lower):
            return float(v)
        if ("tele" in t_lower or "tv" in t_lower) and ("tele" in k_lower or "tv" in k_lower):
            return float(v)
        if ("monitor" in t_lower or "pc" in t_lower) and ("monitor" in k_lower or "pc" in k_lower):
            return float(v)
        if ("pump" in t_lower or "motor" in t_lower) and ("pump" in k_lower or "motor" in k_lower):
            return float(v)

    return 0.0


def predict_electricity_usage(appliance_hours: dict, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    model_artifact = load_trained_model(model_path)

    model_units = model_artifact["model_units"]
    model_bill = model_artifact["model_bill"]
    features = model_artifact.get("feature_cols", FEATURE_COLUMNS)

    # Resolve each feature in the exact order expected by the model
    input_row = {feat: [resolve_feature_value(appliance_hours, feat)] for feat in features}
    df_input = pd.DataFrame(input_row)[features]

    # Zero-appliance guard
    if sum(df_input.iloc[0].values) == 0:
        return {
            "predicted_units": 0.0,
            "predicted_bill": 50.00
        }

    # Predict using the trained models
    pred_units = float(model_units.predict(df_input)[0])
    pred_bill = float(model_bill.predict(df_input)[0])

    # Ensure non-negative predictions
    pred_units = max(0.0, round(pred_units, 1))
    pred_bill = max(0.0, round(pred_bill, 2))

    return {
        "predicted_units": pred_units,
        "predicted_bill": pred_bill
    }
