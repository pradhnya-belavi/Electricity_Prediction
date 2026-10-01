import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "electricity_bill_dataset-selected-columns.csv")
df = pd.read_csv(data_path)

features = ['Fan', 'Refrigerator', 'AirConditioner', 'Television', 'Monitor', 'MotorPump']
X = df[features]
y_units = df['Units']
y_bill = df['MonthlyBill']

# Train Linear Regression models
model_units = LinearRegression().fit(X, y_units)
model_bill = LinearRegression().fit(X, y_bill)

print(f"Units Model R2: {model_units.score(X, y_units):.4f}")
print(f"Bill Model R2:  {model_bill.score(X, y_bill):.4f}")

artifact = {
    'model_units': model_units,
    'model_bill': model_bill,
    'feature_cols': features,
    'coefficients_units': dict(zip(features, model_units.coef_.round(4))),
    'coefficients_bill': dict(zip(features, model_bill.coef_.round(4)))
}

models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, "model.pkl")

joblib.dump(artifact, model_path)
print(f"Saved robust model to {model_path}")
