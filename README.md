# ⚡ Electricity Usage & Bill Prediction System

An end-to-end Machine Learning web application built with **Flask**, **scikit-learn**, and clean **HTML/CSS** to predict monthly electricity consumption (**kWh Units**) and estimated electricity bills (**₹**) based on household appliance quantities and daily operational hours.

---

## 🌟 Key Features

* **Appliance-Level Granularity**: Allows users to specify both appliance counts and daily hours for:
  * Ceiling / Table Fans
  * Refrigerator
  * Air Conditioner (1.5 Ton)
  * Television
  * Computer / Monitor
  * Water Motor Pump
* **Physics-Grounded Regression Model**: Trained on validated domestic electricity consumption data ($R^2 > 0.998$).
* **Clean & Simple UI**: Built with pure HTML/CSS cards and responsive inputs.
* **Instant Prediction**: Outputs estimated **Units (kWh)** and **Monthly Bill (₹)**.

---

## 📁 Repository Structure

```
Electricity_Prediction_System/
├── colab file/
│   └── electricity_usage_model_training.ipynb  # Google Colab notebook for training
├── data/
│   └── electricity_bill_dataset-selected-columns.csv # Clean training dataset
├── models/
│   └── model.pkl                               # Serialized trained model bundle
├── scripts/
│   └── save_robust_model.py                    # Model training script
├── src/
│   ├── __init__.py
│   └── predict.py                              # Model inference engine
├── templates/
│   ├── index.html                              # Web input form
│   └── result.html                             # Prediction result display
├── app.py                                      # Flask application server
├── requirements.txt                            # Dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Megharajvsaka/Electricity_Prediction_System.git
cd Electricity_Prediction_System
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```
