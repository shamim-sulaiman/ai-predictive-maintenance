# train_forecast_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

# --- Settings ---
DATA_PATH = "data/opc_live_data.csv"
SAVE_DIR = sys.argv[1] if len(sys.argv) > 1 else "model/default"
PAST_STEPS = 5  # Number of previous steps to use as features
TARGETS = ['vib_rms', 'motor_temp', 'motor_current', 'discharge_psi', 'flow_m3ph']

os.makedirs(SAVE_DIR, exist_ok=True)

def create_sliding_window_features(series, past_steps):
    X, y = [], []
    for i in range(past_steps, len(series)):
        X.append(series[i-past_steps:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# --- Load and clean dataset ---
df = pd.read_csv("data/opc_live_data.csv")

print(f"\n📊 Training forecasting models using last {PAST_STEPS} steps...\n")

# --- Train model for each target ---
for target in TARGETS:
    series = df[target].values

    # Build features and labels
    X, y = create_sliding_window_features(series, PAST_STEPS)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"✅ {target}: RMSE = {rmse:.4f}")

    # Save model
    joblib.dump(model, f"{SAVE_DIR}/forecast_{target}.pkl")

print("\n✅ All models saved to 'model/' folder.")
