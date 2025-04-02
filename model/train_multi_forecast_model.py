# model/train_multi_forecast_model.py

import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
import sys
# Settings
DATA_PATH = "data/opc_live_data.csv"
SAVE_DIR = sys.argv[1] if len(sys.argv) > 1 else "model"
PAST_STEPS = 5
FUTURE_STEPS = 50
TARGETS = ['vib_rms', 'motor_temp', 'motor_current', 'discharge_psi', 'flow_m3ph']

os.makedirs(SAVE_DIR, exist_ok=True)

# Utility to create sliding window for multi-step output
def create_multistep_dataset(series, past_steps, future_steps):
    X, Y = [], []
    for i in range(past_steps, len(series) - future_steps):
        X.append(series[i - past_steps:i])
        Y.append(series[i:i + future_steps])
    return np.array(X), np.array(Y)

# Load data
print("📥 Loading sensor data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Train model for each sensor tag
print(f"\n🚀 Training multi-step forecast model ({FUTURE_STEPS} steps ahead) using last {PAST_STEPS} readings...\n")

for tag in TARGETS:
    if tag not in df.columns:
        print(f"❌ Column '{tag}' not found in data. Skipping.")
        continue

    series = df[tag].values
    X, Y = create_multistep_dataset(series, PAST_STEPS, FUTURE_STEPS)

    if len(X) == 0:
        print(f"⚠️ Not enough data for {tag}. Skipping.")
        continue

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))

    model_path = os.path.join(SAVE_DIR, f"multi_forecast_{tag}.pkl")
    joblib.dump(model, model_path)

    print(f"✅ Trained & saved: {tag:15s} | RMSE: {rmse:.4f} | Path: {model_path}")

print("\n✅ All multi-step models saved successfully.")
