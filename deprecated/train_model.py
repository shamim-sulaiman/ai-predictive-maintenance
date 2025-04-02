# model/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
import sys

# Allow custom path from Streamlit
SAVE_DIR = sys.argv[1] if len(sys.argv) > 1 else "model/default"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load data
df = pd.read_csv("data/opc_live_data.csv")
X = df.drop(columns=['label'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, f"{SAVE_DIR}/pump_model.pkl")
print("✅ Pump health model saved to", SAVE_DIR)
print("\n📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))
