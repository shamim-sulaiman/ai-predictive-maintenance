# web_app/web_app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

# STEP 0: Page config
st.set_page_config(page_title="ML Predictive Maintenance", layout="wide")
st.markdown("<hr style='margin-top: 2em;'>", unsafe_allow_html=True)
st.markdown("© 2025 Shamim Sulaiman. All rights reserved. For demonstration and educational purposes only.", unsafe_allow_html=True)

# STEP 1: Title + Intro
st.title("🔧 ML Predictive Maintenance - Pump Monitoring Dashboard")

with st.expander("📘 About this system", expanded=True):
    st.markdown("""
    This is an **AI-driven predictive maintenance platform** tailored for pump systems with PLC/SCADA integration. It leverages machine learning models trained on time-series sensor data to:

    - Perform multi-output forecasting (e.g., predict the next 50 readings for a sensor)
    - Visualize historical and projected sensor behavior
    - Enable early detection of equipment degradation trends

    ---

    ### 📁 Input Requirements
    Upload or generate a **CSV file** representing pump sensor logs, containing columns such as:
    - `vib_rms`, `motor_temp`, `motor_current`, `discharge_psi`, `flow_m3ph`

    Data can be:
    - Extracted from a real-time OPC UA interface using `opc_logger.py`
    - Simulated through mock data generation

    ---

    ### ⚙️ Forecasting Engine
    This dashboard uses a **MultiOutputRegressor** trained on sliding windows of past readings (default = 5 steps) to predict the next 50 future values.

    Forecasting models are tag-specific and can be:
    - Pretrained (based on simulated data)
    - Custom trained using user-uploaded datasets

    ---

    ### ✅ Recommended Use
    - Integrate with OPC UA to pull real sensor readings
    - Visualize current state and projected behavior
    - Train custom models for each pump installation

    For advanced use, this tool can be extended with:
    - Anomaly detection
    - Automated alerts (rule-based or AI-assisted)
    - Integration with SCADA dashboards
    - Forecast the next sensor value for vibration, temperature, flow, etc.
    - Detect early signs of failure from trends — without needing labeled data
    """)

# STEP 2: File uploader section
st.markdown("---")
st.header("📤 Upload or Simulate Pump Sensor Data")

use_demo = st.checkbox("✅ Use demo OPC UA data (opc_live_data.csv)", value=False)

uploaded_file = None
if use_demo:
    st.success("Using demo data from data/opc_live_data.csv")
    uploaded_file = "data/opc_live_data.csv"
else:
    uploaded_file = st.file_uploader("📤 Upload your pump sensor CSV", type="csv", key="manual_upload")

# Only proceed if file is uploaded
if uploaded_file:
    if isinstance(uploaded_file, str):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Sensor Trend Visualization")
    st.line_chart(df[['vib_rms', 'motor_temp', 'motor_current', 'discharge_psi', 'flow_m3ph']])

    # STEP 3: Train model section
    st.markdown("---")
    st.header("🧪 Train Custom Forecast Model (Optional)")
    st.markdown("Use your uploaded data to train a custom forecasting model.")

    if st.button("Train My Forecast Model"):
        with st.spinner("Training your custom forecast models..."):
            os.system("python model/train_multi_forecast_model.py model/custom")
        st.success("✅ Custom models trained and saved!")

    # STEP 4: Choose model version
    st.markdown("---")
    st.header("📦 Select Forecast Model")
    model_type = st.selectbox("Choose which model to use for forecasting:", ["Pretrained (Shamim's)", "Custom (My Trained Model)"])
    model_folder = "model/default" if model_type.startswith("Pre") else "model/custom"

    # STEP 5: Forecast next 50 sensor values
    st.markdown("---")
    st.header("🔮 Forecast Next 50 Sensor Values")

    selected_tag = st.selectbox("Select parameter to forecast", ['vib_rms', 'motor_temp', 'motor_current', 'discharge_psi', 'flow_m3ph'])
    past_steps = 5  # must match model training
    future_steps = 50

    if selected_tag not in df.columns:
        st.warning(f"'{selected_tag}' column not found in uploaded file.")
    else:
        try:
            model = joblib.load(f"{model_folder}/multi_forecast_{selected_tag}.pkl")

            series = df[selected_tag].values
            last_sequence = series[-past_steps:]  # use last 5 known values as input
            input_sequence = np.array(last_sequence).reshape(1, -1)

            y_future = model.predict(input_sequence)[0]

            future_df = pd.DataFrame({
                'Forecast Step': [f"t+{i+1}" for i in range(future_steps)],
                'Predicted Value': y_future
            })

            st.markdown(f"📉 **Predicted Next 50 Values for `{selected_tag}`**")
            st.line_chart(future_df.set_index("Forecast Step"))

        except FileNotFoundError:
            st.error(f"Model for '{selected_tag}' not found in {model_folder}. Please train it first.")
