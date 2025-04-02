# 🧠 ML Predictive Maintenance Dashboard

This is an AI-powered predictive maintenance dashboard for industrial pump systems. It enables engineers to:

- 📈 Visualize real-time or uploaded sensor data
- 🔮 Forecast the next 50 values for vibration, temperature, and more
- 🛠 Train custom ML models using user-provided data
- 💡 Integrate with OPC UA data sources (real PLC/SCADA systems)

## ⚙️ Tech Stack

- Python + Streamlit
- scikit-learn (MultiOutputRegressor)
- Pandas, NumPy, Matplotlib
- OPC UA integration via `opc_logger.py` (optional)

## 📁 Project Structure

```
├── model/                        # Forecast models (.pkl files)
├── data/                         # Sensor data (.csv)
├── web_app/                      # Streamlit app
│   └── web_app.py
├── model/train_multi_forecast_model.py
├── opc_logger.py
├── LICENSE
└── README.md
```

## 📁 How to Use

1. [Visit the app link](https://ml-predictive-maintenance.streamlit.app/)
2. Upload your CSV (or check \"Use demo OPC UA data\")
3. Preview your data and plot it
4. (Optional) Click \"Train My Forecast Model\" to build your own model
5. View 50-step future predictions for any sensor tag
