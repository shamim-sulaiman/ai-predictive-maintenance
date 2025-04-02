# 🧠 AI Predictive Maintenance Dashboard

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

## 💻 Getting Started

1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run web_app/web_app.py`
3. Upload a sensor CSV or generate one with `opc_logger.py`
4. Optionally train a custom model
5. Forecast and visualize the next 50 steps
