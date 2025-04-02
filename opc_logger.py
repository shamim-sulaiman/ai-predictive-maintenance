# opc_logger.py

import pandas as pd
from datetime import datetime
import time
import os
import numpy as np

# Uncomment the block below to connect to a real OPC UA server
"""
from opcua import Client

# Replace this with your actual OPC UA server URL
client = Client("opc.tcp://192.168.0.100:4840")

try:
    client.connect()
    print("✅ Connected to OPC UA server")

    # Replace these Node IDs with your actual PLC tags
    nodes = {
        "vib_rms": client.get_node("ns=2;i=2"),
        "motor_temp": client.get_node("ns=2;i=3"),
        "motor_current": client.get_node("ns=2;i=4"),
        "discharge_psi": client.get_node("ns=2;i=5"),
        "flow_m3ph": client.get_node("ns=2;i=6")
    }

    def read_opc_sensor_data():
        return {
            "timestamp": datetime.now().isoformat(),
            "vib_rms": nodes["vib_rms"].get_value(),
            "motor_temp": nodes["motor_temp"].get_value(),
            "motor_current": nodes["motor_current"].get_value(),
            "discharge_psi": nodes["discharge_psi"].get_value(),
            "flow_m3ph": nodes["flow_m3ph"].get_value()
        }

except Exception as e:
    print(f"❌ OPC UA connection error: {e}")
    client.disconnect()
    exit()
"""

# Simulated fallback (used by default)
def read_mock_sensor_data(i):
    return {
        "timestamp": datetime.now().isoformat(),
        "vib_rms": np.random.normal(0.6 + 0.01*i, 0.05),
        "motor_temp": np.random.normal(50 + 0.02*i, 1.0),
        "motor_current": np.random.normal(11 + 0.01*i, 0.5),
        "discharge_psi": np.random.normal(60 - 0.03*i, 2.0),
        "flow_m3ph": np.random.normal(40 - 0.02*i, 1.5)
    }

print("🔧 Logging pump sensor data (simulated or real)...")

data = []
for i in range(100):
    # If using real OPC, switch the function:
    # row = read_opc_sensor_data()
    row = read_mock_sensor_data(i)
    data.append(row)
    print(f"[{i+1}/100] Logged: {row}")
    time.sleep(1)

df = pd.DataFrame(data)
os.makedirs("data", exist_ok=True)
df.to_csv("data/opc_live_data.csv", index=False)

print("✅ Data saved to data/opc_live_data.csv")

# Uncomment if using real OPC UA
# client.disconnect()
# print("🔌 OPC UA connection closed.")
