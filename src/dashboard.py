import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/iot_sensor_data.csv")

print("Dataset loaded successfully!")
print(data.head())

# Create figure
plt.figure(figsize=(10,5))

plt.plot(data["temperature"].values, label="Temperature")
plt.plot(data["vibration"].values, label="Vibration")
plt.plot(data["current"].values, label="Current")

plt.title("IoT Predictive Maintenance Dashboard")
plt.legend()
plt.grid(True)

# IMPORTANT: show graph
plt.show()