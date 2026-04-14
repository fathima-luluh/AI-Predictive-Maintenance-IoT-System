import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n = 2000

# Simulated IoT sensor data
temperature = np.random.normal(60, 15, n)   # °C
vibration = np.random.normal(5, 2, n)       # vibration level
current = np.random.normal(10, 3, n)        # current in mA

# Failure logic (realistic industrial rule simulation)
failure = []

for t, v, c in zip(temperature, vibration, current):
    if t > 80 or v > 8 or c > 15:
        failure.append(1)  # FAILURE
    else:
        failure.append(0)  # NORMAL

# Create DataFrame
df = pd.DataFrame({
    "temperature": temperature,
    "vibration": vibration,
    "current": current,
    "failure": failure
})

# Save dataset
df.to_csv("data/iot_sensor_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())