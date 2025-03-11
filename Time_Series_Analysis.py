import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# comparing date with Amount for find out seasonal and trends
file=("rupa_107_machine_learning_project.csv")
df = pd.read_csv(file)

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the date as index
df.set_index("Date", inplace=True)

# Ensure data is sorted by date
df.sort_index(inplace=True)

# Decompose the time series
result = seasonal_decompose(df['Amount'], model="additive", period=7)

# Plot decomposition
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(df['Amount'], label="Original")
plt.legend(loc="upper left")

plt.subplot(4, 1, 2)
plt.plot(result.trend, label="Trend", color='green')
plt.legend(loc="upper left")

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label="Seasonality", color='red')
plt.legend(loc="upper left")

plt.subplot(4, 1, 4)
plt.plot(result.resid, label="Residuals", color='purple')
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()


# comparing date with appoinment ID for find out seasonal and trends
file=("rupa_107_machine_learning_project.csv")
df = pd.read_csv(file)

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the date as index
df.set_index("Date", inplace=True)

# Ensure data is sorted by date
df.sort_index(inplace=True)

# Decompose the time series
result = seasonal_decompose(df['AppointmentID'], model="additive", period=7)  # Change period as needed

# Plot decomposition
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(df['AppointmentID'], label="Original")
plt.legend(loc="upper left")

plt.subplot(4, 1, 2)
plt.plot(result.trend, label="Trend", color='yellow')
plt.legend(loc="upper left")

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label="Seasonality", color='purple')
plt.legend(loc="upper left")

plt.subplot(4, 1, 4)
plt.plot(result.resid, label="Residuals", color='black')
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()

# comparing date with PatientID  for find out seasonal and trends
file=("rupa_107_machine_learning_project.csv")
df = pd.read_csv(file)

# Convert date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the date as index (important for time series analysis)
df.set_index("Date", inplace=True)

# Ensure data is sorted by date
df.sort_index(inplace=True)

# Decompose the time series
result = seasonal_decompose(df['PatientID'], model="additive", period=7)

# Plot decomposition
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(df['PatientID'], label="Original")
plt.legend(loc="upper left")

plt.subplot(4, 1, 2)
plt.plot(result.trend, label="Trend", color='brown')
plt.legend(loc="upper left")

plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label="Seasonality", color='seagreen')
plt.legend(loc="upper left")

plt.subplot(4, 1, 4)
plt.plot(result.resid, label="Residuals", color='orange')
plt.legend(loc="upper left")

plt.tight_layout()
plt.show()
