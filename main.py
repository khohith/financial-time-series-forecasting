# main.py

# =========================
# 1. IMPORT LIBRARIES
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from scipy.signal import stft
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# =========================
# 2. FETCH DATA
# =========================
stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

data_frames = []

for stock in stocks:
    df = yf.download(stock, start="2018-01-01", end="2024-01-01")
    df['Stock'] = stock
    data_frames.append(df[['Close']])

# Combine data
data = pd.concat(data_frames, axis=1)
data.columns = stocks

print("Data Loaded:")
print(data.head())

# =========================
# 3. HANDLE MISSING VALUES
# =========================
data = data.fillna(method='ffill')

# =========================
# 4. NORMALIZATION
# =========================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# =========================
# 5. PLOT TIME SERIES
# =========================
plt.figure()
for i, stock in enumerate(stocks):
    plt.plot(data.index, data[stock], label=stock)

plt.title("Stock Price Time Series")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# =========================
# 6. APPLY STFT (for each stock)
# =========================
spectrograms = []

for i in range(scaled_data.shape[1]):
    signal = scaled_data[:, i]

    f, t, Zxx = stft(signal, nperseg=64)

    spectrogram = np.abs(Zxx)
    spectrograms.append(spectrogram)

    # Plot spectrogram
    plt.figure()
    plt.pcolormesh(t, f, spectrogram)
    plt.title(f"Spectrogram - {stocks[i]}")
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.colorbar()
    plt.show()

# =========================
# 7. PREPARE DATA FOR CNN
# =========================
X = []
y = []

for spec in spectrograms:
    X.append(spec)

    # Target: last value of original signal
    y.append(spec.mean())

X = np.array(X)
y = np.array(y)

# Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 8. BUILD CNN MODEL
# =========================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.summary()

# =========================
# 9. TRAIN MODEL
# =========================
model.fit(X, y, epochs=20, verbose=1)

# =========================
# 10. PREDICTION
# =========================
predictions = model.predict(X)

print("\nPredictions:")
print(predictions)

# =========================
# 11. EVALUATION
# =========================
mse = mean_squared_error(y, predictions)
print("\nMean Squared Error:", mse)

# =========================
# 12. COMPARE ACTUAL VS PREDICTED
# =========================
plt.figure()
plt.plot(y, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title("Actual vs Predicted")
plt.legend()
plt.show()
