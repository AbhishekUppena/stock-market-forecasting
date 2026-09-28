import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_stock_data(file_path):
    """
    Load stock data from a CSV file.
    """
    df = pd.read_csv(file_path)
    df = df.dropna()
    return df


def prepare_close_price(df, train_size):
    """
    Extract the Close price and scale it.

    The scaler is fitted only on the training data
    to avoid data leakage.
    """
    close_prices = df[["Close"]].values

    train_prices = close_prices[:train_size]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_prices)

    scaled_prices = scaler.transform(close_prices)

    return scaled_prices, scaler


def create_sequences(data, time_steps=60):
    """
    Create input sequences and target values
    for LSTM training.
    """
    X = []
    y = []

    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i])

    return np.array(X), np.array(y)