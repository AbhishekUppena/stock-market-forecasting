import numpy as np


def make_prediction(model, data, scaler):
    """
    Generate predictions using the trained LSTM model.
    """

    predictions = model.predict(data, verbose=0)

    # Convert normalized values back to original stock prices
    predictions = scaler.inverse_transform(predictions)

    return predictions


def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error.
    """

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    return rmse


def calculate_mae(actual, predicted):
    """
    Calculate Mean Absolute Error.
    """

    mae = np.mean(np.abs(actual - predicted))

    return mae