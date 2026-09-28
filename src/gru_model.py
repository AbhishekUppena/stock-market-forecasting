import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout


def create_gru_model(input_shape):
    """
    Create a GRU model for stock price forecasting.
    """

    model = Sequential([
        tf.keras.Input(shape=input_shape),

        GRU(
            64,
            return_sequences=True
        ),

        Dropout(0.2),

        GRU(32),

        Dropout(0.2),

        Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mean_squared_error"
    )

    return model