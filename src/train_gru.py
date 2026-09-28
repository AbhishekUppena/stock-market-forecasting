import os
import matplotlib.pyplot as plt

from preprocessing import (
    load_stock_data,
    prepare_close_price,
    create_sequences
)

from gru_model import create_gru_model

from prediction import (
    make_prediction,
    calculate_mae,
    calculate_rmse
)


DATA_PATH = "../data/GOOG-year.csv"
TIME_STEPS = 60


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = load_stock_data(DATA_PATH)

print("Dataset loaded successfully.")
print("Number of rows:", len(df))
print(df.head())


# --------------------------------------------------
# 2. Train/Test split
# --------------------------------------------------

raw_train_size = int(len(df) * 0.8)

print("\nRaw training rows:", raw_train_size)
print("Raw testing rows:", len(df) - raw_train_size)


# --------------------------------------------------
# 3. Scale data
# --------------------------------------------------

scaled_prices, scaler = prepare_close_price(
    df,
    train_size=raw_train_size
)

print("\nData preprocessing completed.")


# --------------------------------------------------
# 4. Create sequences
# --------------------------------------------------

X, y = create_sequences(
    scaled_prices,
    time_steps=TIME_STEPS
)

print("X shape:", X.shape)
print("y shape:", y.shape)


# --------------------------------------------------
# 5. Split sequences
# --------------------------------------------------

split_index = raw_train_size - TIME_STEPS

X_train = X[:split_index]
y_train = y[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 6. Create GRU model
# --------------------------------------------------

model = create_gru_model(
    input_shape=(X_train.shape[1], X_train.shape[2])
)

model.summary()


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# --------------------------------------------------
# 8. Make predictions
# --------------------------------------------------

predictions = make_prediction(
    model,
    X_test,
    scaler
)

actual_prices = scaler.inverse_transform(y_test)


# --------------------------------------------------
# 9. Evaluate model
# --------------------------------------------------

mae = calculate_mae(
    actual_prices,
    predictions
)

rmse = calculate_rmse(
    actual_prices,
    predictions
)


print("\n------------------------------")
print("GRU Model Evaluation")
print("------------------------------")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")


# --------------------------------------------------
# 10. Save GRU results
# --------------------------------------------------

os.makedirs("../results", exist_ok=True)


# Actual vs Predicted

plt.figure(figsize=(12, 6))

plt.plot(
    actual_prices,
    label="Actual Price"
)

plt.plot(
    predictions,
    label="GRU Predicted Price"
)

plt.title("GRU Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()

plt.savefig(
    "../results/gru_actual_vs_predicted.png"
)

plt.show()


# Training Loss

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("GRU Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    "../results/gru_loss_curve.png"
)

plt.show()