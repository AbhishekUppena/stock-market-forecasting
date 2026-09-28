# Stock Market Forecasting

A deep learning project for forecasting stock prices using LSTM and GRU neural networks.

## Overview

This project uses historical stock price data to train recurrent neural network models for stock price forecasting.

Two deep learning architectures are implemented:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

The models use historical closing prices and a sliding-window approach to learn temporal patterns.

## Project Structure

```text
stock-market-forecasting/
│
├── data/
│   └── GOOG-year.csv
│
├── notebooks/
│
├── results/
│   ├── actual_vs_predicted.png
│   ├── loss_curve.png
│   ├── gru_actual_vs_predicted.png
│   ├── gru_loss_curve.png
│   └── model_comparison.csv
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── gru_model.py
│   ├── prediction.py
│   ├── train.py
│   └── train_gru.py
│
├── web/
│
├── README.md
├── requirements.txt
└── .gitignore