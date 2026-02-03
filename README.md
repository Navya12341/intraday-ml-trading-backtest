# Intraday ML Trading System with Backtesting

An end-to-end intraday trading research pipeline for equities using machine learning and realistic backtesting.

This project demonstrates a full quantitative workflow:

data ingestion → feature engineering → supervised learning → signal generation → execution simulation → transaction costs → performance evaluation.

The system is intentionally kept simple and transparent for interview and learning purposes.

---

## Project Overview

The goal is to predict short-term price direction using recent intraday market information and evaluate whether the predictions can be converted into a viable trading strategy.

The pipeline uses:

- minute-level OHLCV data
- logistic regression as a baseline model
- probability-based trading signals
- realistic execution and transaction costs
- detailed diagnostics and visualisation

---

## Data Source

Intraday price data is fetched using the Twelve Data API.

The project is written so that the data source can easily be replaced by broker or proprietary feeds.

---

## Prediction Target

Binary classification:

y[t] = 1  if (close[t+5] − close[t]) / close[t] > 0
y[t] = 0  otherwise

The model predicts whether the next 5-minute return is positive.

---

## Features

Only past information is used.

- 1-minute return
- 3-minute return
- 5-minute return
- rolling mean of 1-minute returns (10-minute window)
- rolling standard deviation of 1-minute returns (10-minute window)
- rolling volume z-score (10-minute window)

---

## Models

- Logistic Regression (baseline)
- XGBoost or LightGBM (if available)

---

## Train / Test Protocol

- Strict chronological split  
  - first 70% → training  
  - last 30% → testing
- No shuffling
- No look-ahead bias

---

## Trading Strategy

- Predict probability of positive future return
- Go long when probability exceeds a chosen threshold
- No short selling
- One open position at a time
- Fixed holding period of 5 minutes

---

## Execution and Costs

- Entry price: close at time t
- Exit price: close at time t + 5
- Fixed round-trip transaction cost of 0.05% per trade

---

## Evaluation Metrics

- cumulative return (equity curve)
- Sharpe ratio
- maximum drawdown
- win rate

Additional diagnostics include:

- probability distribution of model outputs
- signal frequency for different thresholds

---

## Example Output

Equity curve and probability distribution plots are generated automatically and saved in the `results/` folder.

---

## How to run

1. Install dependencies
  pip install -r requirements.txt
2. Edit the API key in `main.py`
   TWELVE_DATA_API_KEY = “YOUR_API_KEY”
3. Run
   python main.py
---

## Notes

This project is intended for research and educational purposes.

It focuses on correctness of the quantitative pipeline and evaluation methodology rather than on producing profitable strategies.
