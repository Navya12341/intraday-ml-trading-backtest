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
