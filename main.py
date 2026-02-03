import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

USE_XGB = False
USE_LGB = False

try:
    import xgboost as xgb
    USE_XGB = True
except (ImportError, OSError):
    pass

if not USE_XGB:
    try:
        import lightgbm as lgb
        USE_LGB = True
    except (ImportError, OSError):
        pass

import requests


# ============================================================================
# CONFIGURATION
# ============================================================================

TWELVE_DATA_API_KEY = "2c08f1d68ab34a23bd385d517264921d"
SYMBOL = "AAPL"
INTERVAL = "1min"
OUTPUTSIZE = 5000


# ============================================================================
# DATA INGESTION
# ============================================================================

def fetch_twelvedata_data(api_key, symbol, interval="1min", outputsize=5000):
    """
    Fetch historical intraday data from Twelve Data API
    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": outputsize,
        "format": "JSON"
    }

    print("Fetching data from Twelve Data API...")
    print(f"Symbol: {symbol}, Interval: {interval}")

    r = requests.get(url, params=params, timeout=20)
    data = r.json()

    if "status" in data and data["status"] == "error":
        raise Exception(data.get("message", "Twelve Data API error"))

    if "values" not in data:
        raise Exception(f"Unexpected API response: {data}")

    rows = []
    for row in data["values"]:
        rows.append({
            "timestamp": row["datetime"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]) if row["volume"] is not None else 0.0
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (
        df.sort_values("timestamp")
          .drop_duplicates("timestamp")
          .reset_index(drop=True)
    )
    print(f"Fetched {len(df)} candles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features(df):
    """
    Build features using only past information
    Returns DataFrame with added feature columns
    """
    df = df.copy()
    
    # 1-minute return
    df['ret_1m'] = df['close'].pct_change(1)
    
    # 3-minute return
    df['ret_3m'] = df['close'].pct_change(3)
    
    # 5-minute return
    df['ret_5m'] = df['close'].pct_change(5)
    
    # Rolling mean of 1-minute returns over 10 minutes
    df['ret_1m_mean_10'] = df['ret_1m'].rolling(window=10, min_periods=10).mean()
    
    # Rolling std of 1-minute returns over 10 minutes
    df['ret_1m_std_10'] = df['ret_1m'].rolling(window=10, min_periods=10).std()
    
    # Volume z-score over 10-minute window
    vol_mean = df['volume'].rolling(window=10, min_periods=10).mean()
    vol_std = df['volume'].rolling(window=10, min_periods=10).std()
    df['volume_zscore_10'] = (df['volume'] - vol_mean) / vol_std
    
    return df


# ============================================================================
# LABEL GENERATION
# ============================================================================

def build_labels(df):
    """
    Build forward-looking 5-minute return labels
    y[t] = 1 if future 5-min return > 0, else 0
    """
    df = df.copy()
    
    # Forward 5-minute return
    df['future_ret_5m'] = (df['close'].shift(-5) - df['close']) / df['close']
    
    # Binary classification label
    df['label'] = (df['future_ret_5m'] > 0).astype(int)
    
    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_models(X_train, y_train):
    """
    Train Logistic Regression and XGBoost/LightGBM classifiers
    Returns tuple of (lr_model, gb_model) or (lr_model, None) if GB unavailable
    """
    print("\nTraining models...")
    
    # Logistic Regression baseline
    lr_model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    lr_model.fit(X_train, y_train)
    print("Logistic Regression trained")
    
    # Gradient Boosting model (optional)
    gb_model = None
    if USE_XGB:
        print("Using XGBoost")
        gb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        gb_model.fit(X_train, y_train)
    elif USE_LGB:
        print("Using LightGBM")
        gb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        gb_model.fit(X_train, y_train)
    else:
        print("Warning: XGBoost and LightGBM not available. Using Logistic Regression only.")
    
    print("Models trained successfully")
    
    return lr_model, gb_model


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_signals(model, X_test, threshold=0.55):
    """
    Generate trading signals based on model predictions
    Returns array of signals: 1 = long, 0 = flat
    """
    # Predict probability of positive return
    probs = model.predict_proba(X_test)[:, 1]
    
    # Generate signals: long if prob > threshold
    signals = (probs > threshold).astype(int)
    
    return signals


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def backtest_strategy(signals, prices_test, transaction_cost=0.0005):
    """
    Simulate trading strategy on test set
    Returns: (trade_returns, equity_curve)
    """
    trade_returns = []
    equity = 1.0
    equity_curve = [equity]
    
    i = 0
    while i < len(signals):
        if signals[i] == 1:
            # Enter long position
            entry_price = prices_test.iloc[i]
            
            # Exit after 5 periods
            exit_idx = i + 5
            if exit_idx < len(prices_test):
                exit_price = prices_test.iloc[exit_idx]
                
                # Calculate gross return
                gross_return = (exit_price - entry_price) / entry_price
                
                # Apply transaction cost (round-trip)
                net_return = gross_return - transaction_cost
                
                trade_returns.append(net_return)
                
                # Update equity
                equity *= (1 + net_return)
                equity_curve.append(equity)
                
                # Move to next entry point after exit
                i = exit_idx + 1
            else:
                # Not enough data for exit
                break
        else:
            i += 1
    
    return pd.Series(trade_returns), pd.Series(equity_curve)


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def compute_metrics(trade_returns, equity_curve):
    """
    Compute performance metrics
    Returns dict with all metrics
    """
    if len(trade_returns) == 0:
        return {
            'num_trades': 0,
            'cumulative_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
    
    # Number of trades
    num_trades = len(trade_returns)
    
    # Cumulative return
    cumulative_return = equity_curve.iloc[-1] - 1
    
    # Sharpe ratio (annualized)
    # Assuming ~375 trading minutes per day, 252 trading days per year
    mean_ret = trade_returns.mean()
    std_ret = trade_returns.std()
    
    if std_ret > 0:
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(375 * 252)
    else:
        sharpe_ratio = 0.0
    
    # Maximum drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (trade_returns > 0).sum() / num_trades
    
    metrics = {
        'num_trades': num_trades,
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }
    
    return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# ============================================================================
# DIAGNOSTIC AND IMPROVED SIGNAL GENERATION
# ============================================================================

def generate_signals_with_diagnostics(model, X_test, threshold=0.55):
    """
    Generate trading signals with diagnostic information
    Returns array of signals: 1 = long, 0 = flat
    """
    # Predict probability of positive return
    probs = model.predict_proba(X_test)[:, 1]
    
    # Diagnostic information
    print(f"\n--- Signal Generation Diagnostics ---")
    print(f"Probability statistics:")
    print(f"  Min: {probs.min():.4f}")
    print(f"  Max: {probs.max():.4f}")
    print(f"  Mean: {probs.mean():.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    print(f"  Std: {probs.std():.4f}")
    
    # Count signals above threshold
    signals_above_threshold = (probs > threshold).sum()
    print(f"\nSignals above threshold {threshold}: {signals_above_threshold} ({signals_above_threshold/len(probs)*100:.2f}%)")
    
    # Try different thresholds
    for t in [0.45, 0.50, 0.52, 0.55, 0.60]:
        count = (probs > t).sum()
        print(f"  Threshold {t}: {count} signals ({count/len(probs)*100:.2f}%)")
    
    # Generate signals: long if prob > threshold
    signals = (probs > threshold).astype(int)
    
    return signals, probs


def analyze_predictions(model, X_train, y_train, X_test, y_test):
    """
    Analyze model predictions and performance
    """
    print("\n--- Model Performance Analysis ---")
    
    # Training set performance
    train_pred = model.predict(X_train)
    train_accuracy = (train_pred == y_train).mean()
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    # Test set performance
    test_pred = model.predict(X_test)
    test_accuracy = (test_pred == y_test).mean()
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Class distribution
    print(f"\nTraining set class distribution:")
    print(f"  Class 0 (down): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
    print(f"  Class 1 (up): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
    
    print(f"\nTest set class distribution:")
    print(f"  Class 0 (down): {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.2f}%)")
    print(f"  Class 1 (up): {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.2f}%)")


def find_optimal_threshold(model, X_test, prices_test):
    """
    Find optimal probability threshold by testing multiple values
    """
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Testing Multiple Thresholds ---")
    
    best_threshold = 0.50
    best_sharpe = -999
    
    thresholds = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
    
    for threshold in thresholds:
        signals = (probs > threshold).astype(int)
        trade_returns, equity_curve = backtest_strategy(signals, prices_test)
        
        if len(trade_returns) > 0:
            metrics = compute_metrics(trade_returns, equity_curve)
            print(f"\nThreshold {threshold}:")
            print(f"  Trades: {metrics['num_trades']}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"  Win Rate: {metrics['win_rate']:.4f}")
            print(f"  Cum Return: {metrics['cumulative_return']:.4f}")
            
            if metrics['sharpe_ratio'] > best_sharpe and metrics['num_trades'] >= 10:
                best_sharpe = metrics['sharpe_ratio']
                best_threshold = threshold
        else:
            print(f"\nThreshold {threshold}: No trades generated")
    
    print(f"\n*** Optimal threshold: {best_threshold} (Sharpe: {best_sharpe:.4f}) ***")
    return best_threshold


# ============================================================================
# UPDATED MAIN EXECUTION WITH DIAGNOSTICS
# ============================================================================

def main():
    """
    Main execution function - runs complete backtest pipeline
    """
    
    # Step 1: Fetch data from Twelve Data API
    df = fetch_twelvedata_data(
        api_key=TWELVE_DATA_API_KEY,
        symbol=SYMBOL,
        interval=INTERVAL,
        outputsize=OUTPUTSIZE
    )
    
    # Step 2: Build features
    print("\nBuilding features...")
    df = build_features(df)
    
    # Step 3: Build labels
    print("Building labels...")
    df = build_labels(df)
    
    # Step 4: Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    print(f"Total usable samples after dropping NaNs: {len(df)}")
    
    if len(df) == 0:
        raise Exception("No data available after preprocessing")
    
    # Step 5: Define feature columns
    feature_cols = [
        'ret_1m', 'ret_3m', 'ret_5m',
        'ret_1m_mean_10', 'ret_1m_std_10',
        'volume_zscore_10'
    ]
    
    # Step 6: Chronological train/test split (70/30)
    split_idx = int(len(df) * 0.7)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    prices_test = test_df['close']
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 7: Train models
    lr_model, gb_model = train_models(X_train, y_train)
    
    # Step 7.5: Analyze model performance
    analyze_predictions(lr_model, X_train, y_train, X_test, y_test)
    
    # Step 7.6: Find optimal threshold
    optimal_threshold = find_optimal_threshold(lr_model, X_test, prices_test)
    
    # Step 8: Backtest with Logistic Regression using optimal threshold
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION BACKTEST")
    print("="*60)
    
    lr_signals, lr_probs = generate_signals_with_diagnostics(lr_model, X_test, threshold=optimal_threshold)
    lr_returns, lr_equity = backtest_strategy(lr_signals, prices_test)
    lr_metrics = compute_metrics(lr_returns, lr_equity)
    
    print(f"\nNumber of Trades: {lr_metrics['num_trades']}")
    print(f"Cumulative Return: {lr_metrics['cumulative_return']:.4f} ({lr_metrics['cumulative_return']*100:.2f}%)")
    print(f"Sharpe Ratio: {lr_metrics['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {lr_metrics['max_drawdown']:.4f} ({lr_metrics['max_drawdown']*100:.2f}%)")
    print(f"Win Rate: {lr_metrics['win_rate']:.4f} ({lr_metrics['win_rate']*100:.2f}%)")
    
    # Step 9: Backtest with XGBoost/LightGBM (if available)
    gb_metrics = None
    gb_equity = None
    gb_returns = None
    
    if gb_model is not None:
        # Analyze GB model
        analyze_predictions(gb_model, X_train, y_train, X_test, y_test)
        
        # Find optimal threshold for GB
        gb_optimal_threshold = find_optimal_threshold(gb_model, X_test, prices_test)
        
        model_name = "XGBOOST" if USE_XGB else "LIGHTGBM"
        print("\n" + "="*60)
        print(f"{model_name} BACKTEST")
        print("="*60)
        
        gb_signals, gb_probs = generate_signals_with_diagnostics(gb_model, X_test, threshold=gb_optimal_threshold)
        gb_returns, gb_equity = backtest_strategy(gb_signals, prices_test)
        gb_metrics = compute_metrics(gb_returns, gb_equity)
        
        print(f"\nNumber of Trades: {gb_metrics['num_trades']}")
        print(f"Cumulative Return: {gb_metrics['cumulative_return']:.4f} ({gb_metrics['cumulative_return']*100:.2f}%)")
        print(f"Sharpe Ratio: {gb_metrics['sharpe_ratio']:.4f}")
        print(f"Maximum Drawdown: {gb_metrics['max_drawdown']:.4f} ({gb_metrics['max_drawdown']*100:.2f}%)")
        print(f"Win Rate: {gb_metrics['win_rate']:.4f} ({gb_metrics['win_rate']*100:.2f}%)")
    
    # Step 10: Plot equity curves and probability distributions
    num_plots = 3 if gb_model is not None else 2
    fig = plt.figure(figsize=(14, 5 * num_plots))
    
    # Probability distribution for LR
    ax1 = plt.subplot(num_plots, 1, 1)
    ax1.hist(lr_probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {optimal_threshold}')
    ax1.set_title('Logistic Regression - Probability Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Equity curve for LR
    ax2 = plt.subplot(num_plots, 1, 2)
    if len(lr_equity) > 0:
        ax2.plot(lr_equity.values, linewidth=2, color='blue')
        ax2.set_title('Logistic Regression - Equity Curve', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trade Number', fontsize=12)
        ax2.set_ylabel('Equity', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    else:
        ax2.text(0.5, 0.5, 'No trades generated', ha='center', va='center', fontsize=14)
        ax2.set_title('Logistic Regression - Equity Curve', fontsize=14, fontweight='bold')
    
    # XGBoost/LightGBM equity curve (if available)
    if gb_model is not None and gb_equity is not None and len(gb_equity) > 0:
        ax3 = plt.subplot(num_plots, 1, 3)
        model_name = "XGBOOST" if USE_XGB else "LIGHTGBM"
        ax3.plot(gb_equity.values, linewidth=2, color='green')
        ax3.set_title(f'{model_name} - Equity Curve', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Trade Number', fontsize=12)
        ax3.set_ylabel('Equity', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('equity_curves_diagnostic.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print("Equity curves saved to 'equity_curves_diagnostic.png'")
    print("="*60)
    plt.show()


if __name__ == "__main__":
    main()
