#!/usr/bin/env python3
"""
Quick Multi-Timeframe Regime Detection Test

Fast version using manual features only (no OpenFE) to quickly test if longer
timeframes show better predictive power than 5-minute data.

Uses last 50,000 bars for speed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import lightgbm as lgb
from sklearn.metrics import accuracy_score


def load_recent_5m_data(csv_path: Path, n_bars: int = 50000) -> pd.DataFrame:
    """Load most recent n_bars of 5-minute data."""
    print(f"Loading last {n_bars:,} bars from {csv_path.name}...")

    df = pd.read_csv(csv_path, skiprows=10)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Take last n_bars
    df = df.tail(n_bars)

    print(f"  Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    return df


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data."""
    df_resampled = df.resample(timeframe).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    return df_resampled


def calculate_mfe_mae(df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Calculate MFE/MAE for each bar."""
    mfe_list = []
    mae_list = []

    for i in range(len(df)):
        if i + horizon >= len(df):
            mfe_list.append(np.nan)
            mae_list.append(np.nan)
            continue

        current_close = df['Close'].iloc[i]
        future_highs = df['High'].iloc[i+1:i+1+horizon]
        future_lows = df['Low'].iloc[i+1:i+1+horizon]

        mfe = (future_highs.max() - current_close) / current_close * 100
        mae = (future_lows.min() - current_close) / current_close * 100

        mfe_list.append(mfe)
        mae_list.append(mae)

    df = df.copy()
    df['MFE'] = mfe_list
    df['MAE'] = mae_list
    return df


def create_regime_labels(df: pd.DataFrame, threshold: float = 1.2) -> pd.DataFrame:
    """Label regimes: 1=Bullish, -1=Bearish, 0=Neutral."""
    df = df.copy()
    mae_abs = df['MAE'].abs()

    conditions = [
        (df['MFE'] > 0) & (mae_abs > 0) & ((df['MFE'] / mae_abs) > threshold),
        (df['MFE'] > 0) & (mae_abs > 0) & ((mae_abs / df['MFE']) > threshold),
    ]
    choices = [1, -1]

    df['Regime'] = np.select(conditions, choices, default=0)
    return df


def create_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create manual technical features (proven to work)."""
    df = df.copy()

    # Returns
    df['ret_5'] = df['Close'].pct_change(5)
    df['ret_10'] = df['Close'].pct_change(10)
    df['ret_20'] = df['Close'].pct_change(20)

    # Volatility
    df['vol_10'] = df['Close'].pct_change().rolling(10).std()
    df['vol_20'] = df['Close'].pct_change().rolling(20).std()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Price position
    rolling_max = df['High'].rolling(20).max()
    rolling_min = df['Low'].rolling(20).min()
    df['price_pos'] = (df['Close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)

    # Momentum
    df['mom_5'] = df['Close'] - df['Close'].shift(5)
    df['mom_10'] = df['Close'] - df['Close'].shift(10)

    # Volume
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # Range
    df['range'] = (df['High'] - df['Low']) / df['Close']
    df['body'] = abs(df['Close'] - df['Open']) / df['Close']

    # Moving averages
    df['sma_10'] = df['Close'].rolling(10).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['ma_diff'] = (df['sma_10'] - df['sma_20']) / df['Close']

    return df


def test_timeframe(df: pd.DataFrame, timeframe_name: str, horizon: int):
    """Test regime detection on one timeframe."""
    print(f"\n{'='*60}")
    print(f"{timeframe_name}")
    print(f"{'='*60}")

    # Calculate MFE/MAE
    df = calculate_mfe_mae(df, horizon=horizon)

    # Create labels
    df = create_regime_labels(df, threshold=1.2)

    # Create features
    df = create_manual_features(df)

    # Prepare data
    df_clean = df.dropna()

    feature_cols = [col for col in df_clean.columns
                   if col not in ['Regime', 'MFE', 'MAE']]

    X = df_clean[feature_cols].reset_index(drop=True)
    y = df_clean['Regime'].reset_index(drop=True)

    # Train/val/test split
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]

    print(f"  Data: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Train LightGBM
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'num_leaves': 15,  # Reduced for speed
        'learning_rate': 0.1,
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train + 1)
    val_data = lgb.Dataset(X_val, label=y_val + 1, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=30,  # Reduced for speed
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=5, verbose=False)]
    )

    # Evaluate
    y_test_pred = model.predict(X_test).argmax(axis=1) - 1
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"  Test Accuracy: {test_acc:.3f} ({test_acc*100:.1f}%)")

    # Verdict
    if test_acc >= 0.55:
        verdict = "✓ PREDICTIVE - Shows exploitable patterns"
    elif test_acc >= 0.52:
        verdict = "~ MARGINAL - Slight edge"
    else:
        verdict = "✗ RANDOM - No predictive power"

    print(f"  Verdict: {verdict}")

    return {
        'timeframe': timeframe_name,
        'bars': len(df),
        'test_accuracy': test_acc,
        'verdict': verdict
    }


def main():
    """Run quick test on all timeframes."""

    csv_path = Path('user_strategies/data/raw/crypto_5m/binance_spot_BTCUSDT-5m_20220101-20250930_v2.10.0.csv')

    # Load recent data only
    df_5m = load_recent_5m_data(csv_path, n_bars=50000)

    # Timeframe configs: (name, resample_rule, horizon_bars)
    configs = [
        ('15-minute', '15min', 3),   # 3 bars = 45 min
        ('30-minute', '30min', 2),   # 2 bars = 60 min
        ('1-hour', '1h', 1),         # 1 bar = 60 min
        ('2-hour', '2h', 1),         # 1 bar = 120 min
    ]

    results = []

    for name, resample_rule, horizon in configs:
        # Resample
        df_resampled = resample_ohlcv(df_5m, resample_rule)
        print(f"\n{name}: Resampled to {len(df_resampled):,} bars")

        # Test
        result = test_timeframe(df_resampled, name, horizon)
        results.append(result)

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY: Multi-Timeframe Regime Detection")
    print(f"{'='*60}")
    print(f"{'Timeframe':<15} {'Test Accuracy':<15} {'Verdict'}")
    print(f"{'-'*60}")

    # Add 5m baseline for reference
    print(f"{'5-minute (ref)':<15} {'49.7% - 52.5%':<15} {'✗ RANDOM'}")

    for r in results:
        print(f"{r['timeframe']:<15} {r['test_accuracy']*100:>6.1f}%{'':<8} {r['verdict'].split(' - ')[0]}")

    print(f"{'-'*60}")

    # Conclusion
    best = max(results, key=lambda x: x['test_accuracy'])
    if best['test_accuracy'] >= 0.55:
        print(f"\n✓ FINDING: {best['timeframe']} shows predictive power at {best['test_accuracy']*100:.1f}%")
        print("  → Longer timeframes reduce noise and reveal exploitable patterns")
    elif best['test_accuracy'] >= 0.52:
        print(f"\n~ MARGINAL: {best['timeframe']} shows slight edge at {best['test_accuracy']*100:.1f}%")
        print("  → May be worth further investigation with more sophisticated features")
    else:
        print(f"\n✗ CONCLUSION: All timeframes show random performance (50-53%)")
        print("  → BTC appears efficient across all tested timeframes (5m to 2h)")
        print("  → Regime detection approach may be fundamentally limited")
        print("\nNext steps to consider:")
        print("  1. Test completely different markets (not BTC)")
        print("  2. Use regime-independent strategies (e.g., momentum, mean reversion)")
        print("  3. Accept market efficiency and focus on execution/cost optimization")


if __name__ == '__main__':
    main()
