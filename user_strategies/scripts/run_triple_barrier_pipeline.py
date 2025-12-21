#!/usr/bin/env python3
"""End-to-end triple-barrier classification pipeline.

ADR: 2025-12-20-clickhouse-triple-barrier-backtest

This script demonstrates the complete pipeline:
1. Load data from ClickHouse (or use sample data)
2. Compute microstructure features
3. Generate triple-barrier labels
4. Train classifier with purged CV
5. Run backtest with threshold optimization
6. Output calibration diagnostics

Usage:
    uv run user_strategies/scripts/run_triple_barrier_pipeline.py

Environment Variables:
    CLICKHOUSE_HOST, CLICKHOUSE_PORT, etc. - ClickHouse connection
    USE_SAMPLE_DATA=1 - Use built-in sample data instead of ClickHouse
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(
    symbol: str = "BTCUSDT",
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    timeframe: str = "1h",
    use_sample: bool = False,
) -> pd.DataFrame:
    """Load OHLCV data from ClickHouse or sample data.

    Args:
        symbol: Trading pair symbol
        start: Start date
        end: End date
        timeframe: Data timeframe
        use_sample: If True, use backtesting.py sample data

    Returns:
        DataFrame with OHLCV and microstructure columns
    """
    if use_sample:
        logger.info("Using sample data (EURUSD from backtesting.py)")
        from backtesting.test import EURUSD

        df = EURUSD.copy()
        # Add synthetic microstructure features for demonstration
        df["taker_buy_ratio"] = 0.5 + 0.1 * np.random.randn(len(df))
        df["taker_buy_ratio"] = df["taker_buy_ratio"].clip(0.3, 0.7)
        df["order_flow_imbalance"] = 0.1 * np.random.randn(len(df))
        df["total_trades"] = np.random.poisson(1000, len(df))
        df["range_volatility"] = (df["High"] - df["Low"]) / df["Close"]
        return df

    try:
        from user_strategies.strategies.clickhouse_adapter import (
            get_data_from_clickhouse,
        )

        logger.info(f"Loading data from ClickHouse: {symbol} {start} to {end}")
        return get_data_from_clickhouse(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            include_microstructure=True,
        )
    except ImportError:
        logger.warning(
            "gapless-crypto-clickhouse not available, falling back to sample data"
        )
        return load_data(use_sample=True)
    except Exception as e:
        logger.warning(f"ClickHouse connection failed: {e}, using sample data")
        return load_data(use_sample=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute feature matrix from OHLCV + microstructure data.

    Args:
        df: DataFrame with OHLCV and microstructure columns

    Returns:
        DataFrame with computed features
    """
    from user_strategies.strategies.microstructure_features import FEATURE_COLUMNS

    logger.info("Computing features...")

    features = pd.DataFrame(index=df.index)

    # Price-based features
    features["returns"] = df["Close"].pct_change()
    features["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
    features["volatility_20"] = features["returns"].rolling(20).std()

    # Range features
    features["range_pct"] = (df["High"] - df["Low"]) / df["Close"]
    features["body_pct"] = (df["Close"] - df["Open"]) / df["Close"]

    # Microstructure features (if available)
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            features[col] = df[col]
            # Add rolling z-score normalization
            roll_mean = df[col].rolling(20).mean()
            roll_std = df[col].rolling(20).std()
            features[f"{col}_zscore"] = (df[col] - roll_mean) / roll_std.replace(0, 1)

    # Momentum features
    for period in [5, 10, 20]:
        features[f"momentum_{period}"] = df["Close"] / df["Close"].shift(period) - 1

    # Volume features
    if "Volume" in df.columns:
        features["volume_sma_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    logger.info(f"Computed {len(features.columns)} features")
    return features


def run_pipeline(
    symbol: str = "BTCUSDT",
    start: str = "2024-01-01",
    end: str = "2024-06-30",
    barrier_pct: float = 0.01,
    horizon_bars: int = 24,
    train_ratio: float = 0.7,
    use_sample: bool = False,
    output_dir: Path | None = None,
) -> dict:
    """Run the complete triple-barrier pipeline.

    Args:
        symbol: Trading pair
        start: Start date
        end: End date
        barrier_pct: Barrier percentage for labels
        horizon_bars: Horizon bars for labels
        train_ratio: Fraction of data for training
        use_sample: Use sample data instead of ClickHouse
        output_dir: Directory for output files

    Returns:
        Dictionary with pipeline results
    """
    from user_strategies.strategies.calibration import (
        compute_calibration_metrics,
        print_calibration_report,
    )
    from user_strategies.strategies.classifier import TripleBarrierClassifier
    from user_strategies.strategies.purged_cv import purged_train_test_split
    from user_strategies.strategies.triple_barrier import (
        compute_triple_barrier_labels,
        get_label_statistics,
    )

    results = {"symbol": symbol, "start": start, "end": end}

    # Step 1: Load data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 60)
    df = load_data(symbol=symbol, start=start, end=end, use_sample=use_sample)
    results["n_bars"] = len(df)
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Step 2: Compute features
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Computing features")
    logger.info("=" * 60)
    features = compute_features(df)
    feature_cols = [c for c in features.columns if not c.startswith("_")]
    results["n_features"] = len(feature_cols)

    # Step 3: Generate labels
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Generating triple-barrier labels")
    logger.info("=" * 60)
    labels = compute_triple_barrier_labels(
        df["Close"], barrier_pct=barrier_pct, horizon_bars=horizon_bars
    )
    label_stats = get_label_statistics(labels)
    results["label_stats"] = label_stats
    logger.info(f"Label distribution:")
    logger.info(f"  Upper (+1): {label_stats['n_upper']} ({label_stats['pct_upper']:.1f}%)")
    logger.info(f"  Lower (-1): {label_stats['n_lower']} ({label_stats['pct_lower']:.1f}%)")
    logger.info(f"  Timeout (0): {label_stats['n_timeout']} ({label_stats['pct_timeout']:.1f}%)")

    # Step 4: Prepare data for training
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Preparing train/test split with purging")
    logger.info("=" * 60)

    # Align features and labels, remove NaN
    valid_mask = ~labels.isna() & ~features.isna().any(axis=1)
    X = features.loc[valid_mask].values
    y = labels.loc[valid_mask].values
    valid_index = features.index[valid_mask]

    X_train, X_test, y_train, y_test = purged_train_test_split(
        X, y, test_size=1 - train_ratio, horizon=horizon_bars, embargo=0
    )
    results["n_train"] = len(y_train)
    results["n_test"] = len(y_test)
    logger.info(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Step 5: Train classifier
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Training probabilistic classifier")
    logger.info("=" * 60)
    classifier = TripleBarrierClassifier(model_type="logistic", calibrated=True)
    classifier.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    accuracy = np.mean(y_pred == y_test)
    log_loss_val = classifier.log_loss(X_test, y_test)
    results["test_accuracy"] = accuracy
    results["test_log_loss"] = log_loss_val
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Log-Loss: {log_loss_val:.4f}")

    # Step 6: Calibration diagnostics
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Calibration diagnostics")
    logger.info("=" * 60)
    cal_metrics = compute_calibration_metrics(y_test, y_proba)
    results["calibration"] = {
        "brier_score": cal_metrics.brier_score,
        "ece": cal_metrics.ece,
        "mce": cal_metrics.mce,
    }
    print_calibration_report(cal_metrics)

    # Step 7: Run backtest (if enough data)
    if len(X_test) > 100:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 7: Running backtest simulation")
        logger.info("=" * 60)

        try:
            from backtesting import Backtest

            from user_strategies.strategies.triple_barrier_strategy import (
                TripleBarrierProbStrategy,
                prepare_data_for_strategy,
            )

            # Prepare test period data with predictions
            test_start_idx = len(y_train)
            test_df = df.iloc[test_start_idx : test_start_idx + len(y_test)].copy()

            # Generate predictions for test data
            prob_upper = y_proba[:, 2]  # Class index 2 = label +1
            prob_lower = y_proba[:, 0]  # Class index 0 = label -1

            backtest_data = prepare_data_for_strategy(
                test_df, prob_upper=prob_upper, prob_lower=prob_lower
            )

            bt = Backtest(
                backtest_data,
                TripleBarrierProbStrategy,
                cash=10_000_000,
                commission=0.0002,
                margin=1.0,
            )

            stats = bt.run()
            results["backtest"] = {
                "return_pct": stats["Return [%]"],
                "sharpe": stats["Sharpe Ratio"],
                "max_drawdown": stats["Max. Drawdown [%]"],
                "n_trades": stats["# Trades"],
                "win_rate": stats["Win Rate [%]"],
            }

            logger.info("Backtest Results:")
            logger.info(f"  Return: {stats['Return [%]']:.2f}%")
            logger.info(f"  Sharpe: {stats['Sharpe Ratio']:.2f}")
            logger.info(f"  Max DD: {stats['Max. Drawdown [%]']:.2f}%")
            logger.info(f"  Trades: {stats['# Trades']}")
            logger.info(f"  Win Rate: {stats['Win Rate [%]']:.1f}%")

            # Save HTML report
            if output_dir:
                html_path = output_dir / f"backtest_{symbol}_{datetime.now():%Y%m%d_%H%M%S}.html"
                bt.plot(filename=str(html_path), open_browser=False)
                logger.info(f"Saved backtest report to {html_path}")

        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            results["backtest"] = {"error": str(e)}

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Triple-barrier classification pipeline"
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--start", default="2024-01-01", help="Start date")
    parser.add_argument("--end", default="2024-06-30", help="End date")
    parser.add_argument(
        "--barrier-pct", type=float, default=0.01, help="Barrier percentage"
    )
    parser.add_argument(
        "--horizon", type=int, default=24, help="Horizon bars for labels"
    )
    parser.add_argument(
        "--sample", action="store_true", help="Use sample data instead of ClickHouse"
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory")

    args = parser.parse_args()

    # Check environment variable for sample data
    use_sample = args.sample or os.environ.get("USE_SAMPLE_DATA") == "1"

    output_dir = args.output_dir or PROJECT_ROOT / "user_strategies" / "data" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_pipeline(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        barrier_pct=args.barrier_pct,
        horizon_bars=args.horizon,
        use_sample=use_sample,
        output_dir=output_dir,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Symbol: {results['symbol']}")
    print(f"Period: {results['start']} to {results['end']}")
    print(f"Bars: {results['n_bars']}")
    print(f"Features: {results['n_features']}")
    print(f"Train/Test: {results['n_train']}/{results['n_test']}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test Log-Loss: {results['test_log_loss']:.4f}")
    print(f"Brier Score: {results['calibration']['brier_score']:.4f}")
    print(f"ECE: {results['calibration']['ece']:.4f}")

    if "backtest" in results and "return_pct" in results["backtest"]:
        print(f"Backtest Return: {results['backtest']['return_pct']:.2f}%")
        print(f"Sharpe Ratio: {results['backtest']['sharpe']:.2f}")


if __name__ == "__main__":
    main()
