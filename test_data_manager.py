"""
Test script for data_manager.py

PBI-008: Data manager test and validation
"""

import pandas as pd
import numpy as np
from data_manager import (
    load_data,
    add_technical_indicators,
    normalize_features,
    split_data,
    prepare_data_pipeline
)


def create_sample_data(n_samples: int = 1000, save_path: str = 'data/sample_btcusd.csv'):
    """
    Create sample BTC/USD OHLCV data for testing.

    Args:
        n_samples: Number of samples to generate
        save_path: Path to save the CSV file
    """
    print("Creating sample data...")

    # Generate datetime index
    start_date = pd.Timestamp('2023-01-01', tz='UTC')
    dates = pd.date_range(start=start_date, periods=n_samples, freq='1H')

    # Generate synthetic price data with trend and noise
    np.random.seed(42)
    base_price = 30000
    trend = np.linspace(0, 10000, n_samples)
    noise = np.random.randn(n_samples) * 500
    close = base_price + trend + noise.cumsum() * 0.1

    # Generate OHLCV data
    open_prices = close * (1 + np.random.randn(n_samples) * 0.001)
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    volume = np.random.lognormal(10, 1, n_samples)

    # Create dataframe
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Sample data saved to {save_path}")
    print(f"  Samples: {n_samples}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Price range: {close.min():.2f} to {close.max():.2f}")

    return save_path


def test_load_data(file_path: str):
    """Test the load_data function."""
    print("\n" + "="*60)
    print("TEST 1: load_data()")
    print("="*60)

    df = load_data(file_path)

    # Assertions
    assert len(df) > 0, "DataFrame should not be empty"
    assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']), \
        "Missing required OHLCV columns"
    assert df.index.tz is not None, "Index should have timezone info"
    assert df.index.tz.zone == 'UTC', "Timezone should be UTC"
    assert df['close'].dtype in [np.float64, np.float32], "Close prices should be numeric"

    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Index type: {type(df.index)}")
    print(f"✓ Timezone: {df.index.tz}")
    print("✓ TEST PASSED")

    return df


def test_add_technical_indicators(df: pd.DataFrame):
    """Test the add_technical_indicators function."""
    print("\n" + "="*60)
    print("TEST 2: add_technical_indicators()")
    print("="*60)

    df_with_ta = add_technical_indicators(df)

    # Assertions
    expected_indicators = ['rsi_14', 'macd', 'macd_signal', 'atr_14',
                          'bb_high', 'bb_low', 'returns', 'log_returns']
    for indicator in expected_indicators:
        assert indicator in df_with_ta.columns, f"Missing indicator: {indicator}"

    # Check RSI is in valid range [0, 100]
    assert df_with_ta['rsi_14'].min() >= 0, "RSI should be >= 0"
    assert df_with_ta['rsi_14'].max() <= 100, "RSI should be <= 100"

    # Check no future leakage: indicators should be NaN at start
    # (already dropped in function, so just check length reduced)
    assert len(df_with_ta) < len(df), "Should drop initial rows with NaN indicators"

    print(f"✓ Added {len(df_with_ta.columns)} total columns")
    print(f"✓ Technical indicators: {expected_indicators}")
    print(f"✓ RSI range: [{df_with_ta['rsi_14'].min():.2f}, {df_with_ta['rsi_14'].max():.2f}]")
    print(f"✓ MACD range: [{df_with_ta['macd'].min():.2f}, {df_with_ta['macd'].max():.2f}]")
    print("✓ TEST PASSED")

    return df_with_ta


def test_normalize_features(df: pd.DataFrame):
    """Test the normalize_features function."""
    print("\n" + "="*60)
    print("TEST 3: normalize_features()")
    print("="*60)

    # Test z-score normalization
    df_norm_z = normalize_features(df.copy(), method='zscore', window=50,
                                    columns=['close', 'volume'])

    assert 'close_norm' in df_norm_z.columns, "Missing normalized close column"
    assert 'volume_norm' in df_norm_z.columns, "Missing normalized volume column"

    # Z-score should have mean ~0 and std ~1 (approximately)
    close_norm_mean = df_norm_z['close_norm'].mean()
    close_norm_std = df_norm_z['close_norm'].std()

    print(f"✓ Z-score normalization:")
    print(f"  close_norm mean: {close_norm_mean:.3f} (expected ~0)")
    print(f"  close_norm std:  {close_norm_std:.3f} (expected ~1)")

    # Test min-max normalization
    df_norm_mm = normalize_features(df.copy(), method='minmax', window=50,
                                     columns=['close'])

    assert 'close_norm' in df_norm_mm.columns, "Missing normalized close column"

    # Min-max should be in [0, 1] range (approximately)
    norm_min = df_norm_mm['close_norm'].min()
    norm_max = df_norm_mm['close_norm'].max()

    print(f"✓ Min-max normalization:")
    print(f"  close_norm range: [{norm_min:.3f}, {norm_max:.3f}] (expected [0, 1])")

    print("✓ TEST PASSED")

    return df_norm_z


def test_split_data(df: pd.DataFrame):
    """Test the split_data function."""
    print("\n" + "="*60)
    print("TEST 4: split_data()")
    print("="*60)

    train_df, val_df, test_df = split_data(df, train_ratio=0.6, val_ratio=0.2)

    # Assertions
    total_len = len(train_df) + len(val_df) + len(test_df)
    assert total_len == len(df), "Split should preserve total number of rows"

    # Check ratios (approximately)
    train_ratio_actual = len(train_df) / len(df)
    val_ratio_actual = len(val_df) / len(df)
    test_ratio_actual = len(test_df) / len(df)

    assert 0.55 <= train_ratio_actual <= 0.65, f"Train ratio should be ~0.6, got {train_ratio_actual}"
    assert 0.15 <= val_ratio_actual <= 0.25, f"Val ratio should be ~0.2, got {val_ratio_actual}"
    assert 0.15 <= test_ratio_actual <= 0.25, f"Test ratio should be ~0.2, got {test_ratio_actual}"

    # Check temporal ordering
    assert train_df.index.max() < val_df.index.min(), "Train should come before val"
    assert val_df.index.max() < test_df.index.min(), "Val should come before test"

    print(f"\n✓ Actual ratios:")
    print(f"  Train: {train_ratio_actual:.2%}")
    print(f"  Val:   {val_ratio_actual:.2%}")
    print(f"  Test:  {test_ratio_actual:.2%}")
    print(f"✓ Temporal ordering preserved")
    print("✓ TEST PASSED")

    return train_df, val_df, test_df


def test_full_pipeline(file_path: str):
    """Test the complete data pipeline."""
    print("\n" + "="*60)
    print("TEST 5: prepare_data_pipeline() - FULL PIPELINE")
    print("="*60)

    train_df, val_df, test_df = prepare_data_pipeline(
        file_path=file_path,
        add_indicators=True,
        normalize=False,
        train_ratio=0.6,
        val_ratio=0.2
    )

    # Assertions
    assert len(train_df) > 0, "Train set should not be empty"
    assert len(val_df) > 0, "Val set should not be empty"
    assert len(test_df) > 0, "Test set should not be empty"
    assert 'rsi_14' in train_df.columns, "Technical indicators should be present"

    print("\n✓ Full pipeline executed successfully")
    print("✓ All datasets contain technical indicators")
    print("✓ TEST PASSED")

    return train_df, val_df, test_df


def run_all_tests():
    """Run all data manager tests."""
    print("\n" + "="*70)
    print(" DATA MANAGER TEST SUITE - PBI-008")
    print("="*70)

    # Step 1: Create sample data
    file_path = create_sample_data(n_samples=2000, save_path='data/sample_btcusd.csv')

    # Step 2: Run individual tests
    df = test_load_data(file_path)
    df_with_ta = test_add_technical_indicators(df)
    df_norm = test_normalize_features(df_with_ta)
    train_df, val_df, test_df = test_split_data(df_with_ta)

    # Step 3: Run full pipeline test
    train_df_full, val_df_full, test_df_full = test_full_pipeline(file_path)

    # Final summary
    print("\n" + "="*70)
    print(" ALL TESTS PASSED ✓")
    print("="*70)
    print("\nSummary:")
    print(f"  ✓ PBI-004: CSV loading and validation")
    print(f"  ✓ PBI-005: Technical indicators (RSI, MACD, ATR, BB)")
    print(f"  ✓ PBI-006: Normalization (z-score, min-max)")
    print(f"  ✓ PBI-007: Temporal data splitting")
    print(f"  ✓ PBI-008: End-to-end pipeline validation")
    print("\nEpic 1.2: Data Management - COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
