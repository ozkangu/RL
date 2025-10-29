"""
pytest test suite for data_manager.py

Professional test suite with edge cases and comprehensive coverage.

Run with: pytest test_data_manager_pytest.py -v
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile
import os

from data_manager import (
    load_data,
    add_technical_indicators,
    normalize_features,
    split_data,
    prepare_data_pipeline
)


# Fixtures
@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample OHLCV data."""
    # Generate sample data
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H', tz='UTC')

    np.random.seed(42)
    base_price = 30000
    close = base_price + np.cumsum(np.random.randn(n_samples) * 100)
    open_prices = close * (1 + np.random.randn(n_samples) * 0.001)
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.randn(n_samples) * 0.005))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.randn(n_samples) * 0.005))
    volume = np.random.lognormal(10, 1, n_samples)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name

    yield temp_file

    # Cleanup
    if os.path.exists(temp_file):
        os.remove(temp_file)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with OHLCV data."""
    n_samples = 500
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H', tz='UTC')

    np.random.seed(42)
    close = 30000 + np.cumsum(np.random.randn(n_samples) * 100)
    open_prices = close * (1 + np.random.randn(n_samples) * 0.001)
    high = np.maximum(open_prices, close) * 1.01
    low = np.minimum(open_prices, close) * 0.99
    volume = np.random.lognormal(10, 1, n_samples)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


# Test load_data()
class TestLoadData:
    """Test suite for load_data() function."""

    def test_load_valid_csv(self, sample_csv_file):
        """Test loading a valid CSV file."""
        df = load_data(sample_csv_file)

        assert len(df) > 0, "DataFrame should not be empty"
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert df.index.tz is not None
        assert df.index.tz.zone == 'UTC'

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent_file.csv")

    def test_load_missing_columns(self):
        """Test loading CSV with missing required columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create CSV without 'close' column
            pd.DataFrame({'open': [1, 2], 'high': [3, 4], 'low': [0.5, 1]}).to_csv(f.name, index=False)
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                load_data(temp_file)
        finally:
            os.remove(temp_file)

    def test_case_insensitive_columns(self):
        """Test that column names are case-insensitive."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'OPEN': [1, 2],
                'HIGH': [3, 4],
                'LOW': [0.5, 1],
                'CLOSE': [2, 3],
                'VOLUME': [100, 200]
            })
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            result = load_data(temp_file)
            assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        finally:
            os.remove(temp_file)

    def test_ohlcv_validation(self):
        """Test that invalid OHLCV data is filtered out."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=5, freq='1H'),
                'open': [100, 200, -50, 150, 100],  # One negative
                'high': [110, 210, 160, 160, 110],
                'low': [90, 190, 40, 140, 90],
                'close': [105, 205, 55, 155, 105],
                'volume': [1000, 2000, 3000, 4000, 5000]
            })
            df.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            result = load_data(temp_file)
            # Should have removed the negative price row
            assert len(result) < 5
        finally:
            os.remove(temp_file)


# Test add_technical_indicators()
class TestAddTechnicalIndicators:
    """Test suite for add_technical_indicators() function."""

    def test_add_indicators_valid_data(self, sample_dataframe):
        """Test adding technical indicators to valid data."""
        df_with_ta = add_technical_indicators(sample_dataframe)

        expected_indicators = ['rsi_14', 'macd', 'macd_signal', 'atr_14',
                              'bb_high', 'bb_low', 'returns', 'log_returns',
                              'volume_ratio', 'bb_position']

        for indicator in expected_indicators:
            assert indicator in df_with_ta.columns, f"Missing indicator: {indicator}"

    def test_rsi_range(self, sample_dataframe):
        """Test that RSI is within valid range [0, 100]."""
        df_with_ta = add_technical_indicators(sample_dataframe)

        assert df_with_ta['rsi_14'].min() >= 0
        assert df_with_ta['rsi_14'].max() <= 100

    def test_missing_required_column(self):
        """Test adding indicators to DataFrame with missing column."""
        df = pd.DataFrame({'close': [1, 2, 3]})

        with pytest.raises(ValueError, match="Missing required column"):
            add_technical_indicators(df)

    def test_no_nan_in_output(self, sample_dataframe):
        """Test that output has no NaN values (they should be dropped)."""
        df_with_ta = add_technical_indicators(sample_dataframe)

        # Check that NaN values are minimal (only at edges if any)
        nan_count = df_with_ta.isna().sum().sum()
        assert nan_count == 0, "Should have no NaN values after dropna()"


# Test normalize_features()
class TestNormalizeFeatures:
    """Test suite for normalize_features() function."""

    def test_zscore_normalization(self, sample_dataframe):
        """Test z-score normalization."""
        df_norm = normalize_features(sample_dataframe, method='zscore', window=50, columns=['close'])

        assert 'close_norm' in df_norm.columns
        # Mean should be close to 0, std close to 1 (after warm-up period)
        assert abs(df_norm['close_norm'].mean()) < 1.0
        assert 0.5 < df_norm['close_norm'].std() < 1.5

    def test_minmax_normalization(self, sample_dataframe):
        """Test min-max normalization."""
        df_norm = normalize_features(sample_dataframe, method='minmax', window=50, columns=['close'])

        assert 'close_norm' in df_norm.columns
        # Values should be roughly in [0, 1] range
        assert df_norm['close_norm'].min() >= -0.1  # Small tolerance
        assert df_norm['close_norm'].max() <= 1.1

    def test_invalid_method(self, sample_dataframe):
        """Test that invalid normalization method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_features(sample_dataframe, method='invalid')

    def test_outlier_clipping(self, sample_dataframe):
        """Test that outliers are clipped in z-score method."""
        df_norm = normalize_features(sample_dataframe, method='zscore', window=50,
                                     columns=['close'], clip_outliers=True, clip_std=3.0)

        assert 'close_norm' in df_norm.columns
        # All values should be within [-3, 3]
        assert df_norm['close_norm'].min() >= -3.0
        assert df_norm['close_norm'].max() <= 3.0


# Test split_data()
class TestSplitData:
    """Test suite for split_data() function."""

    def test_split_ratios(self, sample_dataframe):
        """Test that split ratios are correct."""
        train_df, val_df, test_df = split_data(sample_dataframe, train_ratio=0.6, val_ratio=0.2)

        total_len = len(train_df) + len(val_df) + len(test_df)
        assert total_len == len(sample_dataframe)

        train_ratio_actual = len(train_df) / len(sample_dataframe)
        val_ratio_actual = len(val_df) / len(sample_dataframe)

        assert 0.55 <= train_ratio_actual <= 0.65
        assert 0.15 <= val_ratio_actual <= 0.25

    def test_temporal_ordering(self, sample_dataframe):
        """Test that temporal ordering is preserved."""
        train_df, val_df, test_df = split_data(sample_dataframe, train_ratio=0.6, val_ratio=0.2)

        assert train_df.index.max() < val_df.index.min()
        assert val_df.index.max() < test_df.index.min()

    def test_invalid_ratios(self, sample_dataframe):
        """Test that invalid ratios raise ValueError."""
        with pytest.raises(ValueError):
            split_data(sample_dataframe, train_ratio=0.7, val_ratio=0.5)

    def test_empty_splits(self, sample_dataframe):
        """Test edge case with very small dataset."""
        small_df = sample_dataframe.iloc[:10]
        train_df, val_df, test_df = split_data(small_df, train_ratio=0.6, val_ratio=0.2)

        assert len(train_df) > 0
        assert len(val_df) >= 0  # Might be 0 for very small datasets
        assert len(test_df) >= 0


# Test prepare_data_pipeline()
class TestPrepareDataPipeline:
    """Test suite for prepare_data_pipeline() function."""

    def test_full_pipeline(self, sample_csv_file):
        """Test complete data pipeline."""
        train_df, val_df, test_df = prepare_data_pipeline(
            file_path=sample_csv_file,
            add_indicators=True,
            normalize=False,
            train_ratio=0.6,
            val_ratio=0.2
        )

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        assert 'rsi_14' in train_df.columns

    def test_pipeline_without_indicators(self, sample_csv_file):
        """Test pipeline without technical indicators."""
        train_df, val_df, test_df = prepare_data_pipeline(
            file_path=sample_csv_file,
            add_indicators=False,
            normalize=False
        )

        assert 'rsi_14' not in train_df.columns
        assert all(col in train_df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_pipeline_with_normalization(self, sample_csv_file):
        """Test pipeline with normalization."""
        train_df, val_df, test_df = prepare_data_pipeline(
            file_path=sample_csv_file,
            add_indicators=True,
            normalize=True,
            normalize_method='zscore'
        )

        # Check for normalized columns
        norm_cols = [col for col in train_df.columns if col.endswith('_norm')]
        assert len(norm_cols) > 0, "Should have normalized columns"


# Edge cases
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_dataframe(self):
        """Test behavior with single row DataFrame."""
        df = pd.DataFrame({
            'open': [100],
            'high': [110],
            'low': [90],
            'close': [105],
            'volume': [1000]
        })

        # Should raise or handle gracefully
        with pytest.raises(Exception):
            add_technical_indicators(df)

    def test_zero_volume(self):
        """Test handling of zero volume."""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [110, 111],
            'low': [90, 91],
            'close': [105, 106],
            'volume': [0, 1000]
        })

        # Should handle without crashing
        result = normalize_features(df, method='zscore', window=2, columns=['volume'])
        assert 'volume_norm' in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
