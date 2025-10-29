"""
Generate Sample BTC/USDT Data
Creates realistic sample data for testing and demo purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_btc_data(n_samples=1000, start_price=40000):
    """
    Generate realistic BTC/USDT price data using geometric Brownian motion.

    Args:
        n_samples: Number of 1-hour candles to generate
        start_price: Starting BTC price in USDT

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducibility

    # Parameters for realistic BTC volatility
    mu = 0.0001  # Drift (slight upward trend)
    sigma = 0.015  # Volatility (1.5% per hour)

    # Generate prices using geometric Brownian motion
    prices = [start_price]
    for _ in range(n_samples):
        change = np.random.normal(mu, sigma)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    prices = np.array(prices[1:])  # Remove initial price

    # Generate OHLCV for each candle
    data = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    for i, close_price in enumerate(prices):
        # Generate realistic OHLCV
        # High is typically higher than close
        high = close_price * (1 + np.abs(np.random.normal(0, 0.005)))
        # Low is typically lower than close
        low = close_price * (1 - np.abs(np.random.normal(0, 0.005)))

        # Open can be anywhere between low and high
        open_price = low + (high - low) * np.random.random()

        # Ensure OHLC consistency
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)

        # Generate realistic volume (BTC)
        base_volume = 50
        volume = base_volume * (1 + np.random.normal(0, 0.5))
        volume = max(volume, 1.0)  # Minimum volume

        # Timestamp
        timestamp = start_time + timedelta(hours=i)

        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })

    df = pd.DataFrame(data)

    # Convert timestamp to UTC
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize('UTC')

    return df


if __name__ == "__main__":
    print("Generating sample BTC/USDT data...")

    # Generate 1000 hours of data (about 42 days)
    df = generate_sample_btc_data(n_samples=1000, start_price=40000)

    # Save to CSV
    output_path = "data/sample_btcusdt_1h.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Sample data saved to {output_path}")
    print(f"\nData summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print(f"\nPrice statistics:")
    print(f"  Min close: ${df['close'].min():,.2f}")
    print(f"  Max close: ${df['close'].max():,.2f}")
    print(f"  Mean close: ${df['close'].mean():,.2f}")
    print(f"  Start: ${df['close'].iloc[0]:,.2f}")
    print(f"  End: ${df['close'].iloc[-1]:,.2f}")
    print(f"  Return: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"\n✅ Ready for training!")
