"""
Binance Data Fetcher
Fetches historical BTC/USD OHLCV data from Binance and saves to CSV.

Usage:
    python fetch_data.py --symbol BTCUSDT --timeframe 1h --days 365
    python fetch_data.py --symbol BTCUSDT --start-date 2023-01-01 --end-date 2024-01-01
"""

import argparse
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
from typing import Optional, List
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_binance_credentials() -> tuple:
    """
    Load Binance API credentials from .env file.

    Returns:
        Tuple of (api_key, api_secret)

    Raises:
        ValueError: If credentials are not found or invalid
    """
    load_dotenv()

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        raise ValueError(
            "Binance API credentials not found!\n"
            "Please:\n"
            "1. Copy .env.example to .env\n"
            "2. Add your Binance API key and secret to .env\n"
            "3. Get API keys from: https://www.binance.com/en/my/settings/api-management"
        )

    if api_key == 'your_api_key_here' or api_secret == 'your_api_secret_here':
        raise ValueError(
            "Please replace placeholder values in .env with your actual Binance API credentials"
        )

    logger.info("Binance credentials loaded successfully")
    return api_key, api_secret


def initialize_binance_client(api_key: str, api_secret: str) -> ccxt.binance:
    """
    Initialize Binance exchange client.

    Args:
        api_key: Binance API key
        api_secret: Binance API secret

    Returns:
        Initialized ccxt Binance exchange object
    """
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True,  # Important: respect rate limits
        'options': {
            'defaultType': 'spot',  # Use spot market
        }
    })

    logger.info("Binance client initialized")
    return exchange


def fetch_ohlcv_data(exchange: ccxt.binance,
                     symbol: str,
                     timeframe: str,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     days: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Binance.

    Args:
        exchange: ccxt Binance exchange object
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for candles (e.g., '1h', '4h', '1d')
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        days: Number of days to fetch (alternative to start/end dates)

    Returns:
        DataFrame with OHLCV data

    Raises:
        ValueError: If date parameters are invalid
    """
    logger.info(f"Fetching {symbol} data with timeframe {timeframe}...")

    # Determine date range
    if days:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)
        logger.info(f"Fetching last {days} days of data")
    elif start_date and end_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        logger.info(f"Fetching data from {start_date} to {end_date}")
    elif start_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.utcnow()
        logger.info(f"Fetching data from {start_date} to now")
    else:
        # Default: last 365 days
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=365)
        logger.info(f"Fetching last 365 days of data (default)")

    # Convert to timestamps (milliseconds)
    since = int(start_dt.timestamp() * 1000)
    end_timestamp = int(end_dt.timestamp() * 1000)

    # Fetch data in batches (Binance limit: 1000 candles per request)
    all_candles = []
    current_since = since

    batch_count = 0
    while current_since < end_timestamp:
        try:
            # Fetch batch
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=1000  # Max candles per request
            )

            if not candles:
                break

            all_candles.extend(candles)
            batch_count += 1

            # Update timestamp for next batch
            current_since = candles[-1][0] + 1

            logger.info(f"Fetched batch {batch_count}: {len(candles)} candles "
                       f"(total: {len(all_candles)})")

            # Stop if we've reached the end date
            if candles[-1][0] >= end_timestamp:
                break

            # Small delay to respect rate limits (even with enableRateLimit=True)
            time.sleep(0.1)

        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded, waiting 60 seconds...")
            time.sleep(60)
            continue

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    if not all_candles:
        raise ValueError(f"No data fetched for {symbol}")

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

    # Filter by end date (in case we fetched extra)
    df = df[df['timestamp'] <= pd.to_datetime(end_dt, utc=True)]

    # Remove duplicates (in case of overlap in batches)
    df = df.drop_duplicates(subset=['timestamp'], keep='first')

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Fetched {len(df)} total candles")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean OHLCV data.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        Cleaned DataFrame
    """
    logger.info("Validating data...")

    initial_len = len(df)

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Found missing values:\n{missing[missing > 0]}")
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with missing values")

    # Check for negative prices or volume
    negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
    if negative_prices.sum() > 0:
        logger.warning(f"Found {negative_prices.sum()} rows with negative/zero prices")
        df = df[~negative_prices]

    # Check high >= low
    invalid_hl = df['high'] < df['low']
    if invalid_hl.sum() > 0:
        logger.warning(f"Found {invalid_hl.sum()} rows where high < low")
        df = df[~invalid_hl]

    # Check close is within [low, high]
    invalid_close = (df['close'] > df['high']) | (df['close'] < df['low'])
    if invalid_close.sum() > 0:
        logger.warning(f"Found {invalid_close.sum()} rows where close is outside [low, high]")
        df = df[~invalid_close]

    # Check for duplicate timestamps
    duplicates = df['timestamp'].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate timestamps")
        df = df.drop_duplicates(subset=['timestamp'], keep='first')

    logger.info(f"Validation complete: {len(df)} valid candles (removed {initial_len - len(df)})")

    return df


def save_to_csv(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to CSV file.

    Args:
        df: DataFrame with OHLCV data
        output_path: Path to output CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)
    logger.info(f"Data saved to {output_file}")

    # Print summary statistics
    logger.info("\n" + "="*70)
    logger.info("DATA SUMMARY")
    logger.info("="*70)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    logger.info(f"\nPrice statistics:")
    logger.info(f"  Min close: ${df['close'].min():,.2f}")
    logger.info(f"  Max close: ${df['close'].max():,.2f}")
    logger.info(f"  Mean close: ${df['close'].mean():,.2f}")
    logger.info(f"  Current close: ${df['close'].iloc[-1]:,.2f}")
    logger.info(f"\nVolume statistics:")
    logger.info(f"  Total volume: {df['volume'].sum():,.2f}")
    logger.info(f"  Mean volume: {df['volume'].mean():,.2f}")
    logger.info("="*70)


def main():
    """
    Main function to fetch and save Binance data.
    """
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV data from Binance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading pair symbol (e.g., BTC/USDT, ETH/USDT)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        help='Candle timeframe'
    )

    parser.add_argument(
        '--days',
        type=int,
        help='Number of days to fetch (from now backwards)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/btcusdt_1h.csv',
        help='Output CSV file path'
    )

    parser.add_argument(
        '--no-api-key',
        action='store_true',
        help='Fetch public data without API key (limited to recent data)'
    )

    args = parser.parse_args()

    try:
        logger.info("="*70)
        logger.info("BINANCE DATA FETCHER")
        logger.info("="*70)

        # Initialize Binance client
        if args.no_api_key:
            logger.info("Using public API (no authentication)")
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
        else:
            api_key, api_secret = load_binance_credentials()
            exchange = initialize_binance_client(api_key, api_secret)

        # Fetch data
        df = fetch_ohlcv_data(
            exchange=exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            days=args.days
        )

        # Validate data
        df = validate_data(df)

        # Save to CSV
        save_to_csv(df, args.output)

        logger.info("\n" + "="*70)
        logger.info("SUCCESS! Data fetching completed")
        logger.info("="*70)
        logger.info(f"\nNext steps:")
        logger.info(f"1. Check the data: data/btcusdt_1h.csv")
        logger.info(f"2. Update configs/env.yaml to use this data file")
        logger.info(f"3. Run training: python train.py")

    except ValueError as e:
        logger.error(f"\nConfiguration error: {e}")
        return 1

    except Exception as e:
        logger.error(f"\nFailed to fetch data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
