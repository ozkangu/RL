# Data Fetching Guide - Binance Historical Data

Complete guide to fetch historical BTC/USDT data from Binance for the RL trading bot.

---

## 📋 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `ccxt>=4.0.0` - Cryptocurrency exchange API library
- `python-dotenv>=1.0.0` - Environment variable management

### 2. Get Binance API Keys (Optional but Recommended)

**Why API Keys?**
- Higher rate limits
- Access to more historical data
- More reliable fetching

**How to Get API Keys:**

1. Go to [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. Click "Create API"
3. Complete 2FA verification
4. Name your API key (e.g., "RL Trading Data Fetcher")
5. Copy the API Key and Secret Key

**Important Security Settings:**
- ✅ Enable "Enable Reading" (required)
- ❌ Disable "Enable Spot & Margin Trading" (not needed)
- ❌ Disable "Enable Futures" (not needed)
- ❌ Disable "Enable Withdrawals" (never enable this!)

### 3. Configure API Credentials

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API keys
# Replace 'your_api_key_here' and 'your_api_secret_here'
nano .env  # or use your favorite editor
```

Your `.env` should look like:
```bash
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_API_SECRET=your_actual_secret_key_here
```

### 4. Fetch Data

```bash
# Fetch last 365 days of BTC/USDT 1-hour data (default)
python fetch_data.py

# Fetch last 30 days
python fetch_data.py --days 30

# Fetch specific date range
python fetch_data.py --start-date 2023-01-01 --end-date 2024-01-01

# Fetch without API key (public data, limited)
python fetch_data.py --no-api-key --days 30
```

---

## 🔧 Usage Examples

### Basic Usage

```bash
# Default: Last 365 days of BTC/USDT 1h data
python fetch_data.py
```

Output: `data/btcusdt_1h.csv`

### Custom Symbol

```bash
# Fetch Ethereum data
python fetch_data.py --symbol ETH/USDT

# Fetch other pairs
python fetch_data.py --symbol BNB/USDT
```

### Custom Timeframe

```bash
# 4-hour candles
python fetch_data.py --timeframe 4h

# Daily candles
python fetch_data.py --timeframe 1d

# Available: 1m, 5m, 15m, 30m, 1h, 4h, 1d
```

### Date Range Fetching

```bash
# Specific date range
python fetch_data.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31

# From specific date to now
python fetch_data.py --start-date 2023-06-01

# Last N days
python fetch_data.py --days 180
```

### Custom Output

```bash
# Save to custom location
python fetch_data.py --output my_data/btc_data.csv

# Different symbol and output
python fetch_data.py \
  --symbol ETH/USDT \
  --timeframe 4h \
  --days 90 \
  --output data/ethusdt_4h.csv
```

---

## 📊 Output Format

The script generates a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime (UTC) | Candle timestamp |
| `open` | float | Opening price (USDT) |
| `high` | float | Highest price (USDT) |
| `low` | float | Lowest price (USDT) |
| `close` | float | Closing price (USDT) |
| `volume` | float | Trading volume (BTC) |

Example:
```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00+00:00,16625.0,16650.5,16600.0,16630.2,125.45
2023-01-01 01:00:00+00:00,16630.2,16680.0,16620.0,16670.5,98.32
...
```

---

## ⚙️ Configuration

### For MVP Training

After fetching data, update `configs/env.yaml`:

```yaml
data:
  data_file: "data/btcusdt_1h.csv"  # Update this path
  timezone: "UTC"
```

### Recommended Data Amount

| Purpose | Minimum | Recommended |
|---------|---------|-------------|
| **Quick Test** | 30 days (~720 hours) | 60 days |
| **MVP Training** | 90 days (~2,160 hours) | 180 days |
| **Production** | 180 days | 365+ days |

---

## 🔍 Data Validation

The script automatically validates data:

✅ **Checks performed:**
- Missing values (NaN)
- Negative prices or volume
- High < Low inconsistencies
- Close outside [Low, High] range
- Duplicate timestamps

⚠️ **Invalid rows are automatically removed**

---

## 📈 Example Output

```
======================================================================
BINANCE DATA FETCHER
======================================================================
2024-10-28 - INFO - Binance credentials loaded successfully
2024-10-28 - INFO - Binance client initialized
2024-10-28 - INFO - Fetching BTC/USDT data with timeframe 1h...
2024-10-28 - INFO - Fetching last 365 days of data
2024-10-28 - INFO - Fetched batch 1: 1000 candles (total: 1000)
2024-10-28 - INFO - Fetched batch 2: 1000 candles (total: 2000)
...
2024-10-28 - INFO - Fetched batch 9: 760 candles (total: 8760)
2024-10-28 - INFO - Fetched 8760 total candles
2024-10-28 - INFO - Date range: 2023-10-28 to 2024-10-28
2024-10-28 - INFO - Validating data...
2024-10-28 - INFO - Validation complete: 8760 valid candles
2024-10-28 - INFO - Data saved to data/btcusdt_1h.csv

======================================================================
DATA SUMMARY
======================================================================
Total samples: 8760
Date range: 2023-10-28 00:00:00+00:00 to 2024-10-28 00:00:00+00:00
Duration: 365 days

Price statistics:
  Min close: $26,500.00
  Max close: $73,500.00
  Mean close: $45,250.50
  Current close: $67,850.00

Volume statistics:
  Total volume: 125,450.50
  Mean volume: 14.32
======================================================================

SUCCESS! Data fetching completed

Next steps:
1. Check the data: data/btcusdt_1h.csv
2. Update configs/env.yaml to use this data file
3. Run training: python train.py
```

---

## ❌ Troubleshooting

### Issue: "Binance API credentials not found"

**Solution:**
```bash
# Make sure .env file exists
ls -la .env

# If not, copy from example
cp .env.example .env

# Edit and add your API keys
nano .env
```

### Issue: "Please replace placeholder values in .env"

**Solution:**
You haven't replaced the placeholder values. Open `.env` and add your actual API keys from Binance.

### Issue: Rate limit exceeded

**Solution:**
- The script automatically retries after 60 seconds
- If using public API, consider getting API keys
- Reduce the date range (fetch smaller chunks)

### Issue: No data fetched

**Solution:**
- Check symbol format: Use `BTC/USDT` not `BTCUSDT`
- Check date range: Binance may not have data before 2017
- Verify internet connection

### Issue: "Symbol not found"

**Solution:**
- Ensure correct format: `BTC/USDT`, `ETH/USDT` (with slash)
- Check if symbol exists on Binance Spot market
- List available symbols: `python -c "import ccxt; print(ccxt.binance().load_markets().keys())"`

---

## 🔒 Security Best Practices

1. **Never commit `.env` file**
   - Already in `.gitignore`
   - Contains sensitive API keys

2. **API Key Permissions**
   - Only enable "Enable Reading"
   - Never enable trading or withdrawal permissions

3. **IP Restrictions (Optional)**
   - Add your IP to API key whitelist in Binance settings
   - Adds extra security layer

4. **Regular Key Rotation**
   - Consider rotating API keys every few months
   - Immediately revoke if compromised

---

## 🚀 Next Steps

After fetching data:

1. **Verify Data**
   ```bash
   # Check first few rows
   head data/btcusdt_1h.csv

   # Check row count
   wc -l data/btcusdt_1h.csv
   ```

2. **Update Config**
   ```bash
   # Edit configs/env.yaml
   nano configs/env.yaml

   # Set: data_file: "data/btcusdt_1h.csv"
   ```

3. **Test Pipeline**
   ```bash
   # Validate data loading
   python -c "from data_manager import load_data; df = load_data('data/btcusdt_1h.csv'); print(df.head())"
   ```

4. **Start Training**
   ```bash
   python train.py
   ```

---

## 📚 Additional Resources

- [CCXT Documentation](https://docs.ccxt.com/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/spot/en/)
- [Binance API FAQ](https://www.binance.com/en/support/faq/api)

---

## 🆘 Support

If you encounter issues:

1. Check this documentation
2. Review error messages carefully
3. Check Binance API status: https://www.binance.com/en/support/announcement
4. Open an issue on GitHub

---

**Happy Data Fetching! 🎉**
