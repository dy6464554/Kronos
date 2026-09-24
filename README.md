# Kronos NSE Terminal for AI Studio

This application implements the reference Kronos NSE terminal features with Upstox replacing Zerodha Kite as the market-data provider.

## Required secret

Add a valid Upstox OAuth access token to AI Studio Secrets:

```text
UPSTOX_ACCESS_TOKEN=...
```

`UPSTOX_API_KEY` is not an access token and is intentionally not used as a bearer token. Upstox access tokens expire and must be refreshed according to your Upstox app configuration.

## Run

```bash
bash start_dev.sh
```

The server listens on `PORT` (default `3000`). The dashboard provides market-data fetching, Plotly candlesticks, ORB, EMA/RSI/MACD/VWAP/Bollinger/ATR/SuperTrend indicators, Kronos model loading, asynchronous forecasts, signals and walk-forward backtests.

The app does not place live orders. Forecasts and backtests are educational and are not financial advice.
