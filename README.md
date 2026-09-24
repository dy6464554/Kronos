# Kronos NSE Terminal — Upstox edition

This is an AI Studio-compatible NSE analytics terminal based on the reference `kronos-nse-terminal` project. Market data is fetched exclusively from Upstox; Zerodha/Kite is not used.

## Security

The Upstox JWT/access token pasted in chat is a credential. Revoke/rotate it immediately in the Upstox developer console and create a new token. Do not commit tokens or put them in source files.

Configure the replacement token in AI Studio Secrets:

```text
UPSTOX_ACCESS_TOKEN=your_new_upstox_access_token
```

## Run

```bash
bash start_dev.sh
```

The server uses `PORT` when supplied and otherwise listens on port 3000. Docker is also supported with `docker build -t kronos-nse .` and `docker run -p 3000:3000 -e UPSTOX_ACCESS_TOKEN=... kronos-nse`.

The terminal provides Upstox OHLCV fetching, candlesticks, ORB, EMA/RSI/MACD/VWAP/Bollinger/ATR/SuperTrend indicators, asynchronous forecasting, signals and walk-forward backtesting. It does not place live orders.
