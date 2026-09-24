import express from "express";
import path from "path";
import cors from "cors";
import fetch from "node-fetch";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import { RSI, MACD, BollingerBands, VWAP, EMA } from "technicalindicators";

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Dictionary of NSE symbols -> Upstox/Yahoo Keys
const SYMBOL_MAP = {
  "NIFTY 50": { upstox: "NSE_INDEX|Nifty 50", yahoo: "^NSEI" },
  "BANK NIFTY": { upstox: "NSE_INDEX|Nifty Bank", yahoo: "^NSEBANK" },
  "RELIANCE": { upstox: "NSE_EQ|INE002A01018", yahoo: "RELIANCE.NS" },
  "TCS": { upstox: "NSE_EQ|INE467B01029", yahoo: "TCS.NS" },
  "HDFC BANK": { upstox: "NSE_EQ|INE040A01034", yahoo: "HDFCBANK.NS" },
  "INFY": { upstox: "NSE_EQ|INE009A01021", yahoo: "INFY.NS" },
  "ICICI BANK": { upstox: "NSE_EQ|INE090A01021", yahoo: "ICICIBANK.NS" }
};

import yfPackage from 'yahoo-finance2';
const YahooFinance = (yfPackage as any).default || yfPackage;
const yahooFinance = new YahooFinance();

// API routes FIRST
app.get("/api/symbols", (req, res) => {
  res.json({ symbols: Object.keys(SYMBOL_MAP), intervals: ["1minute", "5minute", "15minute", "60minute", "1d"] });
});

app.post("/api/fetch-data", async (req, res) => {
  const { symbol = "NIFTY 50", interval = "1minute" } = req.body;
  const mapping = SYMBOL_MAP[symbol as keyof typeof SYMBOL_MAP];
  
  if (!mapping) {
    return res.status(400).json({ error: "Invalid symbol" });
  }

  // Convert our UI interval to Yahoo interval
  const yfIntervalMap: Record<string, "1m" | "5m" | "15m" | "60m" | "1d"> = {
    "1minute": "1m",
    "5minute": "5m",
    "15minute": "15m",
    "60minute": "60m",
    "1d": "1d"
  };
  
  const yfInterval = yfIntervalMap[interval] || "1m";
  
  try {
    // 1. Primary Source: Yahoo Finance
    const period1 = new Date();
    if (yfInterval === "1m") {
        period1.setDate(period1.getDate() - 5); // 1m data is only available for 7 days
    } else {
        period1.setDate(period1.getDate() - 30);
    }
    
    const yfData = await yahooFinance.chart(mapping.yahoo, {
      interval: yfInterval,
      period1
    });

    if (yfData && yfData.quotes && yfData.quotes.length > 0) {
      const df = yfData.quotes
        .filter(q => q.open !== null && q.close !== null)
        .map(q => ({
          time: new Date(q.date).getTime() / 1000,
          open: q.open,
          high: q.high,
          low: q.low,
          close: q.close,
          volume: q.volume || 0
        }));
      
      return res.json({ status: "success", data: df, source: "yahoo-finance" });
    }
    throw new Error("Yahoo Finance returned empty data");
  } catch (yfError: any) {
    console.warn(`Yahoo Finance failed for ${symbol}: ${yfError.message}. Trying Upstox fallback...`);
    
    // 2. Backup Source: Upstox
    try {
      let upstoxInterval = interval === "1d" ? "day" : interval;
      let url = `https://api.upstox.com/v2/historical-candle/intraday/${encodeURIComponent(mapping.upstox)}/1minute`;
      if (upstoxInterval === "day") {
        url = `https://api.upstox.com/v2/historical-candle/${encodeURIComponent(mapping.upstox)}/day`;
      } else if (upstoxInterval !== "1minute") {
        url = `https://api.upstox.com/v2/historical-candle/${encodeURIComponent(mapping.upstox)}/${upstoxInterval}`;
      }
      
      const headers: any = { 'Accept': 'application/json' };
      if (process.env.UPSTOX_API_KEY) {
        headers['Authorization'] = `Bearer ${process.env.UPSTOX_API_KEY}`;
      }
      
      const upstoxRes = await fetch(url, { headers });
      const data: any = await upstoxRes.json();
      
      if (data.status === "success" && data.data && data.data.candles) {
        const rawCandles = data.data.candles.slice().reverse();
        const df = rawCandles.map((c: any) => ({
          time: new Date(c[0]).getTime() / 1000,
          open: parseFloat(c[1]),
          high: parseFloat(c[2]),
          low: parseFloat(c[3]),
          close: parseFloat(c[4]),
          volume: parseFloat(c[5])
        }));
        return res.json({ status: "success", data: df, source: "upstox" });
      } else {
        throw new Error(data.error?.message || "Upstox fetch failed");
      }
    } catch (upstoxError: any) {
      console.error("Upstox fallback also failed:", upstoxError.message);
      return res.status(500).json({ 
        status: "error", 
        message: `Failed to fetch data from both sources. Yahoo: ${yfError.message}, Upstox: ${upstoxError.message}` 
      });
    }
  }
});

// Calculate indicators
app.post("/api/indicators", (req, res) => {
  const { ohlcv } = req.body;
  if (!ohlcv || ohlcv.length === 0) return res.json({});
  
  try {
    const closes = ohlcv.map((c: any) => c.close);
    const highs = ohlcv.map((c: any) => c.high);
    const lows = ohlcv.map((c: any) => c.low);
    const volumes = ohlcv.map((c: any) => c.volume);
    
    // Padding arrays since indicators return shorter arrays
    const pad = (arr: number[], targetLen: number) => {
      const padding = new Array(targetLen - arr.length).fill(null);
      return [...padding, ...arr];
    };
    
    const rsi = pad(RSI.calculate({ period: 14, values: closes }), closes.length);
    const macdRes = MACD.calculate({ fastPeriod: 12, slowPeriod: 26, signalPeriod: 9, SimpleMAOscillator: false, SimpleMASignal: false, values: closes });
    const macd = pad(macdRes.map(m => m.MACD), closes.length);
    const ema9 = pad(EMA.calculate({ period: 9, values: closes }), closes.length);
    const ema21 = pad(EMA.calculate({ period: 21, values: closes }), closes.length);
    
    // VWAP
    const vwapRes = VWAP.calculate({
      high: highs,
      low: lows,
      close: closes,
      volume: volumes
    });
    const vwap = pad(vwapRes, closes.length);
    
    res.json({
      RSI: rsi,
      MACD: macd,
      EMA9: ema9,
      EMA21: ema21,
      VWAP: vwap
    });
  } catch (err: any) {
    console.error("Indicator error:", err);
    res.json({});
  }
});

app.post("/api/predict", async (req, res) => {
  const { symbol, recent_data } = req.body;
  
  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ status: "error", error: "GEMINI_API_KEY is required to generate forecasts. Please configure it in your environment secrets." });
  }
  
  try {
    const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
    const prompt = `
      You are the Kronos AI, a quantitative financial forecasting model.
      Analyze the recent price data for ${symbol} and provide a trading signal.
      Recent candles (Close prices): ${recent_data.map((c: any) => c.close).slice(-10).join(", ")}
      
      Respond with ONLY a strict JSON object (no markdown, no backticks) with this structure:
      {
        "direction": "BUY" or "SELL",
        "confidence": <number 0-100>,
        "entry": <number>,
        "stop_loss": <number>,
        "target": <number>,
        "reasoning": ["point 1", "point 2"]
      }
    `;
    
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-pro',
      contents: prompt
    });
    
    const text = response.text || "{}";
    const cleaned = text.replace(/```json/g, "").replace(/```/g, "").trim();
    const prediction = JSON.parse(cleaned);
    
    res.json({ status: "success", prediction });
  } catch (error: any) {
    console.error("Kronos/Gemini Error:", error);
    res.status(500).json({ status: "error", error: error.message });
  }
});

// Vite middleware for development
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
