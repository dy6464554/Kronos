import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, Time, CandlestickSeries } from 'lightweight-charts';

interface ChartProps {
  data: any[];
  indicators: any;
  symbol: string;
}

export default function Chart({ data, indicators, symbol }: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: '#0d0f1a' },
        textColor: '#c9d1d9',
      },
      grid: {
        vertLines: { color: '#21262d' },
        horzLines: { color: '#21262d' },
      },
      crosshair: {
        mode: 1,
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#21262d',
      },
    });

    chartRef.current = chart;

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#26de81',
      downColor: '#ff4757',
      borderVisible: false,
      wickUpColor: '#26de81',
      wickDownColor: '#ff4757',
    });

    seriesRef.current = candlestickSeries;

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Initial sizing

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  useEffect(() => {
    if (seriesRef.current && data.length > 0) {
      seriesRef.current.setData(data as any);
      
      // If we wanted to draw indicators like EMA or VWAP here we would add line series.
      // We will skip drawing indicators on the chart directly for simplicity, but could be added.
    }
  }, [data]);

  return (
    <div className="w-full h-full relative" ref={chartContainerRef}>
      <div className="absolute top-4 left-4 z-10 text-white font-mono text-xl opacity-50 pointer-events-none">
        {symbol}
      </div>
    </div>
  );
}
