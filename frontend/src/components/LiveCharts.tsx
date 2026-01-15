import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, type IChartApi, type ISeriesApi, CandlestickSeries } from 'lightweight-charts'
import type { CandlestickData, Time } from 'lightweight-charts'

const TICKERS = ['SPY', 'QQQ', 'IWM'] as const
type TickerType = typeof TICKERS[number]

export default function LiveCharts() {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const candleData = useRef<Record<TickerType, CandlestickData<Time>[]>>({
    SPY: [],
    QQQ: [],
    IWM: [],
  })
  const wsRef = useRef<WebSocket | null>(null)
  const mountedRef = useRef(true)
  const activeTickerRef = useRef<TickerType>('SPY')
  const [prices, setPrices] = useState<Record<string, number>>({})
  const [connected, setConnected] = useState(false)
  const [activeTicker, setActiveTicker] = useState<TickerType>('SPY')
  const [chartInitialized, setChartInitialized] = useState(false)

  // Keep activeTickerRef in sync
  useEffect(() => {
    activeTickerRef.current = activeTicker
  }, [activeTicker])

  const initializeChart = useCallback(() => {
    const container = containerRef.current
    if (!container) return

    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
      seriesRef.current = null
    }

    const width = container.clientWidth
    const height = 500

    if (width === 0) return

    const chart = createChart(container, {
      width,
      height,
      layout: {
        background: { color: '#1a1a2e' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#333' },
        horzLines: { color: '#333' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#00ff9d',
      downColor: '#ef4444',
      borderUpColor: '#00ff9d',
      borderDownColor: '#ef4444',
      wickUpColor: '#00ff9d',
      wickDownColor: '#ef4444',
    })

    chartRef.current = chart
    seriesRef.current = candleSeries
    setChartInitialized(true)

    if (candleData.current[activeTicker].length > 0) {
      candleSeries.setData(candleData.current[activeTicker])
      chart.timeScale().fitContent()
    }

    const resizeObserver = new ResizeObserver(() => {
      const newWidth = container.clientWidth
      if (newWidth > 0 && chartRef.current) {
        chartRef.current.applyOptions({ width: newWidth })
      }
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
    }
  }, [activeTicker])

  useEffect(() => {
    const timer = setTimeout(() => {
      initializeChart()
    }, 100)

    return () => {
      clearTimeout(timer)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
        seriesRef.current = null
      }
    }
  }, [initializeChart])

  useEffect(() => {
    if (seriesRef.current && candleData.current[activeTicker].length > 0) {
      seriesRef.current.setData(candleData.current[activeTicker])
      chartRef.current?.timeScale().fitContent()
    }
  }, [activeTicker])

  useEffect(() => {
    mountedRef.current = true
    let connectTimeout: ReturnType<typeof setTimeout> | null = null

    const getWsUrl = () => {
      // Connect directly to backend for WebSocket (Vite proxy doesn't handle WS well)
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.hostname
      return `${protocol}//${host}:8000/api/ws/candles`
    }

    const connect = () => {
      if (!mountedRef.current) return

      const ws = new WebSocket(getWsUrl())
      wsRef.current = ws

      ws.onopen = () => {
        if (!mountedRef.current) {
          ws.close()
          return
        }
        setConnected(true)
        TICKERS.forEach((ticker) => {
          ws.send(JSON.stringify({ type: 'subscribe', ticker }))
        })
      }

      ws.onmessage = (event) => {
        if (!mountedRef.current) return
        try {
          const msg = JSON.parse(event.data)

          // Handle real-time price updates
          if (msg.type === 'prices' && msg.data) {
            setPrices((prev) => ({
              ...prev,
              ...msg.data,
            }))
            return
          }

          // Handle historical candle data
          if (msg.type === 'history' && msg.candles) {
            const ticker = msg.ticker as TickerType
            const candles: CandlestickData<Time>[] = msg.candles.map(
              (c: { time: number; open: number; high: number; low: number; close: number }) => ({
                time: c.time as Time,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
              })
            )

            candleData.current[ticker] = candles

            // Use ref to get current active ticker (avoids stale closure)
            if (ticker === activeTickerRef.current && seriesRef.current) {
              seriesRef.current.setData(candles)
              chartRef.current?.timeScale().fitContent()
            }

            // Only set price from candles if we don't have a real-time price yet
            if (candles.length > 0) {
              setPrices((prev) => {
                // Don't overwrite if we already have a real-time price
                if (prev[ticker]) return prev
                return {
                  ...prev,
                  [ticker]: candles[candles.length - 1].close,
                }
              })
            }
          }
        } catch (err) {
          console.error('WebSocket message error:', err)
        }
      }

      ws.onclose = () => {
        setConnected(false)
        // Only reconnect if still mounted
        if (mountedRef.current) {
          setTimeout(connect, 3000)
        }
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    // Delay connection slightly to handle React StrictMode double-mount
    connectTimeout = setTimeout(connect, 100)

    return () => {
      mountedRef.current = false
      if (connectTimeout) clearTimeout(connectTimeout)
      wsRef.current?.close()
    }
  }, []) // Empty dependency array - WebSocket connects once and stays connected

  return (
    <div className="space-y-4">
      {/* Header with ticker tabs and connection status */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        {/* Ticker Tabs */}
        <div className="flex border-b border-gray-700">
          {TICKERS.map((ticker) => {
            const isActive = activeTicker === ticker
            const price = prices[ticker]
            return (
              <button
                key={ticker}
                onClick={() => setActiveTicker(ticker)}
                className={`px-5 py-2.5 font-semibold transition-all duration-200 relative ${
                  isActive
                    ? 'text-[#f87171]'
                    : 'text-[#4ade80] hover:text-[#86efac]'
                }`}
              >
                <span className="text-lg">{ticker}</span>
                {price && (
                  <span className={`ml-2 font-mono text-sm ${isActive ? 'text-red-300' : 'text-green-300'}`}>
                    ${price.toFixed(2)}
                  </span>
                )}
                {isActive && (
                  <span className="absolute bottom-0 left-0 right-0 h-[3px] bg-[#f87171] rounded-t" />
                )}
              </button>
            )
          })}
        </div>

        {/* Connection Status */}
        <div className={`px-4 py-2 rounded-lg font-bold text-sm border-2 ${
          connected
            ? 'bg-green-900/50 text-green-400 border-green-600'
            : 'bg-yellow-900/50 text-yellow-400 border-yellow-600'
        }`}>
          {connected ? 'CONNECTED' : 'CONNECTING...'}
        </div>
      </div>

      {/* Chart Container */}
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xl font-bold">{activeTicker}</h3>
          <span className="text-accent font-mono text-2xl font-bold">
            {prices[activeTicker] ? `$${prices[activeTicker].toFixed(2)}` : 'Loading...'}
          </span>
        </div>
        <div
          ref={containerRef}
          className="w-full rounded overflow-hidden"
          style={{ height: '500px' }}
        />
        {!chartInitialized && (
          <p className="text-gray-500 text-center py-8">Loading chart...</p>
        )}
      </div>
    </div>
  )
}
