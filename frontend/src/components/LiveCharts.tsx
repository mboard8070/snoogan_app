import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, type IChartApi, type ISeriesApi, CandlestickSeries } from 'lightweight-charts'
import type { CandlestickData, Time } from 'lightweight-charts'

const TICKERS = ['SPY', 'QQQ', 'IWM']

export default function LiveCharts() {
  const containerRefs = useRef<Record<string, HTMLDivElement | null>>({})
  const charts = useRef<Record<string, IChartApi>>({})
  const series = useRef<Record<string, ISeriesApi<'Candlestick'>>>({})
  const candleData = useRef<Record<string, CandlestickData<Time>[]>>({
    SPY: [],
    QQQ: [],
    IWM: [],
  })
  const wsRef = useRef<WebSocket | null>(null)
  const [prices, setPrices] = useState<Record<string, number>>({})
  const [connected, setConnected] = useState(false)
  const [initialized, setInitialized] = useState<Record<string, boolean>>({})

  const initializeChart = useCallback((ticker: string) => {
    const container = containerRefs.current[ticker]
    if (!container || charts.current[ticker]) return

    const width = container.clientWidth
    const height = container.clientHeight || 300

    if (width === 0) return // Container not visible yet

    const chart = createChart(container, {
      width,
      height,
      layout: {
        background: { color: '#1e293b' },
        textColor: '#9ca3af',
      },
      grid: {
        vertLines: { color: '#374151' },
        horzLines: { color: '#374151' },
      },
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    })

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    charts.current[ticker] = chart
    series.current[ticker] = candleSeries

    // If we have stored data, apply it now
    if (candleData.current[ticker].length > 0) {
      candleSeries.setData(candleData.current[ticker])
      chart.timeScale().fitContent()
    }

    setInitialized((prev) => ({ ...prev, [ticker]: true }))

    // Handle resize
    const resizeObserver = new ResizeObserver(() => {
      const newWidth = container.clientWidth
      if (newWidth > 0) {
        chart.applyOptions({ width: newWidth })
      }
    })
    resizeObserver.observe(container)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
    }
  }, [])

  // Initialize charts after component mounts
  useEffect(() => {
    // Small delay to ensure containers are rendered
    const timer = setTimeout(() => {
      TICKERS.forEach((ticker) => initializeChart(ticker))
    }, 100)

    return () => clearTimeout(timer)
  }, [initializeChart])

  // Re-initialize when tab becomes visible
  useEffect(() => {
    const observer = new MutationObserver(() => {
      TICKERS.forEach((ticker) => {
        const container = containerRefs.current[ticker]
        if (container && container.clientWidth > 0 && !charts.current[ticker]) {
          initializeChart(ticker)
        }
      })
    })

    observer.observe(document.body, { attributes: true, subtree: true, attributeFilter: ['class', 'style'] })

    return () => observer.disconnect()
  }, [initializeChart])

  useEffect(() => {
    const getWsUrl = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      return `${protocol}//${window.location.host}/api/ws/candles`
    }

    const connect = () => {
      const ws = new WebSocket(getWsUrl())
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        // Subscribe to all tickers
        TICKERS.forEach((ticker) => {
          ws.send(JSON.stringify({ type: 'subscribe', ticker }))
        })
      }

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          if (msg.type === 'history' && msg.candles) {
            const ticker = msg.ticker as string
            const candles: CandlestickData<Time>[] = msg.candles.map(
              (c: { time: number; open: number; high: number; low: number; close: number }) => ({
                time: c.time as Time,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
              })
            )

            // Store data in ref
            candleData.current[ticker] = candles

            // Apply to chart if initialized
            if (series.current[ticker]) {
              series.current[ticker].setData(candles)
              charts.current[ticker]?.timeScale().fitContent()
            }

            // Update current price
            if (candles.length > 0) {
              setPrices((prev) => ({
                ...prev,
                [ticker]: candles[candles.length - 1].close,
              }))
            }
          }
        } catch (err) {
          console.error('WebSocket message error:', err)
        }
      }

      ws.onclose = () => {
        setConnected(false)
        setTimeout(connect, 3000)
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      wsRef.current?.close()
    }
  }, [])

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-white">Live Charts</h2>
        <span className={`px-3 py-1 rounded-full text-sm ${connected ? 'bg-snoogans-green/20 text-snoogans-green' : 'bg-yellow-500/20 text-yellow-500'}`}>
          {connected ? 'Connected' : 'Connecting...'}
        </span>
      </div>

      <div className="grid gap-4">
        {TICKERS.map((ticker) => (
          <div key={ticker} className="bg-snoogans-card rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold text-white">{ticker}</h3>
              <span className="text-snoogans-accent font-mono">
                {prices[ticker] ? `$${prices[ticker].toFixed(2)}` : '—'}
              </span>
            </div>
            <div
              ref={(el) => {
                containerRefs.current[ticker] = el
              }}
              className="h-[300px] w-full"
            />
            {!initialized[ticker] && (
              <p className="text-gray-500 text-center py-4">Loading chart...</p>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
