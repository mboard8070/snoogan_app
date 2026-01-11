import { useState, useEffect, useRef } from 'react'

interface LogMessage {
  type: string
  data: {
    equity: number
    daily_pnl: number
    trade_count: number
    market_open: boolean
    positions_1m: unknown[]
    positions_15m: unknown[]
    positions_scalp: unknown[]
  }
}

interface TradingLogsProps {
  wsUrl?: string
}

export default function TradingLogs({ wsUrl }: TradingLogsProps) {
  const [logs, setLogs] = useState<string[]>([])
  const [status, setStatus] = useState<LogMessage['data'] | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const getWsUrl = () => {
    if (wsUrl) return wsUrl
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}/api/ws/logs`
  }

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(getWsUrl())
      wsRef.current = ws

      ws.onopen = () => {
        setConnected(true)
        setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] Connected to trading server`])
      }

      ws.onmessage = (event) => {
        try {
          const msg: LogMessage = JSON.parse(event.data)
          if (msg.type === 'state') {
            setStatus(msg.data)
            setLogs((prev) => [
              ...prev.slice(-99),
              `[${new Date().toLocaleTimeString()}] State update - Equity: $${msg.data.equity.toLocaleString()}, P&L: $${msg.data.daily_pnl}`,
            ])
          }
        } catch {
          setLogs((prev) => [...prev.slice(-99), event.data])
        }
      }

      ws.onclose = () => {
        setConnected(false)
        setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] Disconnected - reconnecting...`])
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
  }, [wsUrl])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const totalPositions =
    (status?.positions_1m?.length || 0) +
    (status?.positions_15m?.length || 0) +
    (status?.positions_scalp?.length || 0)

  return (
    <div className="space-y-4">
      {/* Status Panel */}
      <div className="bg-snoogans-card rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-3">Strategy Status</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-gray-400 text-xs">Market</p>
            <p className={`font-bold ${status?.market_open ? 'text-snoogans-green' : 'text-snoogans-red'}`}>
              {status?.market_open ? 'OPEN' : 'CLOSED'}
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">Daily P&L</p>
            <p className={`font-bold ${(status?.daily_pnl || 0) >= 0 ? 'text-snoogans-green' : 'text-snoogans-red'}`}>
              ${status?.daily_pnl?.toLocaleString() || '0'}
            </p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">Open Positions</p>
            <p className="font-bold text-white">{totalPositions}</p>
          </div>
          <div>
            <p className="text-gray-400 text-xs">Status</p>
            <p className={`font-bold ${connected ? 'text-snoogans-green' : 'text-yellow-500'}`}>
              {connected ? 'CONNECTED' : 'CONNECTING...'}
            </p>
          </div>
        </div>
      </div>

      {/* Log Output */}
      <div className="bg-snoogans-card rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-3">Live Logs</h3>
        <div className="bg-black rounded-lg p-3 h-96 overflow-y-auto font-mono text-sm">
          {logs.length === 0 ? (
            <p className="text-gray-500">Waiting for logs...</p>
          ) : (
            logs.map((log, i) => (
              <p key={i} className="text-green-400 whitespace-pre-wrap">
                {log}
              </p>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  )
}
