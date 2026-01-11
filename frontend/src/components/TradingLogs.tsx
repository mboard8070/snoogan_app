import { useState, useEffect, useRef } from 'react'

interface Position {
  ticker?: string
  is_call?: boolean
  strike_price?: number
  debit?: number
  contracts?: number
  best_mark?: number
  unrealized?: number
  trail_active?: boolean
  trail_level?: number
  entry_time?: string
}

interface StrategyStatus {
  daily_pnl: number
  daily_loss_limit: number
  trades_today: number
  can_trade: boolean
  current_trends: Record<string, string>
}

interface LogMessage {
  type: string
  data: {
    equity: number
    daily_pnl: number
    trade_count: number
    market_open: boolean
    positions_1m: Record<string, Position>
    positions_15m: Record<string, Position>
    positions_scalp: Record<string, Position>
    status_1m: StrategyStatus
    status_15m: StrategyStatus
    status_scalp: StrategyStatus
  }
}

interface TradingLogsProps {
  wsUrl?: string
}

const STRATEGIES = ['1m', '15m', 'scalp'] as const
type StrategyType = typeof STRATEGIES[number]

export default function TradingLogs({ wsUrl }: TradingLogsProps) {
  const [logs, setLogs] = useState<Record<StrategyType, string[]>>({
    '1m': [],
    '15m': [],
    'scalp': [],
  })
  const [status, setStatus] = useState<LogMessage['data'] | null>(null)
  const [connected, setConnected] = useState(false)
  const [activeStrategy, setActiveStrategy] = useState<StrategyType>('1m')
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
        const timestamp = new Date().toLocaleTimeString()
        setLogs((prev) => ({
          '1m': [...prev['1m'], `[${timestamp}] Connected to trading server`],
          '15m': [...prev['15m'], `[${timestamp}] Connected to trading server`],
          'scalp': [...prev['scalp'], `[${timestamp}] Connected to trading server`],
        }))
      }

      ws.onmessage = (event) => {
        try {
          const msg: LogMessage = JSON.parse(event.data)
          if (msg.type === 'state') {
            setStatus(msg.data)
            const timestamp = new Date().toLocaleTimeString()

            const pos1m = Object.keys(msg.data.positions_1m || {}).length
            const pos15m = Object.keys(msg.data.positions_15m || {}).length
            const posScalp = Object.keys(msg.data.positions_scalp || {}).length

            setLogs((prev) => ({
              '1m': [...prev['1m'].slice(-99), `[${timestamp}] 1m Strategy - ${pos1m} positions, P&L: $${msg.data.daily_pnl}`],
              '15m': [...prev['15m'].slice(-99), `[${timestamp}] 15m Strategy - ${pos15m} positions`],
              'scalp': [...prev['scalp'].slice(-99), `[${timestamp}] Scalp Strategy - ${posScalp} positions`],
            }))
          }
        } catch {
          // Ignore parse errors
        }
      }

      ws.onclose = () => {
        setConnected(false)
        const timestamp = new Date().toLocaleTimeString()
        setLogs((prev) => ({
          '1m': [...prev['1m'], `[${timestamp}] Disconnected - reconnecting...`],
          '15m': [...prev['15m'], `[${timestamp}] Disconnected - reconnecting...`],
          'scalp': [...prev['scalp'], `[${timestamp}] Disconnected - reconnecting...`],
        }))
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
  }, [logs, activeStrategy])

  const getPositionsForStrategy = (strategy: StrategyType): Record<string, Position> => {
    if (!status) return {}
    switch (strategy) {
      case '1m': return status.positions_1m || {}
      case '15m': return status.positions_15m || {}
      case 'scalp': return status.positions_scalp || {}
    }
  }

  const getStatusForStrategy = (strategy: StrategyType): StrategyStatus | null => {
    if (!status) return null
    switch (strategy) {
      case '1m': return status.status_1m || null
      case '15m': return status.status_15m || null
      case 'scalp': return status.status_scalp || null
    }
  }

  const positions = getPositionsForStrategy(activeStrategy)
  const positionList = Object.entries(positions).map(([ticker, pos]) => ({ ticker, ...pos }))
  const strategyStatus = getStatusForStrategy(activeStrategy)
  const trends = strategyStatus?.current_trends || {}

  return (
    <div className="space-y-4">
      {/* Status Panel - Full stats like Streamlit */}
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        {/* Top row: P&L and limits */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-4">
          <div>
            <p className="text-xs text-gray-400 uppercase">Daily P&L</p>
            <p className={`text-xl font-bold ${(strategyStatus?.daily_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${(strategyStatus?.daily_pnl || 0) >= 0 ? '+' : ''}{(strategyStatus?.daily_pnl || 0).toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase">Loss Limit</p>
            <p className="text-xl font-bold text-red-400">
              ${(strategyStatus?.daily_loss_limit || 0).toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase">Equity</p>
            <p className="text-xl font-bold text-accent">${status?.equity?.toLocaleString() || '0'}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase">Trades Today</p>
            <p className="text-xl font-bold text-accent">{strategyStatus?.trades_today || 0}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase">Can Trade</p>
            <p className={`text-xl font-bold ${strategyStatus?.can_trade ? 'text-green-400' : 'text-red-400'}`}>
              {strategyStatus?.can_trade ? 'Yes' : 'No'}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase">Status</p>
            <p className={`text-xl font-bold ${connected ? 'text-green-400' : 'text-yellow-400'}`}>
              {connected ? 'LIVE' : 'CONNECTING'}
            </p>
          </div>
        </div>

        {/* Trends row */}
        <div className="border-t border-gray-700 pt-3">
          <p className="text-xs text-gray-400 uppercase mb-2">Trends</p>
          <div className="flex gap-4">
            {['SPY', 'QQQ', 'IWM'].map((ticker) => {
              const trend = trends[ticker] || '---'
              const color = trend === 'bull' ? 'text-green-400' : trend === 'bear' ? 'text-red-400' : 'text-yellow-400'
              return (
                <div key={ticker} className="flex items-center gap-2">
                  <span className="text-gray-400 font-bold">{ticker}:</span>
                  <span className={`font-bold uppercase ${color}`}>{trend}</span>
                </div>
              )
            })}
          </div>
        </div>
      </div>

      {/* Strategy Tabs */}
      <div className="flex border-b border-gray-700">
        {STRATEGIES.map((strategy) => {
          const posCount = Object.keys(getPositionsForStrategy(strategy)).length
          const isActive = activeStrategy === strategy
          return (
            <button
              key={strategy}
              onClick={() => setActiveStrategy(strategy)}
              className={`px-5 py-2.5 font-semibold transition-all duration-200 relative flex items-center gap-2 ${
                isActive
                  ? 'text-[#f87171]'
                  : 'text-[#4ade80] hover:text-[#86efac]'
              }`}
            >
              <span>{strategy === '1m' ? '1 Minute' : strategy === '15m' ? '15 Minute' : 'Scalp'}</span>
              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                isActive ? 'bg-red-900/50 text-red-300' : 'bg-green-900/50 text-green-300'
              }`}>
                {posCount}
              </span>
              {isActive && (
                <span className="absolute bottom-0 left-0 right-0 h-[3px] bg-[#f87171] rounded-t" />
              )}
            </button>
          )
        })}
      </div>

      {/* Content Grid - positions and logs side by side on larger screens */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Positions Table */}
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-bold mb-3">
            {activeStrategy === '1m' ? '1 Minute' : activeStrategy === '15m' ? '15 Minute' : 'Scalp'} Positions
          </h3>
          {positionList.length === 0 ? (
            <p className="text-gray-500 text-sm">No active positions</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
                    <th className="pb-2 text-left">Ticker</th>
                    <th className="pb-2 text-left">Type</th>
                    <th className="pb-2 text-right">Strike</th>
                    <th className="pb-2 text-right">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {positionList.map((pos) => (
                    <tr key={pos.ticker} className="border-b border-gray-800">
                      <td className="py-2 font-bold">{pos.ticker}</td>
                      <td className="py-2">{pos.is_call ? 'CALL' : 'PUT'}</td>
                      <td className="py-2 text-right">${pos.strike_price?.toFixed(0) || '-'}</td>
                      <td className={`py-2 text-right font-bold ${(pos.unrealized || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${pos.unrealized?.toFixed(2) || '0.00'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Log Output */}
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-bold mb-3">Live Logs</h3>
          <div className="bg-background rounded p-3 h-64 overflow-y-auto font-mono text-xs">
            {logs[activeStrategy].length === 0 ? (
              <p className="text-gray-500">Waiting for logs...</p>
            ) : (
              logs[activeStrategy].map((log, i) => (
                <p key={i} className="text-accent mb-1">
                  {log}
                </p>
              ))
            )}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>
    </div>
  )
}
