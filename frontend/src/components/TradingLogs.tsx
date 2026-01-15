import { useState, useEffect, useRef } from 'react'

// Credit spread position (1m & 15m strategies)
interface CreditSpreadPosition {
  is_put?: boolean
  short?: number        // Short strike
  long?: number         // Long strike
  credit?: number       // Credit received
  contracts?: number
  current_value?: number
  best_value?: number
  trail_active?: boolean
}

// Scalp position (long options)
interface ScalpPosition {
  is_call?: boolean
  strike_price?: number
  debit?: number        // Debit paid
  contracts?: number
  current_mark?: number
  best_mark?: number
  trail_active?: boolean
}

type Position = CreditSpreadPosition & ScalpPosition & { ticker?: string }

interface StrategyStatus {
  daily_pnl: number
  daily_loss_limit: number
  trades_today: number
  can_trade: boolean
  current_trends: Record<string, string>
}

interface StateMessage {
  type: 'state'
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

interface TradeEvent {
  type: 'trade'
  data: {
    timestamp: string
    strategy: '1m' | '15m' | 'scalp'
    event: 'entry' | 'exit'
    ticker: string
    type?: string      // CALL or PUT
    strikes?: string   // For credit spreads
    strike?: number    // For scalp
    credit?: number    // For credit spreads
    debit?: number     // For scalp
    contracts?: number
  }
}

type LogMessage = StateMessage | TradeEvent

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
  const [status, setStatus] = useState<StateMessage['data'] | null>(null)
  const [connected, setConnected] = useState(false)
  const [activeStrategy, setActiveStrategy] = useState<StrategyType>('1m')
  const wsRef = useRef<WebSocket | null>(null)
  const mountedRef = useRef(true)
  const logsEndRef = useRef<HTMLDivElement>(null)

  const getWsUrl = () => {
    if (wsUrl) return wsUrl
    // Connect directly to backend for WebSocket (Vite proxy doesn't handle WS well)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.hostname
    return `${protocol}//${host}:8000/api/ws/logs`
  }

  useEffect(() => {
    mountedRef.current = true
    let connectTimeout: ReturnType<typeof setTimeout> | null = null

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
        const timestamp = new Date().toLocaleTimeString()
        setLogs((prev) => ({
          '1m': [...prev['1m'], `[${timestamp}] Connected to trading server`],
          '15m': [...prev['15m'], `[${timestamp}] Connected to trading server`],
          'scalp': [...prev['scalp'], `[${timestamp}] Connected to trading server`],
        }))
      }

      ws.onmessage = (event) => {
        if (!mountedRef.current) return
        try {
          const msg: LogMessage = JSON.parse(event.data)

          // Handle trade events from backend
          if (msg.type === 'trade') {
            const trade = msg.data
            const time = new Date(trade.timestamp).toLocaleTimeString()
            const strategy = trade.strategy as StrategyType

            let logEntry: string
            if (trade.event === 'entry') {
              if (strategy === 'scalp') {
                logEntry = `[${time}] ENTRY: ${trade.ticker} ${trade.type} ${trade.strike?.toFixed(0) || '?'} @ $${trade.debit?.toFixed(2) || '?'} x${trade.contracts || '?'}`
              } else {
                logEntry = `[${time}] ENTRY: ${trade.ticker} ${trade.type} spread ${trade.strikes || '?'} @ $${trade.credit?.toFixed(2) || '?'} x${trade.contracts || '?'}`
              }
            } else {
              logEntry = `[${time}] EXIT: ${trade.ticker} position closed`
            }

            setLogs((prev) => ({
              ...prev,
              [strategy]: [...prev[strategy].slice(-99), logEntry],
            }))
            return
          }

          // Handle state updates
          if (msg.type === 'state') {
            setStatus(msg.data)
            const timestamp = new Date().toLocaleTimeString()

            // Get current position counts for status line
            const pos1m = Object.keys(msg.data.positions_1m || {}).length
            const pos15m = Object.keys(msg.data.positions_15m || {}).length
            const posScalp = Object.keys(msg.data.positions_scalp || {}).length

            // Get P&L for each strategy
            const pnl1m = msg.data.status_1m?.daily_pnl ?? msg.data.daily_pnl ?? 0
            const pnl15m = msg.data.status_15m?.daily_pnl ?? 0
            const pnlScalp = msg.data.status_scalp?.daily_pnl ?? 0

            setLogs((prev) => ({
              '1m': [...prev['1m'].slice(-99), `[${timestamp}] Status: ${pos1m} positions | P&L: $${pnl1m >= 0 ? '+' : ''}${pnl1m.toFixed(2)}`],
              '15m': [...prev['15m'].slice(-99), `[${timestamp}] Status: ${pos15m} positions | P&L: $${pnl15m >= 0 ? '+' : ''}${pnl15m.toFixed(2)}`],
              'scalp': [...prev['scalp'].slice(-99), `[${timestamp}] Status: ${posScalp} positions | P&L: $${pnlScalp >= 0 ? '+' : ''}${pnlScalp.toFixed(2)}`],
            }))
          }
        } catch {
          // Ignore parse errors
        }
      }

      ws.onclose = () => {
        setConnected(false)
        // Only reconnect if still mounted
        if (mountedRef.current) {
          const timestamp = new Date().toLocaleTimeString()
          setLogs((prev) => ({
            '1m': [...prev['1m'], `[${timestamp}] Disconnected - reconnecting...`],
            '15m': [...prev['15m'], `[${timestamp}] Disconnected - reconnecting...`],
            'scalp': [...prev['scalp'], `[${timestamp}] Disconnected - reconnecting...`],
          }))
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

  // Calculate total daily P&L from all strategies
  const totalDailyPnl = (status?.status_1m?.daily_pnl || 0) +
                        (status?.status_15m?.daily_pnl || 0) +
                        (status?.status_scalp?.daily_pnl || 0)

  return (
    <div className="space-y-4">
      {/* Status Panel - Full stats like Streamlit */}
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        {/* Top row: P&L and limits */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-4">
          <div>
            <p className="text-xs text-gray-400 uppercase">Daily P&L (Total)</p>
            <p className={`text-xl font-bold ${totalDailyPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${totalDailyPnl >= 0 ? '+' : ''}{totalDailyPnl.toFixed(2)}
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
          ) : activeStrategy === 'scalp' ? (
            /* Scalp positions table (long options) */
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
                    <th className="pb-2 text-left">Ticker</th>
                    <th className="pb-2 text-left">Type</th>
                    <th className="pb-2 text-right">Strike</th>
                    <th className="pb-2 text-right">Debit</th>
                    <th className="pb-2 text-right">P&L</th>
                    <th className="pb-2 text-right">P&L %</th>
                    <th className="pb-2 text-center">Trail</th>
                  </tr>
                </thead>
                <tbody>
                  {positionList.map((pos) => {
                    const debit = pos.debit ?? 0
                    const bestMark = pos.best_mark ?? pos.current_mark ?? debit
                    const contracts = pos.contracts ?? 0
                    const pnl = (bestMark - debit) * contracts * 100
                    const pnlPct = debit > 0 ? ((bestMark - debit) / debit) * 100 : 0
                    return (
                      <tr key={pos.ticker} className="border-b border-gray-800">
                        <td className="py-2 font-bold">{pos.ticker}</td>
                        <td className="py-2">{pos.is_call ? 'CALL' : 'PUT'}</td>
                        <td className="py-2 text-right">{pos.strike_price?.toFixed(0) || '-'}</td>
                        <td className="py-2 text-right">${debit.toFixed(2)}</td>
                        <td className={`py-2 text-right font-bold ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                        </td>
                        <td className={`py-2 text-right font-bold ${pnlPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%
                        </td>
                        <td className="py-2 text-center">{pos.trail_active ? '✓' : '-'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            /* Credit spread positions table (1m & 15m strategies) */
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
                    <th className="pb-2 text-left">Ticker</th>
                    <th className="pb-2 text-left">Type</th>
                    <th className="pb-2 text-right">Strikes</th>
                    <th className="pb-2 text-right">Credit</th>
                    <th className="pb-2 text-right">Qty</th>
                    <th className="pb-2 text-right">P&L</th>
                    <th className="pb-2 text-center">Trail</th>
                  </tr>
                </thead>
                <tbody>
                  {positionList.map((pos) => {
                    const credit = pos.credit ?? 0
                    const bestValue = pos.best_value ?? pos.current_value ?? credit
                    const contracts = pos.contracts ?? 0
                    const pnl = (credit - bestValue) * contracts * 100
                    return (
                      <tr key={pos.ticker} className="border-b border-gray-800">
                        <td className="py-2 font-bold">{pos.ticker}</td>
                        <td className="py-2">{pos.is_put ? 'PUT' : 'CALL'}</td>
                        <td className="py-2 text-right">{pos.short?.toFixed(0) || '-'}/{pos.long?.toFixed(0) || '-'}</td>
                        <td className="py-2 text-right">${credit.toFixed(2)}</td>
                        <td className="py-2 text-right">{contracts}</td>
                        <td className={`py-2 text-right font-bold ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                        </td>
                        <td className="py-2 text-center">{pos.trail_active ? '✓' : '-'}</td>
                      </tr>
                    )
                  })}
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
