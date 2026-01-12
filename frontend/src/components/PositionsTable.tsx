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
  trail_level?: number
  entry_time?: string
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
  trail_level?: number
  entry_time?: string
}

type Position = CreditSpreadPosition & ScalpPosition & { ticker?: string }

interface PositionsTableProps {
  title: string
  positions: Record<string, Position>
  isScalp?: boolean     // Flag to distinguish position type
}

export function PositionsTable({ title, positions, isScalp = false }: PositionsTableProps) {
  const positionList = Object.entries(positions).map(([ticker, pos]) => ({
    ticker,
    ...pos,
  }))

  if (positionList.length === 0) {
    return (
      <div className="bg-surface rounded-lg p-6 shadow-lg">
        <h3 className="text-xl font-bold mb-4">{title}</h3>
        <p className="text-gray-500">No active positions</p>
      </div>
    )
  }

  // Calculate P&L for credit spreads: (credit - best_value) * contracts * 100
  const calcCreditSpreadPnL = (pos: Position) => {
    const credit = pos.credit ?? 0
    const bestValue = pos.best_value ?? pos.current_value ?? credit
    const contracts = pos.contracts ?? 0
    return (credit - bestValue) * contracts * 100
  }

  // Calculate P&L for scalp: (best_mark - debit) * contracts * 100
  const calcScalpPnL = (pos: Position) => {
    const debit = pos.debit ?? 0
    const bestMark = pos.best_mark ?? pos.current_mark ?? debit
    const contracts = pos.contracts ?? 0
    return (bestMark - debit) * contracts * 100
  }

  // Calculate P&L % for scalp: ((best_mark - debit) / debit) * 100
  const calcScalpPnLPercent = (pos: Position) => {
    const debit = pos.debit ?? 0
    if (debit <= 0) return 0
    const bestMark = pos.best_mark ?? pos.current_mark ?? debit
    return ((bestMark - debit) / debit) * 100
  }

  // Render credit spread table (1m & 15m strategies)
  if (!isScalp) {
    return (
      <div className="bg-surface rounded-lg p-6 shadow-lg">
        <h3 className="text-xl font-bold mb-4">{title}</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="pb-2">Ticker</th>
                <th className="pb-2">Type</th>
                <th className="pb-2">Strikes</th>
                <th className="pb-2">Credit</th>
                <th className="pb-2">Contracts</th>
                <th className="pb-2">P/L</th>
                <th className="pb-2">Trail</th>
              </tr>
            </thead>
            <tbody>
              {positionList.map((pos) => {
                const pnl = calcCreditSpreadPnL(pos)
                return (
                  <tr key={pos.ticker} className="border-b border-gray-800">
                    <td className="py-2 font-bold">{pos.ticker}</td>
                    <td className="py-2">{pos.is_put ? 'PUT' : 'CALL'}</td>
                    <td className="py-2">{pos.short?.toFixed(0) || '-'}/{pos.long?.toFixed(0) || '-'}</td>
                    <td className="py-2">${pos.credit?.toFixed(2) || '0.00'}</td>
                    <td className="py-2">{pos.contracts || 0}</td>
                    <td className={`py-2 font-medium ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                    </td>
                    <td className="py-2">{pos.trail_active ? '✓' : '-'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    )
  }

  // Render scalp table (long options)
  return (
    <div className="bg-surface rounded-lg p-6 shadow-lg">
      <h3 className="text-xl font-bold mb-4">{title}</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="pb-2">Ticker</th>
              <th className="pb-2">Type</th>
              <th className="pb-2">Strike</th>
              <th className="pb-2">Debit</th>
              <th className="pb-2">Contracts</th>
              <th className="pb-2">P/L</th>
              <th className="pb-2">P/L %</th>
              <th className="pb-2">Trail</th>
            </tr>
          </thead>
          <tbody>
            {positionList.map((pos) => {
              const pnl = calcScalpPnL(pos)
              const pnlPct = calcScalpPnLPercent(pos)
              return (
                <tr key={pos.ticker} className="border-b border-gray-800">
                  <td className="py-2 font-bold">{pos.ticker}</td>
                  <td className="py-2">{pos.is_call ? 'CALL' : 'PUT'}</td>
                  <td className="py-2">{pos.strike_price?.toFixed(0) || '-'}</td>
                  <td className="py-2">${pos.debit?.toFixed(2) || '0.00'}</td>
                  <td className="py-2">{pos.contracts || 0}</td>
                  <td className={`py-2 font-medium ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                  </td>
                  <td className={`py-2 font-medium ${pnlPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {pnlPct >= 0 ? '+' : ''}{pnlPct.toFixed(1)}%
                  </td>
                  <td className="py-2">{pos.trail_active ? '✓' : '-'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
