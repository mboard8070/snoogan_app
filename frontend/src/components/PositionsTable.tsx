interface Position {
  ticker: string
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

interface PositionsTableProps {
  title: string
  positions: Record<string, Position>
}

export function PositionsTable({ title, positions }: PositionsTableProps) {
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
              <th className="pb-2">Best Mark</th>
              <th className="pb-2">Unrealized</th>
            </tr>
          </thead>
          <tbody>
            {positionList.map((pos) => (
              <tr key={pos.ticker} className="border-b border-gray-800">
                <td className="py-2">{pos.ticker}</td>
                <td className="py-2">{pos.is_call ? 'CALL' : 'PUT'}</td>
                <td className="py-2">{pos.strike_price?.toFixed(1) || '-'}</td>
                <td className="py-2">${pos.debit?.toFixed(2) || '0.00'}</td>
                <td className="py-2">{pos.contracts || 0}</td>
                <td className="py-2">${pos.best_mark?.toFixed(2) || '0.00'}</td>
                <td className={`py-2 font-medium ${pos.unrealized && pos.unrealized >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${pos.unrealized?.toFixed(2) || '0.00'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
