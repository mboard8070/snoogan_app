import { useState, useEffect } from 'react'

interface LearnerStats {
  total_states: number
  total_trades: number
  total_skips: number
  win_rate: number
  total_pnl: number
  avg_pnl: number
  epsilon: number
  exploration_rate: number
  counterfactual_good_skips: number
  counterfactual_bad_skips: number
  skip_accuracy: number
  pending_counterfactuals: number
}

interface StateInfo {
  state: string
  q_enter: number
  q_skip: number
  advantage: number
}

interface Decision {
  timestamp: string
  state: string
  action: string
  reward?: number
  explored?: boolean
}

interface DecisionSummary {
  total: number
  enters: number
  skips: number
  enter_rate: number
  explorations: number
  exploration_rate: number
}

interface LearnerData {
  stats: LearnerStats
  decisions: DecisionSummary
  best_states: StateInfo[]
  worst_states: StateInfo[]
  recent_decisions: Decision[]
}

// Parse state string like "SPY|bull|mid|mid_morning|medium" into components
function parseState(state: string): { ticker: string; trend: string; rsi: string; hour: string; conf: string } {
  const parts = state.split('|')
  return {
    ticker: parts[0] || '?',
    trend: parts[1] || '?',
    rsi: parts[2] || '?',
    hour: parts[3] || '?',
    conf: parts[4] || '?',
  }
}

export default function RLLearner() {
  const [data, setData] = useState<LearnerData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('/api/learner')
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`)
        const json = await res.json()
        setData(json)
        setError(null)
      } catch (err) {
        console.error('Learner fetch error:', err)
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-surface rounded-lg p-6 text-center border border-gray-700">
        <p className="text-accent text-xl">Loading RL Learner data...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-surface rounded-lg p-6 text-center border border-gray-700">
        <p className="text-red-400 text-xl">Error: {error}</p>
        <p className="text-gray-500 mt-2">Make sure the backend is running.</p>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="bg-surface rounded-lg p-6 text-center border border-gray-700">
        <p className="text-gray-500 text-xl">No learner data available</p>
      </div>
    )
  }

  const stats = data.stats
  const goodSkips = stats?.counterfactual_good_skips ?? 0
  const badSkips = stats?.counterfactual_bad_skips ?? 0
  const totalSkips = goodSkips + badSkips

  return (
    <div className="space-y-4">
      {/* Stats Overview - 6 column grid like Streamlit */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">States</p>
          <p className="text-2xl font-bold text-accent">{stats?.total_states ?? 0}</p>
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">Trades</p>
          <p className="text-2xl font-bold text-accent">{stats?.total_trades ?? 0}</p>
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">Win %</p>
          <p className={`text-2xl font-bold ${(stats?.win_rate ?? 0) >= 50 ? 'text-green-400' : 'text-red-400'}`}>
            {(stats?.win_rate ?? 0).toFixed(0)}%
          </p>
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">P&L</p>
          <p className={`text-2xl font-bold ${(stats?.total_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${(stats?.total_pnl ?? 0) >= 0 ? '+' : ''}{(stats?.total_pnl ?? 0).toFixed(0)}
          </p>
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">Epsilon</p>
          <p className="text-2xl font-bold text-yellow-400">
            {((stats?.epsilon ?? 0) * 100).toFixed(0)}%
          </p>
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">Skips</p>
          <p className="text-2xl font-bold text-blue-400">
            {goodSkips}/{totalSkips}
          </p>
          <p className="text-xs text-gray-500">{totalSkips > 0 ? ((goodSkips / totalSkips) * 100).toFixed(0) : 0}% accuracy</p>
        </div>
      </div>

      {/* Best/Worst States - side by side with parsed components */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-bold text-green-400 mb-3">Best States to Trade</h3>
          {(data.best_states ?? []).length === 0 ? (
            <p className="text-gray-500 text-sm">No states learned yet</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
                    <th className="pb-2 text-left">Tkr</th>
                    <th className="pb-2 text-left">Trend</th>
                    <th className="pb-2 text-left">RSI</th>
                    <th className="pb-2 text-right">Q</th>
                  </tr>
                </thead>
                <tbody>
                  {data.best_states.map((s, i) => {
                    const parsed = parseState(s.state)
                    return (
                      <tr key={i} className="border-b border-gray-800">
                        <td className="py-2 font-bold text-accent">{parsed.ticker}</td>
                        <td className={`py-2 ${parsed.trend === 'bull' ? 'text-green-400' : parsed.trend === 'bear' ? 'text-red-400' : 'text-yellow-400'}`}>
                          {parsed.trend}
                        </td>
                        <td className="py-2 text-gray-300">{parsed.rsi}</td>
                        <td className="py-2 text-right font-mono text-green-400">
                          {s.q_enter?.toFixed(3) ?? 'N/A'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-bold text-red-400 mb-3">States to Avoid</h3>
          {(data.worst_states ?? []).length === 0 ? (
            <p className="text-gray-500 text-sm">No states learned yet</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
                    <th className="pb-2 text-left">Tkr</th>
                    <th className="pb-2 text-left">Trend</th>
                    <th className="pb-2 text-left">RSI</th>
                    <th className="pb-2 text-right">Q</th>
                  </tr>
                </thead>
                <tbody>
                  {data.worst_states.map((s, i) => {
                    const parsed = parseState(s.state)
                    return (
                      <tr key={i} className="border-b border-gray-800">
                        <td className="py-2 font-bold text-accent">{parsed.ticker}</td>
                        <td className={`py-2 ${parsed.trend === 'bull' ? 'text-green-400' : parsed.trend === 'bear' ? 'text-red-400' : 'text-yellow-400'}`}>
                          {parsed.trend}
                        </td>
                        <td className="py-2 text-gray-300">{parsed.rsi}</td>
                        <td className="py-2 text-right font-mono text-red-400">
                          {s.q_enter?.toFixed(3) ?? 'N/A'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Recent Decisions */}
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-bold">Recent Decisions</h3>
          {data.decisions && (
            <div className="flex gap-4 text-sm">
              <span className="text-green-400">
                Enters: <span className="font-bold">{data.decisions.enters ?? 0}</span>
              </span>
              <span className="text-red-400">
                Skips: <span className="font-bold">{data.decisions.skips ?? 0}</span>
              </span>
              <span className="text-yellow-400">
                Explores: <span className="font-bold">{data.decisions.explorations ?? 0}</span>
              </span>
            </div>
          )}
        </div>
        {(data.recent_decisions ?? []).length === 0 ? (
          <p className="text-gray-500 text-sm">No recent decisions</p>
        ) : (
          <div className="overflow-x-auto max-h-64 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface">
                <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
                  <th className="pb-2 text-left">Time</th>
                  <th className="pb-2 text-left">State</th>
                  <th className="pb-2 text-left">Action</th>
                  <th className="pb-2 text-center">Exp</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_decisions.map((d, i) => {
                  const timestamp = d.timestamp ? d.timestamp.slice(-8) : 'N/A'
                  const stateShort = d.state ? d.state.slice(0, 25) : 'N/A'
                  return (
                    <tr key={i} className="border-b border-gray-800">
                      <td className="py-2 text-gray-400 text-xs font-mono">
                        {timestamp}
                      </td>
                      <td className="py-2 text-gray-300 font-mono text-xs">
                        {stateShort}
                      </td>
                      <td className={`py-2 font-bold ${d.action === 'ENTER' || d.action === 'enter' || d.action === 'take' ? 'text-green-400' : 'text-red-400'}`}>
                        {d.action?.toUpperCase() ?? 'N/A'}
                      </td>
                      <td className="py-2 text-center text-yellow-400 font-bold">
                        {d.explored ? 'Y' : ''}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
