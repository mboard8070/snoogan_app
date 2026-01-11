import { useState, useEffect } from 'react'

interface LearnerStats {
  total_states: number
  total_decisions: number
  epsilon: number
  learning_rate: number
}

interface StateInfo {
  state: string
  q_value: number
  visits: number
}

interface Decision {
  timestamp: string
  state: string
  action: string
  reward: number
}

interface LearnerData {
  stats: LearnerStats
  decisions: {
    total_take: number
    total_skip: number
    take_ratio: number
  }
  best_states: StateInfo[]
  worst_states: StateInfo[]
  recent_decisions: Decision[]
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

  return (
    <div className="space-y-4">
      {/* Stats Overview - compact grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">Total States</p>
          <p className="text-2xl font-bold text-accent">{data.stats?.total_states ?? 0}</p>
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">Total Decisions</p>
          <p className="text-2xl font-bold text-accent">{data.stats?.total_decisions ?? 0}</p>
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">Exploration</p>
          <p className="text-2xl font-bold text-yellow-400">
            {((data.stats?.epsilon ?? 0) * 100).toFixed(1)}%
          </p>
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <p className="text-xs text-gray-400 uppercase">Take Ratio</p>
          <p className="text-2xl font-bold text-green-400">
            {((data.decisions?.take_ratio ?? 0) * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Best/Worst States - side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-bold text-green-400 mb-3">Best States</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {(data.best_states ?? []).length === 0 ? (
              <p className="text-gray-500 text-sm">No states learned yet</p>
            ) : (
              data.best_states.map((s, i) => (
                <div key={i} className="bg-background rounded p-2 text-sm">
                  <p className="text-gray-300 font-mono text-xs truncate">{s.state}</p>
                  <p className="text-green-400 text-xs mt-1">
                    Q: {s.q_value?.toFixed(4) ?? 'N/A'} | Visits: {s.visits ?? 0}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-bold text-red-400 mb-3">Worst States</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {(data.worst_states ?? []).length === 0 ? (
              <p className="text-gray-500 text-sm">No states learned yet</p>
            ) : (
              data.worst_states.map((s, i) => (
                <div key={i} className="bg-background rounded p-2 text-sm">
                  <p className="text-gray-300 font-mono text-xs truncate">{s.state}</p>
                  <p className="text-red-400 text-xs mt-1">
                    Q: {s.q_value?.toFixed(4) ?? 'N/A'} | Visits: {s.visits ?? 0}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Recent Decisions */}
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-bold mb-3">Recent Decisions</h3>
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
                  <th className="pb-2 text-right">Reward</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_decisions.map((d, i) => (
                  <tr key={i} className="border-b border-gray-800">
                    <td className="py-2 text-gray-400 text-xs">
                      {d.timestamp ? new Date(d.timestamp).toLocaleTimeString() : 'N/A'}
                    </td>
                    <td className="py-2 text-gray-300 font-mono text-xs truncate max-w-xs">
                      {d.state}
                    </td>
                    <td className={`py-2 font-bold ${d.action === 'take' ? 'text-green-400' : 'text-red-400'}`}>
                      {d.action?.toUpperCase() ?? 'N/A'}
                    </td>
                    <td className={`py-2 text-right font-mono ${(d.reward ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {d.reward?.toFixed(4) ?? '0.0000'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
