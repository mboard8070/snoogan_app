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
        if (!res.ok) throw new Error('Failed to fetch learner data')
        const json = await res.json()
        setData(json)
        setError(null)
      } catch (err) {
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
      <div className="bg-snoogans-card rounded-lg p-6 text-center">
        <p className="text-gray-400">Loading RL Learner data...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-snoogans-card rounded-lg p-6 text-center">
        <p className="text-snoogans-red">Error: {error}</p>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="space-y-4">
      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-snoogans-card rounded-lg p-4">
          <p className="text-gray-400 text-xs">Total States</p>
          <p className="text-2xl font-bold text-white">{data.stats.total_states}</p>
        </div>
        <div className="bg-snoogans-card rounded-lg p-4">
          <p className="text-gray-400 text-xs">Total Decisions</p>
          <p className="text-2xl font-bold text-white">{data.stats.total_decisions}</p>
        </div>
        <div className="bg-snoogans-card rounded-lg p-4">
          <p className="text-gray-400 text-xs">Exploration (ε)</p>
          <p className="text-2xl font-bold text-snoogans-accent">
            {(data.stats.epsilon * 100).toFixed(1)}%
          </p>
        </div>
        <div className="bg-snoogans-card rounded-lg p-4">
          <p className="text-gray-400 text-xs">Take/Skip Ratio</p>
          <p className="text-2xl font-bold text-snoogans-green">
            {(data.decisions.take_ratio * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Best/Worst States */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="bg-snoogans-card rounded-lg p-4">
          <h3 className="text-lg font-semibold text-snoogans-green mb-3">Best States</h3>
          <div className="space-y-2">
            {data.best_states.map((s, i) => (
              <div key={i} className="bg-black/30 rounded p-2 text-sm">
                <p className="text-gray-300 font-mono truncate">{s.state}</p>
                <p className="text-snoogans-green">Q: {s.q_value.toFixed(4)} | Visits: {s.visits}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="bg-snoogans-card rounded-lg p-4">
          <h3 className="text-lg font-semibold text-snoogans-red mb-3">Worst States</h3>
          <div className="space-y-2">
            {data.worst_states.map((s, i) => (
              <div key={i} className="bg-black/30 rounded p-2 text-sm">
                <p className="text-gray-300 font-mono truncate">{s.state}</p>
                <p className="text-snoogans-red">Q: {s.q_value.toFixed(4)} | Visits: {s.visits}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Decisions */}
      <div className="bg-snoogans-card rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-3">Recent Decisions</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left py-2">Time</th>
                <th className="text-left py-2">State</th>
                <th className="text-left py-2">Action</th>
                <th className="text-right py-2">Reward</th>
              </tr>
            </thead>
            <tbody>
              {data.recent_decisions.map((d, i) => (
                <tr key={i} className="border-b border-gray-800">
                  <td className="py-2 text-gray-400">
                    {new Date(d.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="py-2 text-gray-300 font-mono truncate max-w-xs">
                    {d.state}
                  </td>
                  <td className={`py-2 font-medium ${d.action === 'take' ? 'text-snoogans-green' : 'text-snoogans-red'}`}>
                    {d.action.toUpperCase()}
                  </td>
                  <td className={`py-2 text-right ${d.reward >= 0 ? 'text-snoogans-green' : 'text-snoogans-red'}`}>
                    {d.reward.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
