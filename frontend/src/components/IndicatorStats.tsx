import { useState, useEffect } from 'react'
import axios from 'axios'

interface RegimeStat {
  wins: number
  losses: number
  returns: number[]
}

interface BinomialStat {
  win_rate: number
  confidence_interval: [number, number]
  sample_size: number
  reliable: boolean
  expected_value: number
  recommendation: string
  error?: string
}

interface KellyStat {
  kelly_fraction: number
  half_kelly: number
  recommendation: string
  max_bet_percent: number
  error?: string
}

interface ConditionalEV {
  conditional_win_rate: number
  expected_value: number
  sample_size: number
  recommendation: string
  error?: string
}

interface IndicatorStatsData {
  available: boolean
  trade_history: Array<{
    pnl: number
    win: boolean
    regime: string
    setup_type: string
    entry_time: string
    ticker: string
  }>
  regime_stats: Record<string, RegimeStat>
  binomial_stats: Record<string, BinomialStat>
  kelly_stats: KellyStat
  conditional_ev: Record<string, ConditionalEV>
  summary?: {
    total_trades: number
    win_rate: number
    avg_win: number
    avg_loss: number
    total_pnl: number
  }
  last_updated: string | null
}

export default function IndicatorStats() {
  const [stats, setStats] = useState<IndicatorStatsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await axios.get('/api/indicator-stats')
        setStats(res.data)
        setError(null)
      } catch (err) {
        setError('Failed to load indicator stats')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
    const interval = setInterval(fetchStats, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        <p className="text-gray-400">Loading indicator stats...</p>
      </div>
    )
  }

  if (error || !stats) {
    return (
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        <p className="text-red-400">{error || 'No stats available'}</p>
      </div>
    )
  }

  if (!stats.available) {
    return (
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        <p className="text-yellow-400">Indicator stats module not available</p>
      </div>
    )
  }

  const summary = stats.summary || { total_trades: 0, win_rate: 0, avg_win: 0, avg_loss: 0, total_pnl: 0 }
  const regimes = ['bull', 'bear', 'chop']

  return (
    <div className="space-y-4">
      {/* Summary Header */}
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-bold mb-3 text-accent">Binomial Statistics</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <div>
            <p className="text-xs text-gray-400 uppercase">Total Trades</p>
            <p className="text-xl font-bold text-accent">{summary.total_trades}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase">Win Rate</p>
            <p className={`text-xl font-bold ${summary.win_rate >= 50 ? 'text-green-400' : 'text-red-400'}`}>
              {summary.win_rate.toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase">Avg Win</p>
            <p className="text-xl font-bold text-green-400">${summary.avg_win.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase">Avg Loss</p>
            <p className="text-xl font-bold text-red-400">${summary.avg_loss.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400 uppercase">Total P&L</p>
            <p className={`text-xl font-bold ${summary.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${summary.total_pnl >= 0 ? '+' : ''}{summary.total_pnl.toFixed(2)}
            </p>
          </div>
        </div>
      </div>

      {/* Regime Breakdown */}
      <div className="bg-surface rounded-lg p-4 border border-gray-700">
        <h3 className="text-lg font-bold mb-3">Win Rate by Regime</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {regimes.map((regime) => {
            const regimeStat = stats.regime_stats[regime]
            const binomial = stats.binomial_stats[regime]
            const condEV = stats.conditional_ev[regime]

            if (!regimeStat && !binomial) {
              return (
                <div key={regime} className="bg-background rounded-lg p-3 border border-gray-800">
                  <h4 className="font-bold text-gray-400 uppercase mb-2">{regime}</h4>
                  <p className="text-gray-500 text-sm">No data yet</p>
                </div>
              )
            }

            const wins = regimeStat?.wins || 0
            const losses = regimeStat?.losses || 0
            const total = wins + losses
            const rawWinRate = total > 0 ? (wins / total * 100) : 0

            return (
              <div key={regime} className="bg-background rounded-lg p-3 border border-gray-800">
                <h4 className={`font-bold uppercase mb-2 ${
                  regime === 'bull' ? 'text-green-400' :
                  regime === 'bear' ? 'text-red-400' : 'text-yellow-400'
                }`}>
                  {regime}
                </h4>

                {/* Raw stats */}
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">Record:</span>
                  <span className="font-bold">{wins}W / {losses}L</span>
                </div>

                {/* Win rate with confidence interval */}
                {binomial && !binomial.error && (
                  <>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Win Rate:</span>
                      <span className={`font-bold ${rawWinRate >= 50 ? 'text-green-400' : 'text-red-400'}`}>
                        {rawWinRate.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">95% CI:</span>
                      <span className="text-gray-300">
                        [{(binomial.confidence_interval[0] * 100).toFixed(0)}% - {(binomial.confidence_interval[1] * 100).toFixed(0)}%]
                      </span>
                    </div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-400">Expected Value:</span>
                      <span className={`font-bold ${binomial.expected_value >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${binomial.expected_value.toFixed(2)}
                      </span>
                    </div>

                    {/* Reliability indicator */}
                    <div className={`text-xs px-2 py-1 rounded text-center ${
                      binomial.reliable
                        ? 'bg-green-900/50 text-green-300'
                        : 'bg-yellow-900/50 text-yellow-300'
                    }`}>
                      {binomial.reliable ? 'Statistically Reliable' : `Need ${20 - total} more trades`}
                    </div>
                  </>
                )}

                {/* Conditional EV recommendation */}
                {condEV && !condEV.error && condEV.recommendation && (
                  <div className={`text-xs mt-2 px-2 py-1 rounded text-center ${
                    condEV.recommendation === 'favorable'
                      ? 'bg-green-900/50 text-green-300'
                      : condEV.recommendation === 'unfavorable'
                      ? 'bg-red-900/50 text-red-300'
                      : 'bg-gray-800 text-gray-400'
                  }`}>
                    {condEV.recommendation.charAt(0).toUpperCase() + condEV.recommendation.slice(1)}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Kelly Criterion */}
      {stats.kelly_stats && !stats.kelly_stats.error && (
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-bold mb-3">Kelly Criterion</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <p className="text-xs text-gray-400 uppercase">Full Kelly</p>
              <p className="text-xl font-bold text-accent">
                {(stats.kelly_stats.kelly_fraction * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400 uppercase">Half Kelly (Safer)</p>
              <p className="text-xl font-bold text-green-400">
                {(stats.kelly_stats.half_kelly * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400 uppercase">Max Bet</p>
              <p className="text-xl font-bold text-yellow-400">
                {stats.kelly_stats.max_bet_percent.toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-400 uppercase">Recommendation</p>
              <p className={`text-lg font-bold ${
                stats.kelly_stats.recommendation === 'strong_edge' ? 'text-green-400' :
                stats.kelly_stats.recommendation === 'moderate_edge' ? 'text-yellow-400' :
                stats.kelly_stats.recommendation === 'slight_edge' ? 'text-orange-400' :
                'text-red-400'
              }`}>
                {stats.kelly_stats.recommendation.replace('_', ' ').toUpperCase()}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Recent Trades */}
      {stats.trade_history.length > 0 && (
        <div className="bg-surface rounded-lg p-4 border border-gray-700">
          <h3 className="text-lg font-bold mb-3">Recent Trades</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700 text-gray-400 text-xs uppercase">
                  <th className="pb-2 text-left">Time</th>
                  <th className="pb-2 text-left">Ticker</th>
                  <th className="pb-2 text-left">Setup</th>
                  <th className="pb-2 text-left">Regime</th>
                  <th className="pb-2 text-right">P&L</th>
                  <th className="pb-2 text-center">Result</th>
                </tr>
              </thead>
              <tbody>
                {stats.trade_history.slice(-10).reverse().map((trade, i) => (
                  <tr key={i} className="border-b border-gray-800">
                    <td className="py-2 text-gray-400">
                      {new Date(trade.entry_time).toLocaleString(undefined, {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </td>
                    <td className="py-2 font-bold">{trade.ticker}</td>
                    <td className="py-2">{trade.setup_type}</td>
                    <td className={`py-2 ${
                      trade.regime === 'bull' ? 'text-green-400' :
                      trade.regime === 'bear' ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {trade.regime.toUpperCase()}
                    </td>
                    <td className={`py-2 text-right font-bold ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      ${trade.pnl >= 0 ? '+' : ''}{trade.pnl.toFixed(2)}
                    </td>
                    <td className="py-2 text-center">
                      <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                        trade.win ? 'bg-green-900/50 text-green-300' : 'bg-red-900/50 text-red-300'
                      }`}>
                        {trade.win ? 'WIN' : 'LOSS'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Last Updated */}
      {stats.last_updated && (
        <p className="text-xs text-gray-500 text-right">
          Last updated: {new Date(stats.last_updated).toLocaleString()}
        </p>
      )}
    </div>
  )
}
