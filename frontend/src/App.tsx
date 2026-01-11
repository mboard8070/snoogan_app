import { useState, useEffect } from 'react'
import TradingLogs from './components/TradingLogs'
import LiveCharts from './components/LiveCharts'
import RLLearner from './components/RLLearner'

interface DashboardState {
  equity: number
  daily_pnl: number
  trade_count: number
  market_open: boolean
  positions_1m: Record<string, unknown>[]
  positions_15m: Record<string, unknown>[]
  positions_scalp: Record<string, unknown>[]
  equity_history: number[]
}

type TabType = 'dashboard' | 'logs' | 'learner' | 'charts'

function App() {
  const [state, setState] = useState<DashboardState | null>(null)
  const [activeTab, setActiveTab] = useState<TabType>('dashboard')

  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await fetch('/api/state')
        const data = await res.json()
        setState(data)
      } catch (err) {
        console.error('Failed to fetch state:', err)
      }
    }
    fetchState()
    const interval = setInterval(fetchState, 5000)
    return () => clearInterval(interval)
  }, [])

  const tabs: { id: TabType; label: string }[] = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'logs', label: 'Trading Logs' },
    { id: 'learner', label: 'RL Learner' },
    { id: 'charts', label: 'Live Charts' },
  ]

  return (
    <div className="min-h-screen bg-snoogans-dark p-4">
      <h1 className="text-3xl font-bold text-center mb-6 text-white">
        Snoogans Trading Dashboard
      </h1>

      {/* Tab Navigation */}
      <div className="flex justify-center gap-2 mb-6">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeTab === tab.id
                ? 'bg-snoogans-accent text-white'
                : 'bg-snoogans-card text-gray-400 hover:text-white'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && state && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {/* Equity Card */}
          <div className="bg-snoogans-card rounded-lg p-4">
            <h3 className="text-gray-400 text-sm mb-1">Portfolio Equity</h3>
            <p className="text-2xl font-bold text-white">
              ${state.equity.toLocaleString()}
            </p>
          </div>

          {/* Daily P&L Card */}
          <div className="bg-snoogans-card rounded-lg p-4">
            <h3 className="text-gray-400 text-sm mb-1">Daily P&L</h3>
            <p
              className={`text-2xl font-bold ${
                state.daily_pnl >= 0 ? 'text-snoogans-green' : 'text-snoogans-red'
              }`}
            >
              ${state.daily_pnl.toLocaleString()}
            </p>
          </div>

          {/* Trade Count Card */}
          <div className="bg-snoogans-card rounded-lg p-4">
            <h3 className="text-gray-400 text-sm mb-1">Total Trades</h3>
            <p className="text-2xl font-bold text-white">{state.trade_count}</p>
          </div>

          {/* Market Status Card */}
          <div className="bg-snoogans-card rounded-lg p-4">
            <h3 className="text-gray-400 text-sm mb-1">Market Status</h3>
            <p
              className={`text-2xl font-bold ${
                state.market_open ? 'text-snoogans-green' : 'text-snoogans-red'
              }`}
            >
              {state.market_open ? 'OPEN' : 'CLOSED'}
            </p>
          </div>

          {/* Brain Progress Card */}
          <div className="bg-snoogans-card rounded-lg p-4 col-span-full md:col-span-2">
            <h3 className="text-gray-400 text-sm mb-1">Brain Progress</h3>
            <p className="text-2xl font-bold text-white">
              {state.trade_count % 30}/30 to next batch
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {Math.floor(state.trade_count / 30)} batches saved
            </p>
            <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-snoogans-accent transition-all"
                style={{ width: `${((state.trade_count % 30) / 30) * 100}%` }}
              />
            </div>
          </div>

          {/* Positions Summary */}
          <div className="bg-snoogans-card rounded-lg p-4 col-span-full md:col-span-2">
            <h3 className="text-gray-400 text-sm mb-2">Open Positions</h3>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div>
                <p className="text-lg font-bold text-white">
                  {state.positions_1m?.length || 0}
                </p>
                <p className="text-xs text-gray-500">1m Strategy</p>
              </div>
              <div>
                <p className="text-lg font-bold text-white">
                  {state.positions_15m?.length || 0}
                </p>
                <p className="text-xs text-gray-500">15m Strategy</p>
              </div>
              <div>
                <p className="text-lg font-bold text-white">
                  {state.positions_scalp?.length || 0}
                </p>
                <p className="text-xs text-gray-500">Scalp Strategy</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Trading Logs Tab */}
      {activeTab === 'logs' && <TradingLogs />}

      {/* RL Learner Tab */}
      {activeTab === 'learner' && <RLLearner />}

      {/* Live Charts Tab */}
      {activeTab === 'charts' && <LiveCharts />}
    </div>
  )
}

export default App
