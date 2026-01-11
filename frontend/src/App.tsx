import { useEffect, useState } from 'react'
import axios from 'axios'

// Components
import { EquityChart } from './components/EquityChart'
import { ChatBox } from './components/ChatBox'
import TradingLogs from './components/TradingLogs'
import RLLearner from './components/RLLearner'
import LiveCharts from './components/LiveCharts'

// Type definition for the data returned by /api/state
interface DashboardState {
  equity?: number
  daily_pnl?: number
  trade_count?: number
  market_open?: boolean
  equity_history?: number[]
  positions_1m?: Record<string, any>
  positions_15m?: Record<string, any>
  positions_scalp?: Record<string, any>
}

type TabType = 'dashboard' | 'logs' | 'learner' | 'charts'

function App() {
  const [state, setState] = useState<DashboardState | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<TabType>('dashboard')

  // Data fetching from FastAPI backend
  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await axios.get('/api/state')
        setState(res.data)
        setError(null)
      } catch (err) {
        console.error('API fetch error:', err)
        setError('Failed to load data from API')
      } finally {
        setLoading(false)
      }
    }

    fetchState()
    const interval = setInterval(fetchState, 5000)
    return () => clearInterval(interval)
  }, [])

  // Loading screen
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="text-2xl text-accent">Loading Snoogans dashboard...</div>
      </div>
    )
  }

  // Error screen
  if (error || !state) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="text-2xl text-red-500">
          {error || 'No data received'}
        </div>
      </div>
    )
  }

  // Safe defaults
  const equity = state.equity ?? 0
  const daily_pnl = state.daily_pnl ?? 0
  const trade_count = state.trade_count ?? 0
  const market_open = state.market_open ?? false

  const tabs: { id: TabType; label: string }[] = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'logs', label: 'Trading Logs' },
    { id: 'learner', label: 'RL Learner' },
    { id: 'charts', label: 'Live Charts' },
  ]

  return (
    <div className="min-h-screen bg-background text-text p-4">
      {/* Tab Navigation */}
      <div className="flex justify-center border-b border-gray-700 mb-6">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-6 py-3 font-semibold transition-all duration-200 relative ${
              activeTab === tab.id
                ? 'text-[#f87171]'
                : 'text-[#4ade80] hover:text-[#86efac]'
            }`}
          >
            {tab.label}
            {activeTab === tab.id && (
              <span className="absolute bottom-0 left-0 right-0 h-[3px] bg-[#f87171] rounded-t" />
            )}
          </button>
        ))}
      </div>

      {/* Dashboard Tab */}
      {activeTab === 'dashboard' && (
        <div className="space-y-4">
          {/* Header metrics - more compact */}
          <header className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <div className="bg-surface rounded-lg p-4 border border-gray-700">
              <h2 className="text-xs text-gray-400 uppercase tracking-wide">Equity</h2>
              <p className="text-2xl font-bold text-accent">
                ${equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </p>
            </div>
            <div className="bg-surface rounded-lg p-4 border border-gray-700">
              <h2 className="text-xs text-gray-400 uppercase tracking-wide">Daily P&L</h2>
              <p className={`text-2xl font-bold ${daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {daily_pnl >= 0 ? '+' : ''}${daily_pnl.toFixed(2)}
              </p>
            </div>
            <div className="bg-surface rounded-lg p-4 border border-gray-700">
              <h2 className="text-xs text-gray-400 uppercase tracking-wide">Brain Progress</h2>
              <div className="flex items-center gap-2">
                <p className="text-xl font-bold">{trade_count % 30}/30</p>
                <div className="flex-1 bg-gray-800 rounded-full h-3">
                  <div
                    className="bg-accent h-3 rounded-full transition-all"
                    style={{ width: `${((trade_count % 30) / 30) * 100}%` }}
                  />
                </div>
              </div>
              <p className="text-xs text-gray-500">{Math.floor(trade_count / 30)} batches</p>
            </div>
            <div className="bg-surface rounded-lg p-4 border border-gray-700">
              <h2 className="text-xs text-gray-400 uppercase tracking-wide">Market</h2>
              <p className={`text-2xl font-bold ${market_open ? 'text-green-400' : 'text-red-400'}`}>
                {market_open ? 'OPEN' : 'CLOSED'}
              </p>
            </div>
          </header>

          {/* Equity Curve and Chat - side by side */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-surface rounded-lg p-4 border border-gray-700">
              <h2 className="text-lg font-bold mb-2">Equity Curve</h2>
              <div className="h-48">
                <EquityChart data={state.equity_history ?? []} />
              </div>
            </div>
            <ChatBox />
          </section>
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
