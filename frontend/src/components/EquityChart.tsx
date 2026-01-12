import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface EquityChartProps {
  data: number[]
}

export function EquityChart({ data }: EquityChartProps) {
  // Convert flat array of equity values to chart data points
  const chartData = data.map((value, index) => ({
    point: index + 1,
    equity: Number(value.toFixed(2)),
  }))

  if (chartData.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        No equity history yet - waiting for first trade
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
        <XAxis dataKey="point" stroke="#888" />
        <YAxis stroke="#888" />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e1e1e', border: '1px solid #00ff9d' }}
          labelStyle={{ color: '#00ff9d' }}
          formatter={(value: number | undefined) => value !== undefined ? `$${value.toLocaleString()}` : '-'}
        />
        <Line
          type="monotone"
          dataKey="equity"
          stroke="#00ff9d"
          strokeWidth={3}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
