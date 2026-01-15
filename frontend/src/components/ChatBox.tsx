import { useState } from 'react'
import axios from 'axios'

export function ChatBox() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([
    { role: 'assistant', content: 'Router active. Standing by.' }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMsg = { role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const res = await axios.post('/api/ask', { question: input })
      const botMsg = { role: 'assistant', content: res.data.reply }
      setMessages(prev => [...prev, botMsg])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'API error - try again' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-surface rounded-lg p-4 border-2 border-[#4ade80] shadow-lg shadow-green-500/10">
      <h3 className="text-lg font-bold text-[#4ade80] mb-3 flex items-center gap-2">
        <span className="w-2 h-2 bg-[#4ade80] rounded-full animate-pulse"></span>
        Iron Spark Brain
      </h3>
      <div className="h-48 overflow-y-auto mb-3 p-3 bg-background rounded border border-gray-700">
        {messages.map((msg, i) => (
          <div key={i} className={`mb-3 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
            <span className={`inline-block max-w-xs px-3 py-2 rounded-lg text-sm ${msg.role === 'user' ? 'bg-[#1e3a5f] text-[#4ade80] border border-[#4ade80]' : 'bg-gray-800 text-gray-200 border border-gray-600'}`}>
              {msg.content}
            </span>
          </div>
        ))}
        {loading && <div className="text-center text-accent text-sm">Thinking...</div>}
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Ask Iron Spark..."
          className="flex-1 bg-gray-800 rounded px-3 py-2 text-sm border border-gray-600 focus:outline-none focus:border-[#4ade80] focus:ring-1 focus:ring-[#4ade80]"
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="bg-[#4ade80] text-black px-4 py-2 rounded font-semibold text-sm hover:brightness-110 disabled:opacity-50 transition-all"
        >
          Send
        </button>
      </div>
    </div>
  )
}
