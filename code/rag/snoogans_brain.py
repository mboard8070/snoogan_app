# /home/mboard76/nvidia-workbench/snoogan_app/code/rag/snoogans_brain.py

import os
import json
import shutil
import stat
import re
from datetime import datetime, timedelta
from collections import defaultdict

# Import indicators for real-time analysis (optional - fails gracefully)
try:
    from code.trader.indicators import get_trend_signal, _rsi, _macd, _atr
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False

from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIG CONSTANTS ---
BASE_DIR = "/home/mboard76/nvidia-workbench/snoogan_app"
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "knowledge")
DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")
OLLAMA_HOST = "http://172.17.0.1:11434"
DATA_DIR = os.path.join(BASE_DIR, "data")
LEARNER_STATE_FILE = os.path.join(BASE_DIR, "code", "rag", "adaptive_learner_state.json")


class SnoogansBrain:
    def __init__(self, chunk_size=400, chunk_overlap=150):
        os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

        if os.path.exists(DATA_DIR):
            try:
                os.chmod(DATA_DIR, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            except Exception as e:
                print(f"WARNING: Could not set permissions on {DATA_DIR}. Error: {e}")

        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_HOST
        )
        self.llm = ChatOllama(
            model="snoogans:latest",
            temperature=0.1,
            base_url=OLLAMA_HOST
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Track last known counts for auto-rebuild
        self._last_trade_count = self._count_trade_batches()
        self._last_trade_total = len(self._load_trade_batches())
        learner_state = self._load_learner_state()
        self._last_learner_trades = learner_state.get('stats', {}).get('total_trades', 0)

        # Load or Build Vector Store
        if os.path.exists(DB_PATH) and os.path.isdir(DB_PATH):
            print("Snoogans loaded his existing brain from disk — ready to sling spreads.")
            self.vectorstore = Chroma(
                persist_directory=DB_PATH,
                embedding_function=self.embeddings
            )
        else:
            print("Building Snoogans' brain from scratch — grab a blunt...")
            self._build_brain()

        if not hasattr(self, 'vectorstore'):
            raise RuntimeError("Vector store failed to build. Check 'data/knowledge' contents.")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        self.rag_chain = self._make_rag_chain()

    def _count_trade_batches(self) -> int:
        """Count number of trade batch files in knowledge dir."""
        if not os.path.exists(KNOWLEDGE_DIR):
            return 0
        return len([f for f in os.listdir(KNOWLEDGE_DIR)
                    if f.startswith('trades_batch_') and f.endswith('.json')])

    def _load_trade_batches(self) -> list:
        """Load all trade JSON files and return list of trades."""
        all_trades = []

        for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
            # Load trades_batch_*.json files (legacy format)
            if filename.startswith('trades_batch_') and filename.endswith('.json'):
                filepath = os.path.join(KNOWLEDGE_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        trades = json.load(f)
                        for trade in trades:
                            trade['_batch_file'] = filename
                        all_trades.extend(trades)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"WARNING: Could not load {filename}: {e}")

            # Also load trades.json (scalp strategy format)
            elif filename == 'trades.json':
                filepath = os.path.join(KNOWLEDGE_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        trades = json.load(f)
                        if isinstance(trades, list):
                            for trade in trades:
                                trade['_source'] = 'scalp_strategy'
                            all_trades.extend(trades)
                            print(f"Loaded {len(trades)} trades from trades.json")
                except (json.JSONDecodeError, IOError) as e:
                    print(f"WARNING: Could not load trades.json: {e}")

        # Load live trades from strategy state files (not yet batched)
        state_files = [
            (os.path.join(BASE_DIR, "strategy_15m_state.json"), "15m"),
            (os.path.join(BASE_DIR, "code", "strategy_state.json"), "1m"),
            (os.path.join(BASE_DIR, "code", "trader", "scalp_strategy_state.json"), "scalp"),
        ]

        for state_file, strategy_name in state_files:
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                        closed_trades = state.get('closed_trades', [])
                        for trade in closed_trades:
                            trade['_source'] = f'{strategy_name}_live'
                        all_trades.extend(closed_trades)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"WARNING: Could not load {state_file}: {e}")

        return all_trades

    def _analyze_trades(self, trades: list) -> dict:
        """Analyze trades and extract insights including market context patterns."""
        if not trades:
            return {}

        stats = {
            'total_trades': len(trades),
            'wins': 0,
            'losses': 0,
            'breakeven': 0,
            'total_pnl': 0.0,
            'by_ticker': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}),
            'by_type': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}),
            'by_hour': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}),
            'by_day': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}),
            'by_rsi_range': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}),
            'by_trend': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}),
            'by_macd': defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0}),
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_pnls': [],
            'loss_pnls': [],
        }

        for trade in trades:
            pnl = trade.get('pnl', 0)
            ticker = trade.get('ticker', 'UNKNOWN')
            entry_time = trade.get('entry_time', '')

            # Determine trade type
            if 'is_call' in trade:
                trade_type = 'LONG_CALL' if trade['is_call'] else 'LONG_PUT'
            elif 'is_put' in trade:
                trade_type = 'PUT_SPREAD' if trade['is_put'] else 'CALL_SPREAD'
            else:
                trade_type = 'UNKNOWN'

            # Parse entry time for hour/day analysis
            try:
                if 'T' in entry_time:
                    dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    hour = dt.hour
                    day_name = dt.strftime('%A')
                else:
                    hour = None
                    day_name = None
            except:
                hour = None
                day_name = None

            # Update stats
            stats['total_pnl'] += pnl

            if pnl > 0:
                stats['wins'] += 1
                stats['win_pnls'].append(pnl)
                stats['largest_win'] = max(stats['largest_win'], pnl)
            elif pnl < 0:
                stats['losses'] += 1
                stats['loss_pnls'].append(pnl)
                stats['largest_loss'] = min(stats['largest_loss'], pnl)
            else:
                stats['breakeven'] += 1

            # By ticker
            stats['by_ticker'][ticker]['trades'] += 1
            stats['by_ticker'][ticker]['pnl'] += pnl
            if pnl > 0:
                stats['by_ticker'][ticker]['wins'] += 1

            # By type
            stats['by_type'][trade_type]['trades'] += 1
            stats['by_type'][trade_type]['pnl'] += pnl
            if pnl > 0:
                stats['by_type'][trade_type]['wins'] += 1

            # By hour
            if hour is not None:
                stats['by_hour'][hour]['trades'] += 1
                stats['by_hour'][hour]['pnl'] += pnl
                if pnl > 0:
                    stats['by_hour'][hour]['wins'] += 1

            # By day
            if day_name:
                stats['by_day'][day_name]['trades'] += 1
                stats['by_day'][day_name]['pnl'] += pnl
                if pnl > 0:
                    stats['by_day'][day_name]['wins'] += 1

            # Analyze market context if available
            context = trade.get('market_context', {})
            if context:
                # By trend at entry
                trend = context.get('trend')
                if trend:
                    stats['by_trend'][trend]['trades'] += 1
                    stats['by_trend'][trend]['pnl'] += pnl
                    if pnl > 0:
                        stats['by_trend'][trend]['wins'] += 1

                # By RSI range
                rsi = context.get('rsi')
                if rsi is not None:
                    if rsi < 30:
                        rsi_range = 'oversold (<30)'
                    elif rsi < 40:
                        rsi_range = 'low (30-40)'
                    elif rsi < 50:
                        rsi_range = 'mid-low (40-50)'
                    elif rsi < 60:
                        rsi_range = 'mid-high (50-60)'
                    elif rsi < 70:
                        rsi_range = 'high (60-70)'
                    else:
                        rsi_range = 'overbought (>70)'

                    stats['by_rsi_range'][rsi_range]['trades'] += 1
                    stats['by_rsi_range'][rsi_range]['pnl'] += pnl
                    if pnl > 0:
                        stats['by_rsi_range'][rsi_range]['wins'] += 1

                # By MACD signal
                macd_bull = context.get('macd_bull')
                if macd_bull is not None:
                    macd_signal = 'MACD_BULL' if macd_bull else 'MACD_BEAR'
                    stats['by_macd'][macd_signal]['trades'] += 1
                    stats['by_macd'][macd_signal]['pnl'] += pnl
                    if pnl > 0:
                        stats['by_macd'][macd_signal]['wins'] += 1

        # Calculate averages
        if stats['win_pnls']:
            stats['avg_win'] = sum(stats['win_pnls']) / len(stats['win_pnls'])
        if stats['loss_pnls']:
            stats['avg_loss'] = sum(stats['loss_pnls']) / len(stats['loss_pnls'])

        stats['win_rate'] = (stats['wins'] / stats['total_trades'] * 100
                            if stats['total_trades'] > 0 else 0)

        return stats

    def _trades_to_documents(self, trades: list) -> list:
        """Convert trades into LangChain documents for the vector store."""
        if not trades:
            return []

        docs = []
        stats = self._analyze_trades(trades)

        # Document 1: Overall Performance Summary
        summary = f"""SNOOGANS TRADE PERFORMANCE SUMMARY
Total Trades Analyzed: {stats['total_trades']}
Overall Win Rate: {stats['win_rate']:.1f}%
Total P&L: ${stats['total_pnl']:+.2f}
Wins: {stats['wins']} | Losses: {stats['losses']} | Breakeven: {stats['breakeven']}
Largest Win: ${stats['largest_win']:+.2f}
Largest Loss: ${stats['largest_loss']:+.2f}
Average Win: ${stats['avg_win']:+.2f}
Average Loss: ${stats['avg_loss']:+.2f}
"""
        docs.append(Document(
            page_content=summary,
            metadata={"source": "trade_analysis", "type": "summary"}
        ))

        # Document 2: Performance by Ticker
        ticker_report = "PERFORMANCE BY TICKER\n"
        for ticker, data in sorted(stats['by_ticker'].items()):
            win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
            ticker_report += f"""
{ticker}:
  Trades: {data['trades']}
  Win Rate: {win_rate:.1f}%
  Total P&L: ${data['pnl']:+.2f}
  Average P&L per trade: ${data['pnl']/data['trades']:+.2f}
"""
        docs.append(Document(
            page_content=ticker_report,
            metadata={"source": "trade_analysis", "type": "by_ticker"}
        ))

        # Document 3: Performance by Trade Type
        type_report = "PERFORMANCE BY TRADE TYPE\n"
        for trade_type, data in sorted(stats['by_type'].items()):
            win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
            type_report += f"""
{trade_type}:
  Trades: {data['trades']}
  Win Rate: {win_rate:.1f}%
  Total P&L: ${data['pnl']:+.2f}
"""
        docs.append(Document(
            page_content=type_report,
            metadata={"source": "trade_analysis", "type": "by_trade_type"}
        ))

        # Document 4: Performance by Hour
        hour_report = "PERFORMANCE BY ENTRY HOUR (EST)\n"
        for hour in sorted(stats['by_hour'].keys()):
            data = stats['by_hour'][hour]
            win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
            hour_report += f"""
{hour}:00 - {hour+1}:00:
  Trades: {data['trades']}
  Win Rate: {win_rate:.1f}%
  Total P&L: ${data['pnl']:+.2f}
"""
        docs.append(Document(
            page_content=hour_report,
            metadata={"source": "trade_analysis", "type": "by_hour"}
        ))

        # Document 5: Performance by Day of Week
        day_report = "PERFORMANCE BY DAY OF WEEK\n"
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for day in day_order:
            if day in stats['by_day']:
                data = stats['by_day'][day]
                win_rate = (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
                day_report += f"""
{day}:
  Trades: {data['trades']}
  Win Rate: {win_rate:.1f}%
  Total P&L: ${data['pnl']:+.2f}
"""
        docs.append(Document(
            page_content=day_report,
            metadata={"source": "trade_analysis", "type": "by_day"}
        ))

        # Document 6: Market Context Analysis
        context_report = "MARKET CONTEXT PERFORMANCE PATTERNS\n"

        # By Trend
        if stats['by_trend']:
            context_report += "\nPERFORMANCE BY TREND SIGNAL:\n"
            for trend, data in sorted(stats['by_trend'].items()):
                if data['trades'] > 0:
                    win_rate = (data['wins'] / data['trades'] * 100)
                    context_report += f"  {trend.upper()}: {data['trades']} trades, {win_rate:.1f}% win rate, ${data['pnl']:+.2f} P&L\n"

        # By RSI Range
        if stats['by_rsi_range']:
            context_report += "\nPERFORMANCE BY RSI RANGE:\n"
            rsi_order = ['oversold (<30)', 'low (30-40)', 'mid-low (40-50)',
                        'mid-high (50-60)', 'high (60-70)', 'overbought (>70)']
            for rsi_range in rsi_order:
                if rsi_range in stats['by_rsi_range']:
                    data = stats['by_rsi_range'][rsi_range]
                    if data['trades'] > 0:
                        win_rate = (data['wins'] / data['trades'] * 100)
                        context_report += f"  RSI {rsi_range}: {data['trades']} trades, {win_rate:.1f}% win rate, ${data['pnl']:+.2f} P&L\n"

        # By MACD
        if stats['by_macd']:
            context_report += "\nPERFORMANCE BY MACD SIGNAL:\n"
            for macd_signal, data in sorted(stats['by_macd'].items()):
                if data['trades'] > 0:
                    win_rate = (data['wins'] / data['trades'] * 100)
                    context_report += f"  {macd_signal}: {data['trades']} trades, {win_rate:.1f}% win rate, ${data['pnl']:+.2f} P&L\n"

        docs.append(Document(
            page_content=context_report,
            metadata={"source": "trade_analysis", "type": "market_context"}
        ))

        # Document 7: Best and Worst Performers
        insights = "TRADE INSIGHTS AND PATTERNS\n"

        # Find best/worst ticker
        if stats['by_ticker']:
            best_ticker = max(stats['by_ticker'].items(),
                            key=lambda x: x[1]['wins']/x[1]['trades'] if x[1]['trades'] >= 5 else 0)
            worst_ticker = min(stats['by_ticker'].items(),
                             key=lambda x: x[1]['wins']/x[1]['trades'] if x[1]['trades'] >= 5 else 1)

            if best_ticker[1]['trades'] >= 5:
                insights += f"\nBest performing ticker: {best_ticker[0]} with {best_ticker[1]['wins']/best_ticker[1]['trades']*100:.1f}% win rate over {best_ticker[1]['trades']} trades"
            if worst_ticker[1]['trades'] >= 5:
                insights += f"\nWorst performing ticker: {worst_ticker[0]} with {worst_ticker[1]['wins']/worst_ticker[1]['trades']*100:.1f}% win rate over {worst_ticker[1]['trades']} trades"

        # Find best/worst hour
        if stats['by_hour']:
            valid_hours = {h: d for h, d in stats['by_hour'].items() if d['trades'] >= 3}
            if valid_hours:
                best_hour = max(valid_hours.items(),
                              key=lambda x: x[1]['wins']/x[1]['trades'])
                worst_hour = min(valid_hours.items(),
                               key=lambda x: x[1]['wins']/x[1]['trades'])
                insights += f"\nBest entry hour: {best_hour[0]}:00 EST with {best_hour[1]['wins']/best_hour[1]['trades']*100:.1f}% win rate"
                insights += f"\nWorst entry hour: {worst_hour[0]}:00 EST with {worst_hour[1]['wins']/worst_hour[1]['trades']*100:.1f}% win rate"

        docs.append(Document(
            page_content=insights,
            metadata={"source": "trade_analysis", "type": "insights"}
        ))

        # Document 7: Recent trades detail (last 30)
        recent = "RECENT TRADE HISTORY (Last 30 trades)\n"
        for trade in trades[-30:]:
            ticker = trade.get('ticker', '???')
            pnl = trade.get('pnl', 0)
            entry = trade.get('entry_time', 'unknown')[:19] if trade.get('entry_time') else 'unknown'

            if 'is_call' in trade:
                direction = 'LONG CALL' if trade['is_call'] else 'LONG PUT'
                strike = trade.get('strike_price', 0)
                desc = f"{direction} @ {strike}"
            elif 'is_put' in trade:
                direction = 'PUT SPREAD' if trade['is_put'] else 'CALL SPREAD'
                short = trade.get('short', 0)
                long = trade.get('long', 0)
                desc = f"{direction} {short}/{long}"
            else:
                desc = "UNKNOWN"

            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BE"
            recent += f"{entry} | {ticker} {desc} | ${pnl:+.2f} ({result})\n"

        docs.append(Document(
            page_content=recent,
            metadata={"source": "trade_analysis", "type": "recent_trades"}
        ))

        return docs

    def _load_learner_state(self) -> dict:
        """Load the RL learner's state file."""
        if not os.path.exists(LEARNER_STATE_FILE):
            return {}

        try:
            with open(LEARNER_STATE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"WARNING: Could not load learner state: {e}")
            return {}

    def _learner_to_documents(self, learner_state: dict) -> list:
        """
        Convert RL learner insights to documents for RAG.

        This provides the brain with learned patterns about which market conditions
        favor entering vs skipping trades, without overfitting to specific Q-values.
        """
        if not learner_state:
            return []

        docs = []
        q_table = learner_state.get('q_table', {})
        stats = learner_state.get('stats', {})

        if not q_table:
            return []

        # Document 1: Learner Overview
        total_states = len(q_table)
        total_trades = stats.get('total_trades', 0)
        enter_wins = stats.get('enter_wins', 0)
        enter_losses = stats.get('enter_losses', 0)
        total_pnl = stats.get('total_pnl', 0)
        good_skips = stats.get('counterfactual_good_skips', 0)
        bad_skips = stats.get('counterfactual_bad_skips', 0)

        win_rate = (enter_wins / total_trades * 100) if total_trades > 0 else 0
        skip_accuracy = (good_skips / (good_skips + bad_skips) * 100) if (good_skips + bad_skips) > 0 else 0

        overview = f"""RL LEARNER PERFORMANCE SUMMARY
States Learned: {total_states}
Total Trades Evaluated: {total_trades}
Trade Win Rate: {win_rate:.1f}%
Total P&L from Trades: ${total_pnl:+.2f}
Good Skips (avoided losses): {good_skips}
Bad Skips (missed wins): {bad_skips}
Skip Accuracy: {skip_accuracy:.1f}%

The RL learner uses Q-learning to decide whether to ENTER or SKIP trades based on
market conditions. Higher Q[enter] means historically profitable conditions.
Higher Q[skip] means historically unfavorable conditions.
"""
        docs.append(Document(
            page_content=overview,
            metadata={"source": "rl_learner", "type": "overview"}
        ))

        # Document 2: Best States to Trade (where entering has worked well)
        # Sort by Q[enter] - Q[skip] advantage
        state_advantages = []
        for state_key, q_vals in q_table.items():
            q_enter = q_vals.get('enter', 0)
            q_skip = q_vals.get('skip', 0)
            advantage = q_enter - q_skip
            state_advantages.append((state_key, q_enter, q_skip, advantage))

        state_advantages.sort(key=lambda x: x[3], reverse=True)

        # Top states to trade
        best_states = "RL LEARNER: BEST CONDITIONS TO TRADE\n"
        best_states += "These market conditions have historically favored ENTERING trades:\n\n"

        for state_key, q_enter, q_skip, advantage in state_advantages[:10]:
            if advantage > 0:
                parts = state_key.split('|')
                if len(parts) >= 5:
                    ticker, trend, rsi_bucket, hour_bucket, conf_bucket = parts[:5]
                    best_states += f"  {ticker} | {trend} trend | RSI {rsi_bucket} | {hour_bucket} session | {conf_bucket} confidence\n"
                    best_states += f"    → Q[enter]={q_enter:+.3f}, Q[skip]={q_skip:+.3f}, advantage={advantage:+.3f}\n\n"

        docs.append(Document(
            page_content=best_states,
            metadata={"source": "rl_learner", "type": "best_states"}
        ))

        # Document 3: Worst States (where skipping has been better)
        worst_states = "RL LEARNER: CONDITIONS TO AVOID\n"
        worst_states += "These market conditions have historically favored SKIPPING trades:\n\n"

        for state_key, q_enter, q_skip, advantage in reversed(state_advantages[-10:]):
            if advantage < 0:
                parts = state_key.split('|')
                if len(parts) >= 5:
                    ticker, trend, rsi_bucket, hour_bucket, conf_bucket = parts[:5]
                    worst_states += f"  {ticker} | {trend} trend | RSI {rsi_bucket} | {hour_bucket} session | {conf_bucket} confidence\n"
                    worst_states += f"    → Q[enter]={q_enter:+.3f}, Q[skip]={q_skip:+.3f}, advantage={advantage:+.3f}\n\n"

        docs.append(Document(
            page_content=worst_states,
            metadata={"source": "rl_learner", "type": "worst_states"}
        ))

        # Document 4: Pattern Insights by Ticker
        ticker_patterns = defaultdict(lambda: {'enter_advantage': [], 'skip_advantage': []})
        for state_key, q_enter, q_skip, advantage in state_advantages:
            parts = state_key.split('|')
            if len(parts) >= 1:
                ticker = parts[0]
                if advantage > 0:
                    ticker_patterns[ticker]['enter_advantage'].append(advantage)
                else:
                    ticker_patterns[ticker]['skip_advantage'].append(abs(advantage))

        ticker_doc = "RL LEARNER: PATTERNS BY TICKER\n\n"
        for ticker in ['SPY', 'QQQ', 'IWM']:
            data = ticker_patterns.get(ticker, {})
            enter_adv = data.get('enter_advantage', [])
            skip_adv = data.get('skip_advantage', [])

            if enter_adv or skip_adv:
                avg_enter = sum(enter_adv) / len(enter_adv) if enter_adv else 0
                avg_skip = sum(skip_adv) / len(skip_adv) if skip_adv else 0
                ticker_doc += f"{ticker}:\n"
                ticker_doc += f"  States favoring entry: {len(enter_adv)} (avg advantage: {avg_enter:.3f})\n"
                ticker_doc += f"  States favoring skip: {len(skip_adv)} (avg advantage: {avg_skip:.3f})\n\n"

        docs.append(Document(
            page_content=ticker_doc,
            metadata={"source": "rl_learner", "type": "ticker_patterns"}
        ))

        # Document 5: Recent Decisions (if available)
        decision_history = learner_state.get('decision_history', [])
        if decision_history:
            recent_decisions = "RL LEARNER: RECENT DECISIONS\n\n"
            for decision in decision_history[-20:]:
                action = decision.get('action', 'unknown')
                state = decision.get('state_key', 'unknown')
                reason = decision.get('reason', '')
                exploration = " (EXPLORATION)" if decision.get('exploration') else ""
                timestamp = decision.get('timestamp', '')[:19]

                recent_decisions += f"{timestamp} | {action.upper()}{exploration}\n"
                recent_decisions += f"  State: {state}\n"
                recent_decisions += f"  Reason: {reason}\n\n"

            docs.append(Document(
                page_content=recent_decisions,
                metadata={"source": "rl_learner", "type": "recent_decisions"}
            ))

        return docs

    def _build_brain(self):
        """Build the vector store from all knowledge sources."""
        docs = []
        loaded_files = []

        # Load PDFs
        pdf_files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.lower().endswith('.pdf')]
        if pdf_files:
            pdf_loader = PyPDFDirectoryLoader(KNOWLEDGE_DIR)
            docs.extend(pdf_loader.load())
            loaded_files.extend(pdf_files)

        # Load Text Files (manifesto, etc.)
        for file in os.listdir(KNOWLEDGE_DIR):
            if file.lower().endswith(('.txt', '.md', '.markdown')) and not file.startswith('.'):
                path = os.path.join(KNOWLEDGE_DIR, file)
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
                loaded_files.append(file)

        # Load Trade Batches
        trades = self._load_trade_batches()
        if trades:
            trade_docs = self._trades_to_documents(trades)
            docs.extend(trade_docs)
            batch_count = self._count_trade_batches()
            loaded_files.append(f"trades ({len(trades)} total)")
            print(f"Analyzed {len(trades)} trades")

        # Load RL Learner Insights
        learner_state = self._load_learner_state()
        if learner_state:
            learner_docs = self._learner_to_documents(learner_state)
            if learner_docs:
                docs.extend(learner_docs)
                q_table_size = len(learner_state.get('q_table', {}))
                loaded_files.append(f"RL learner ({q_table_size} states)")
                print(f"Loaded RL learner insights: {q_table_size} learned states")

        if not docs:
            raise ValueError("Knowledge dir empty — drop manifesto .txt or trade batches in data/knowledge/")

        print(f"Loaded: {', '.join(loaded_files)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=DB_PATH
        )

        print(f"Snoogans ingested {len(chunks)} chunks of theta wisdom. Brain built.")

    def rebuild_brain(self):
        """Force rebuild the brain with current knowledge."""
        print("Rebuilding Snoogans' brain with latest trade data...")

        # Delete existing vector store
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
            print("Cleared old brain data.")

        # Rebuild
        self._build_brain()

        # Update retriever and chain
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        self.rag_chain = self._make_rag_chain()
        self._last_trade_count = self._count_trade_batches()

        print("Brain rebuild complete!")

    def check_for_new_trades(self) -> bool:
        """Check if new trades or learner updates exist and rebuild if needed."""
        # Check trade count
        current_trade_count = len(self._load_trade_batches())
        trade_count_changed = current_trade_count > getattr(self, '_last_trade_total', 0)

        # Check learner state (rebuild every 10 new trades learned)
        learner_state = self._load_learner_state()
        learner_trades = learner_state.get('stats', {}).get('total_trades', 0)
        last_learner_trades = getattr(self, '_last_learner_trades', 0)
        learner_changed = (learner_trades - last_learner_trades) >= 10

        if trade_count_changed or learner_changed:
            reason = []
            if trade_count_changed:
                reason.append(f"new trades ({getattr(self, '_last_trade_total', 0)} -> {current_trade_count})")
            if learner_changed:
                reason.append(f"learner updated ({last_learner_trades} -> {learner_trades} trades)")

            print(f"Brain rebuild triggered: {', '.join(reason)}")
            self.rebuild_brain()

            # Update tracking
            self._last_trade_total = current_trade_count
            self._last_learner_trades = learner_trades
            return True

        return False

    def get_trade_stats(self) -> dict:
        """Get current trade statistics without rebuilding."""
        trades = self._load_trade_batches()
        return self._analyze_trades(trades)

    def _make_rag_chain(self):
        """Creates the RAG chain for both manifesto rules and trade insights."""
        template = """<SYSTEM_INSTRUCTIONS>
You are Snoogans — a Red Bank degenerate 0DTE options trader with a brain full of trading rules AND actual trade history data.

Your knowledge includes:
1. The Snoogans Manifesto (trading rules and guidelines)
2. Historical trade performance data (win rates, P&L by ticker, patterns)

When answering:
- For rule questions: Quote the manifesto directly
- For performance questions: Reference the actual trade data and statistics
- Be specific with numbers when you have them
- Keep the Jersey energy — chill, funny, concise

If the answer isn't in your context, say: "Don't have that data yet, bro. Need more trades."
</SYSTEM_INSTRUCTIONS>

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _parse_date_reference(self, question: str) -> tuple:
        """
        Parse temporal references in a question.

        Returns:
            (start_date, end_date, time_description) or (None, None, None) if no temporal reference
        """
        question_lower = question.lower()
        today = datetime.now().date()

        # Today
        if any(word in question_lower for word in ['today', "today's", 'this morning', 'this afternoon']):
            return (today, today, 'today')

        # Yesterday
        if 'yesterday' in question_lower:
            yesterday = today - timedelta(days=1)
            return (yesterday, yesterday, 'yesterday')

        # This week
        if 'this week' in question_lower:
            start_of_week = today - timedelta(days=today.weekday())
            return (start_of_week, today, 'this week')

        # Last week
        if 'last week' in question_lower:
            start_of_last_week = today - timedelta(days=today.weekday() + 7)
            end_of_last_week = start_of_last_week + timedelta(days=4)  # Friday
            return (start_of_last_week, end_of_last_week, 'last week')

        # Last N days
        match = re.search(r'last\s+(\d+)\s+days?', question_lower)
        if match:
            days = int(match.group(1))
            start = today - timedelta(days=days)
            return (start, today, f'last {days} days')

        # Specific day of week (e.g., "on Friday", "Monday's trades")
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        for i, day in enumerate(days_of_week):
            if day in question_lower:
                # Find the most recent occurrence of that day
                days_back = (today.weekday() - i) % 7
                if days_back == 0 and 'last' in question_lower:
                    days_back = 7
                target_date = today - timedelta(days=days_back)
                return (target_date, target_date, day.capitalize())

        return (None, None, None)

    def _filter_trades_by_date(self, trades: list, start_date, end_date) -> list:
        """Filter trades to those within the date range."""
        filtered = []
        for trade in trades:
            entry_time = trade.get('entry_time', '')
            if not entry_time:
                continue

            try:
                if 'T' in entry_time:
                    trade_date = datetime.fromisoformat(entry_time.replace('Z', '+00:00')).date()
                else:
                    continue

                if start_date <= trade_date <= end_date:
                    filtered.append(trade)
            except (ValueError, TypeError):
                continue

        return filtered

    def _format_trades_summary(self, trades: list, time_desc: str) -> str:
        """Format a summary of trades for the LLM context."""
        if not trades:
            return f"No trades found for {time_desc}."

        # Calculate stats
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
        win_rate = (wins / len(trades) * 100) if trades else 0

        summary = f"TRADES FOR {time_desc.upper()}\n"
        summary += f"Total Trades: {len(trades)}\n"
        summary += f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%\n"
        summary += f"Total P&L: ${total_pnl:+.2f}\n\n"

        summary += "TRADE DETAILS:\n"
        for trade in trades:
            ticker = trade.get('ticker', '???')
            pnl = trade.get('pnl', 0)
            entry = trade.get('entry_time', '')[:16] if trade.get('entry_time') else '???'

            # Determine trade type
            if 'is_call' in trade:
                direction = 'CALL' if trade['is_call'] else 'PUT'
                strike = trade.get('strike_price', 0)
                desc = f"LONG {direction} @ {strike}"
            elif 'is_put' in trade:
                direction = 'PUT SPREAD' if trade['is_put'] else 'CALL SPREAD'
                short = trade.get('short', 0)
                long = trade.get('long', 0)
                desc = f"{direction} {short}/{long}"
            else:
                desc = "UNKNOWN"

            result = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BE"
            exit_reason = trade.get('exit_reason', '')

            summary += f"  {entry} | {ticker} {desc} | ${pnl:+.2f} ({result})"
            if exit_reason:
                summary += f" | Exit: {exit_reason}"
            summary += "\n"

        return summary

    def ask(self, question: str) -> str:
        """
        Smart query router that handles temporal queries and general questions.

        - Temporal queries (today, yesterday, this week): Filter trades and analyze
        - Trade/manifesto questions: Use RAG
        - General chat: Use general LLM
        """
        question_lower = question.lower()

        # Check for temporal reference
        start_date, end_date, time_desc = self._parse_date_reference(question)

        if start_date is not None:
            # Temporal query - filter trades and build focused context
            trades = self._load_trade_batches()
            filtered_trades = self._filter_trades_by_date(trades, start_date, end_date)
            trade_context = self._format_trades_summary(filtered_trades, time_desc)

            # Use LLM with focused trade context
            temporal_template = """You are Snoogans — Red Bank degenerate 0DTE options trader.
Answer questions about trading performance using the data provided.
Be specific with numbers. Keep it chill and concise with Jersey energy.

Trade Data:
{context}

Question: {question}

Answer:"""

            prompt = ChatPromptTemplate.from_template(temporal_template)
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"context": trade_context, "question": question})

        # Check if it's a trade/strategy question (use RAG)
        trade_keywords = ['trade', 'spread', 'scalp', 'entry', 'exit', 'rsi', 'macd',
                         'trend', 'ticker', 'spy', 'qqq', 'iwm', 'win', 'loss', 'pnl',
                         'profit', 'stop', 'target', 'manifesto', 'rule', 'learner',
                         'performance', 'best', 'worst', 'avoid', 'pattern']

        if any(kw in question_lower for kw in trade_keywords):
            return self.ask_rag(question)

        # Default to general chat
        return self.ask_general(question)

    def ask_rag(self, question: str) -> str:
        """Invokes the RAG chain for manifesto rules and trade insights."""
        return self.rag_chain.invoke(question)

    def ask_general(self, question: str) -> str:
        """Handles conversational/general questions without using RAG context."""
        general_template = """You are Snoogans — Red Bank degenerate 0DTE options trader.
You are chill, concise, and funny. Use pure Jersey energy.
Always sign off with "Lunch money secured." or "Snoochie boochies."

User Question: {question}

Answer:"""

        general_prompt = ChatPromptTemplate.from_template(general_template)
        general_llm = ChatOllama(model="snoogans:latest", temperature=0.6, base_url=OLLAMA_HOST)
        general_chain = general_prompt | general_llm | StrOutputParser()

        return general_chain.invoke({"question": question})

    def should_enter_trade(self, ticker: str, is_call: bool, trend: str,
                          rsi: float = None, macd_bull: bool = None,
                          hour: int = None, day_of_week: str = None) -> dict:
        """
        Query historical trade data to decide if entry conditions are favorable.

        Returns dict with:
            - should_enter: bool (True if conditions look favorable)
            - confidence: float (0-100, how confident based on sample size)
            - reason: str (explanation for the decision)
            - stats: dict (relevant statistics)
        """
        stats = self.get_trade_stats()
        if not stats or stats.get('total_trades', 0) < 10:
            return {
                'should_enter': True,  # Default to yes if no data
                'confidence': 0,
                'reason': "Insufficient trade history - need at least 10 trades to make recommendations",
                'stats': {}
            }

        # Collect relevant win rates and sample sizes
        factors = []
        total_weight = 0

        # Check ticker performance
        ticker_data = stats.get('by_ticker', {}).get(ticker)
        if ticker_data and ticker_data['trades'] >= 5:
            ticker_wr = ticker_data['wins'] / ticker_data['trades'] * 100
            factors.append({
                'name': f'{ticker} history',
                'win_rate': ticker_wr,
                'trades': ticker_data['trades'],
                'weight': 2  # Higher weight for ticker-specific data
            })
            total_weight += 2

        # Check trend performance
        if trend:
            trend_data = stats.get('by_trend', {}).get(trend)
            if trend_data and trend_data['trades'] >= 3:
                trend_wr = trend_data['wins'] / trend_data['trades'] * 100
                factors.append({
                    'name': f'{trend} trend',
                    'win_rate': trend_wr,
                    'trades': trend_data['trades'],
                    'weight': 2
                })
                total_weight += 2

        # Check RSI range performance
        if rsi is not None:
            if rsi < 30:
                rsi_range = 'oversold (<30)'
            elif rsi < 40:
                rsi_range = 'low (30-40)'
            elif rsi < 50:
                rsi_range = 'mid-low (40-50)'
            elif rsi < 60:
                rsi_range = 'mid-high (50-60)'
            elif rsi < 70:
                rsi_range = 'high (60-70)'
            else:
                rsi_range = 'overbought (>70)'

            rsi_data = stats.get('by_rsi_range', {}).get(rsi_range)
            if rsi_data and rsi_data['trades'] >= 3:
                rsi_wr = rsi_data['wins'] / rsi_data['trades'] * 100
                factors.append({
                    'name': f'RSI {rsi_range}',
                    'win_rate': rsi_wr,
                    'trades': rsi_data['trades'],
                    'weight': 1
                })
                total_weight += 1

        # Check MACD signal performance
        if macd_bull is not None:
            macd_key = 'MACD_BULL' if macd_bull else 'MACD_BEAR'
            macd_data = stats.get('by_macd', {}).get(macd_key)
            if macd_data and macd_data['trades'] >= 3:
                macd_wr = macd_data['wins'] / macd_data['trades'] * 100
                factors.append({
                    'name': macd_key,
                    'win_rate': macd_wr,
                    'trades': macd_data['trades'],
                    'weight': 1
                })
                total_weight += 1

        # Check hour performance
        if hour is not None:
            hour_data = stats.get('by_hour', {}).get(hour)
            if hour_data and hour_data['trades'] >= 3:
                hour_wr = hour_data['wins'] / hour_data['trades'] * 100
                factors.append({
                    'name': f'{hour}:00 hour',
                    'win_rate': hour_wr,
                    'trades': hour_data['trades'],
                    'weight': 1
                })
                total_weight += 1

        # Check day of week performance
        if day_of_week:
            day_data = stats.get('by_day', {}).get(day_of_week)
            if day_data and day_data['trades'] >= 3:
                day_wr = day_data['wins'] / day_data['trades'] * 100
                factors.append({
                    'name': day_of_week,
                    'win_rate': day_wr,
                    'trades': day_data['trades'],
                    'weight': 1
                })
                total_weight += 1

        # If no relevant data found
        if not factors:
            return {
                'should_enter': True,
                'confidence': 10,
                'reason': "No matching historical patterns found - proceeding with caution",
                'stats': {}
            }

        # Calculate weighted average win rate
        weighted_sum = sum(f['win_rate'] * f['weight'] for f in factors)
        avg_win_rate = weighted_sum / total_weight

        # Calculate confidence based on sample sizes
        total_trades = sum(f['trades'] for f in factors)
        confidence = min(100, total_trades * 2)  # Cap at 100

        # Build reason
        favorable = [f for f in factors if f['win_rate'] >= 40]
        unfavorable = [f for f in factors if f['win_rate'] < 40]

        reasons = []
        if favorable:
            fav_str = ', '.join(f"{item['name']} ({item['win_rate']:.0f}%)" for item in favorable)
            reasons.append(f"Favorable: {fav_str}")
        if unfavorable:
            unfav_str = ', '.join(f"{item['name']} ({item['win_rate']:.0f}%)" for item in unfavorable)
            reasons.append(f"Unfavorable: {unfav_str}")

        # Decision threshold: 35% weighted win rate (since options often have <50% win rate)
        should_enter = avg_win_rate >= 35

        return {
            'should_enter': should_enter,
            'confidence': confidence,
            'reason': ' | '.join(reasons) if reasons else "Mixed signals",
            'avg_win_rate': avg_win_rate,
            'stats': {f['name']: {'win_rate': f['win_rate'], 'trades': f['trades']} for f in factors}
        }

    def get_trade_recommendation(self, ticker: str, is_call: bool, market_context: dict) -> str:
        """Get a human-readable trade recommendation based on historical patterns."""
        result = self.should_enter_trade(
            ticker=ticker,
            is_call=is_call,
            trend=market_context.get('trend'),
            rsi=market_context.get('rsi'),
            macd_bull=market_context.get('macd_bull'),
            hour=market_context.get('hour'),
            day_of_week=market_context.get('day_of_week')
        )

        direction = "LONG CALL" if is_call else "LONG PUT"
        action = "ENTER" if result['should_enter'] else "SKIP"

        msg = f"{action} {ticker} {direction}\n"
        msg += f"Confidence: {result['confidence']:.0f}%\n"
        msg += f"Weighted Win Rate: {result.get('avg_win_rate', 0):.1f}%\n"
        msg += f"Analysis: {result['reason']}"

        return msg

    def analyze_live_bars(self, ticker: str, bars: 'pd.DataFrame') -> dict:
        """
        Analyze live price bars using indicators and return current market context.

        Args:
            ticker: Symbol (SPY, QQQ, IWM)
            bars: DataFrame with OHLCV data (needs 'close', 'high', 'low', 'volume')

        Returns:
            dict with current indicator values and trend signal
        """
        if not INDICATORS_AVAILABLE:
            return {'error': 'Indicators not available'}

        if bars is None or len(bars) < 30:
            return {'error': 'Insufficient bar data (need 30+ bars)'}

        try:
            close = bars['close']
            price = close.iloc[-1]

            # Calculate indicators
            trend = get_trend_signal(bars, None, None)
            rsi_value = _rsi(close)
            macd_line, macd_signal = _macd(close)
            atr_value = _atr(bars)

            now = datetime.now()

            return {
                'ticker': ticker,
                'price': round(price, 2),
                'trend': trend,
                'rsi': round(rsi_value, 2),
                'macd_line': round(macd_line, 4),
                'macd_signal': round(macd_signal, 4),
                'macd_bull': macd_line > macd_signal,
                'atr': round(atr_value, 4),
                'hour': now.hour,
                'day_of_week': now.strftime('%A'),
                'timestamp': now.isoformat()
            }

        except Exception as e:
            return {'error': f'Analysis failed: {e}'}

    def get_live_recommendation(self, ticker: str, bars: 'pd.DataFrame', is_call: bool = None) -> dict:
        """
        Get trade recommendation using live indicator data combined with historical patterns.

        Args:
            ticker: Symbol (SPY, QQQ, IWM)
            bars: DataFrame with OHLCV data
            is_call: True for calls, False for puts, None to auto-detect from trend

        Returns:
            dict with recommendation, indicators, and historical analysis
        """
        # Get live indicator analysis
        live = self.analyze_live_bars(ticker, bars)

        if 'error' in live:
            return {'should_enter': False, 'error': live['error']}

        # Auto-detect direction from trend if not specified
        if is_call is None:
            if live['trend'] == 'bull':
                is_call = True
            elif live['trend'] == 'bear':
                is_call = False
            else:
                return {
                    'should_enter': False,
                    'reason': 'Trend is CHOP - no clear direction',
                    'live_indicators': live,
                    'direction': None
                }

        # Get historical pattern recommendation
        historical = self.should_enter_trade(
            ticker=ticker,
            is_call=is_call,
            trend=live['trend'],
            rsi=live['rsi'],
            macd_bull=live['macd_bull'],
            hour=live['hour'],
            day_of_week=live['day_of_week']
        )

        # Combine live indicators with historical patterns
        direction = "LONG CALL" if is_call else "LONG PUT"

        # Additional live indicator checks
        indicator_warnings = []

        # RSI extremes
        if is_call and live['rsi'] > 70:
            indicator_warnings.append("RSI overbought (>70) - risky for calls")
        elif not is_call and live['rsi'] < 30:
            indicator_warnings.append("RSI oversold (<30) - risky for puts")

        # Trend/direction mismatch
        if is_call and live['trend'] == 'bear':
            indicator_warnings.append("Trend is BEAR but buying calls")
        elif not is_call and live['trend'] == 'bull':
            indicator_warnings.append("Trend is BULL but buying puts")

        # MACD divergence
        if is_call and not live['macd_bull']:
            indicator_warnings.append("MACD bearish crossover")
        elif not is_call and live['macd_bull']:
            indicator_warnings.append("MACD bullish crossover")

        # Final decision: combine historical and live
        live_ok = len(indicator_warnings) == 0
        final_enter = historical['should_enter'] and live_ok

        # If historical says no but live looks good, still skip (trust history)
        # If historical says yes but live has warnings, skip (trust live)

        return {
            'should_enter': final_enter,
            'direction': direction,
            'ticker': ticker,
            'live_indicators': live,
            'historical_analysis': {
                'should_enter': historical['should_enter'],
                'confidence': historical['confidence'],
                'avg_win_rate': historical.get('avg_win_rate', 0),
                'reason': historical['reason']
            },
            'indicator_warnings': indicator_warnings,
            'final_reason': self._build_final_reason(historical, live_ok, indicator_warnings)
        }

    def _build_final_reason(self, historical: dict, live_ok: bool, warnings: list) -> str:
        """Build human-readable final recommendation reason."""
        parts = []

        if historical['should_enter']:
            parts.append(f"History: ENTER ({historical.get('avg_win_rate', 0):.0f}% win rate)")
        else:
            parts.append(f"History: SKIP ({historical['reason']})")

        if live_ok:
            parts.append("Live: All indicators aligned")
        else:
            parts.append(f"Live: WARNINGS - {', '.join(warnings)}")

        return " | ".join(parts)

    def get_optimal_exits(self, ticker: str = None, trend: str = None,
                          rsi_range: str = None, trade_type: str = None) -> dict:
        """
        Analyze historical trades matching pattern to determine optimal exit parameters.

        Args:
            ticker: Filter by ticker (SPY, QQQ, IWM)
            trend: Filter by trend at entry (bull, bear)
            rsi_range: Filter by RSI range (e.g., 'mid-high (50-60)')
            trade_type: Filter by type (LONG_CALL, LONG_PUT, PUT_SPREAD, CALL_SPREAD)

        Returns:
            dict with:
            - profit_target_pct: Suggested profit target (% of credit/debit)
            - stop_loss_pct: Suggested stop loss (% of credit/debit)
            - confidence: 0-100 based on sample size
            - sample_size: Number of trades analyzed
            - analysis: Detailed breakdown
        """
        trades = self._load_trade_batches()

        if not trades:
            return {
                'profit_target_pct': 50.0,  # Default 50% profit
                'stop_loss_pct': 100.0,     # Default 100% loss (2x credit)
                'confidence': 0,
                'sample_size': 0,
                'analysis': 'No trade history available'
            }

        # Filter trades matching pattern
        filtered = []
        for trade in trades:
            # Skip trades without learning metrics
            if 'max_profit_pct' not in trade or 'exit_reason' not in trade:
                continue

            # Apply filters
            if ticker and trade.get('ticker') != ticker:
                continue

            ctx = trade.get('market_context', {})
            if trend and ctx.get('trend') != trend:
                continue

            if rsi_range:
                rsi = ctx.get('rsi')
                if rsi is None:
                    continue
                # Map RSI to range
                if rsi < 30:
                    t_rsi_range = 'oversold (<30)'
                elif rsi < 40:
                    t_rsi_range = 'low (30-40)'
                elif rsi < 50:
                    t_rsi_range = 'mid-low (40-50)'
                elif rsi < 60:
                    t_rsi_range = 'mid-high (50-60)'
                elif rsi < 70:
                    t_rsi_range = 'high (60-70)'
                else:
                    t_rsi_range = 'overbought (>70)'
                if t_rsi_range != rsi_range:
                    continue

            if trade_type:
                if 'is_call' in trade:
                    t_type = 'LONG_CALL' if trade['is_call'] else 'LONG_PUT'
                elif 'is_put' in trade:
                    t_type = 'PUT_SPREAD' if trade['is_put'] else 'CALL_SPREAD'
                else:
                    continue
                if t_type != trade_type:
                    continue

            filtered.append(trade)

        sample_size = len(filtered)

        if sample_size < 5:
            return {
                'profit_target_pct': 50.0,
                'stop_loss_pct': 100.0,
                'confidence': min(sample_size * 10, 40),
                'sample_size': sample_size,
                'analysis': f'Insufficient data ({sample_size} trades) - using defaults'
            }

        # Analyze winners vs losers
        winners = [t for t in filtered if t.get('pnl', 0) > 0]
        losers = [t for t in filtered if t.get('pnl', 0) < 0]

        # Calculate optimal profit target
        # Look at max_profit_pct of winners - where did they peak before exit?
        if winners:
            max_profits = [t.get('max_profit_pct', 0) for t in winners]
            avg_max_profit = sum(max_profits) / len(max_profits)

            # Also look at actual exit profit
            actual_profits = []
            for t in winners:
                credit_or_debit = t.get('credit', t.get('debit', 1))
                if credit_or_debit > 0:
                    actual_pct = t.get('pnl', 0) / (credit_or_debit * t.get('contracts', 10) * 100) * 100
                    actual_profits.append(actual_pct)

            avg_actual_profit = sum(actual_profits) / len(actual_profits) if actual_profits else 0

            # Money left on table
            left_on_table = avg_max_profit - avg_actual_profit

            # Suggest profit target that captures more of the max profit
            # If leaving too much on table, lower the target slightly
            if left_on_table > 20:
                suggested_profit_target = max(30, avg_actual_profit - 10)
            else:
                suggested_profit_target = avg_actual_profit
        else:
            suggested_profit_target = 50.0
            avg_max_profit = 0
            left_on_table = 0

        # Calculate optimal stop loss
        # Look at max_drawdown_pct of losers - could we have cut losses earlier?
        if losers:
            max_drawdowns = [t.get('max_drawdown_pct', 0) for t in losers]
            avg_max_drawdown = sum(max_drawdowns) / len(max_drawdowns)

            # Look at exit reasons for losers
            stop_loss_exits = [t for t in losers if t.get('exit_reason') == 'stop_loss']
            time_exits = [t for t in losers if t.get('exit_reason') in ['time_exit', 'eod_force_close']]

            # If many losers hit stop_loss, the stop is working
            # If many losers exit via time with big drawdowns, stop should be tighter
            if time_exits and len(time_exits) > len(stop_loss_exits):
                # Too many losers running to time exit with big losses
                suggested_stop_loss = max(50, avg_max_drawdown * 0.7)  # Tighten stop
            else:
                # Stop loss is catching most losers - keep similar
                suggested_stop_loss = 100.0  # Default 2x credit
        else:
            suggested_stop_loss = 100.0
            avg_max_drawdown = 0

        # Calculate confidence based on sample size
        confidence = min(100, sample_size * 4)  # 25 trades = 100% confidence

        # Build analysis summary
        analysis_parts = [
            f"Analyzed {sample_size} trades",
            f"Winners: {len(winners)}, Losers: {len(losers)}",
            f"Win rate: {len(winners)/sample_size*100:.0f}%"
        ]

        if winners:
            analysis_parts.append(f"Avg max profit seen: {avg_max_profit:.0f}%")
            analysis_parts.append(f"Money left on table: {left_on_table:.0f}%")

        if losers:
            analysis_parts.append(f"Avg max drawdown: {avg_max_drawdown:.0f}%")

        return {
            'profit_target_pct': round(suggested_profit_target, 1),
            'stop_loss_pct': round(suggested_stop_loss, 1),
            'confidence': confidence,
            'sample_size': sample_size,
            'win_count': len(winners),
            'loss_count': len(losers),
            'avg_max_profit_seen': round(avg_max_profit, 1) if winners else 0,
            'avg_max_drawdown_seen': round(avg_max_drawdown, 1) if losers else 0,
            'left_on_table_pct': round(left_on_table, 1) if winners else 0,
            'analysis': ' | '.join(analysis_parts)
        }

    def _normalize_features(self, features: dict) -> dict:
        """
        Normalize features to 0-1 scale for distance calculation.
        """
        normalized = {}

        # RSI: 0-100 -> 0-1
        if 'rsi' in features:
            normalized['rsi'] = features['rsi'] / 100.0

        # MACD line: typically -1 to +1, clamp and scale
        if 'macd_line' in features:
            normalized['macd_line'] = (max(-1, min(1, features['macd_line'])) + 1) / 2.0

        # MACD signal: same as line
        if 'macd_signal' in features:
            normalized['macd_signal'] = (max(-1, min(1, features['macd_signal'])) + 1) / 2.0

        # ATR: typically 0-5 for SPY, scale accordingly
        if 'atr' in features:
            normalized['atr'] = min(1.0, features['atr'] / 5.0)

        # Trend strength: typically -1 to +1
        if 'trend_strength' in features:
            normalized['trend_strength'] = (max(-1, min(1, features['trend_strength'])) + 1) / 2.0

        # Hour: 9-16 -> 0-1
        if 'hour' in features:
            normalized['hour'] = (features['hour'] - 9) / 7.0

        # Trend: categorical -> one-hot encoding
        if 'trend' in features:
            trend = features['trend']
            normalized['trend_bull'] = 1.0 if trend == 'bull' else 0.0
            normalized['trend_bear'] = 1.0 if trend == 'bear' else 0.0

        # MACD bull: boolean -> 0 or 1
        if 'macd_bull' in features:
            normalized['macd_bull_flag'] = 1.0 if features['macd_bull'] else 0.0

        return normalized

    def _euclidean_distance(self, f1: dict, f2: dict) -> float:
        """
        Calculate Euclidean distance between two normalized feature vectors.
        """
        common_keys = set(f1.keys()) & set(f2.keys())
        if not common_keys:
            return float('inf')

        sum_sq = 0.0
        for key in common_keys:
            diff = f1[key] - f2[key]
            sum_sq += diff * diff

        return sum_sq ** 0.5

    def find_similar_trades(self, features: dict, k: int = 10,
                            max_history: int = 150) -> list:
        """
        Find k most similar historical trades using feature-based distance.

        Args:
            features: Current market context features
            k: Number of similar trades to return
            max_history: How many recent trades to consider

        Returns:
            List of (distance, trade) tuples sorted by similarity
        """
        trades = self._load_trade_batches()
        if not trades:
            return []

        # Use most recent trades
        recent_trades = trades[-max_history:]

        # Normalize current features
        normalized_current = self._normalize_features(features)

        distances = []
        for trade in recent_trades:
            ctx = trade.get('market_context', {})
            if not ctx:
                continue

            # Extract and normalize trade features
            trade_features = self._normalize_features(ctx)

            # Calculate distance
            dist = self._euclidean_distance(normalized_current, trade_features)

            if dist < float('inf'):
                distances.append((dist, trade))

        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[0])

        return distances[:k]

    def get_pattern_stats(self, similar_trades: list) -> dict:
        """
        Compute statistics from a list of similar trades.

        Args:
            similar_trades: List of (distance, trade) tuples from find_similar_trades()

        Returns:
            dict with win_rate, avg_pnl, pattern insights
        """
        if not similar_trades:
            return None

        # Extract just the trades
        trades = [t for _, t in similar_trades]

        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_pnl = total_pnl / len(trades)

        # Get average distance (similarity)
        avg_distance = sum(d for d, _ in similar_trades) / len(similar_trades)

        # Analyze exit reasons
        exit_reasons = defaultdict(int)
        for t in trades:
            reason = t.get('exit_reason', 'unknown')
            exit_reasons[reason] += 1

        # Find most common exit
        most_common_exit = max(exit_reasons.items(), key=lambda x: x[1])[0] if exit_reasons else 'unknown'

        # Average hold duration
        hold_durations = [t.get('hold_duration_minutes', 0) for t in trades
                        if 'hold_duration_minutes' in t]
        avg_hold = sum(hold_durations) / len(hold_durations) if hold_durations else 0

        return {
            'sample_size': len(trades),
            'win_rate': wins / len(trades) * 100,
            'wins': wins,
            'losses': losses,
            'avg_pnl': round(avg_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_similarity': round(1 - min(1, avg_distance), 2),  # Convert distance to similarity
            'avg_hold_minutes': round(avg_hold, 1),
            'common_exit': most_common_exit,
            'exit_breakdown': dict(exit_reasons),
            'confidence': min(100, len(trades) * 10)  # 10 trades = 100% confidence
        }

    def get_pattern_recommendation(self, features: dict, k: int = 10) -> dict:
        """
        Get trade recommendation based on similar historical patterns.

        Args:
            features: Current market context (rsi, macd_line, trend, etc.)
            k: Number of similar trades to analyze

        Returns:
            dict with recommendation, confidence, and pattern analysis
        """
        similar = self.find_similar_trades(features, k=k)

        if len(similar) < 3:
            return {
                'should_enter': True,  # Default to yes if insufficient data
                'confidence': 0,
                'reason': f"Insufficient similar patterns ({len(similar)} found)",
                'pattern_stats': None
            }

        stats = self.get_pattern_stats(similar)

        # Decision based on win rate and sample size
        win_rate = stats['win_rate']
        sample = stats['sample_size']

        # Adjust threshold based on sample size
        if sample >= 10:
            threshold = 35  # Full confidence threshold
        elif sample >= 5:
            threshold = 30  # Lower threshold for smaller samples
        else:
            threshold = 25  # Very low threshold for minimal data

        should_enter = win_rate >= threshold

        # Build reason
        if should_enter:
            reason = f"Similar patterns: {win_rate:.0f}% win rate over {sample} trades (avg P&L ${stats['avg_pnl']:+.2f})"
        else:
            reason = f"Similar patterns underperform: {win_rate:.0f}% win rate over {sample} trades"

        return {
            'should_enter': should_enter,
            'confidence': stats['confidence'],
            'reason': reason,
            'pattern_stats': stats,
            'similar_trades_count': len(similar)
        }
