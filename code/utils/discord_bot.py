# discord-bot.py – Snoogans Discord Bot v3 (alerts + @mention LLM responses)
import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM  # 2025 fixed
from langchain_core.prompts import PromptTemplate
import requests
import datetime
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

# Secrets
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Add this to .env – bot token from Discord Dev Portal
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not all([DISCORD_TOKEN, DISCORD_WEBHOOK_URL, ALPACA_API_KEY, ALPACA_SECRET_KEY]):
    raise ValueError("Yo, missing secrets in .env – need DISCORD_BOT_TOKEN + others, degenerates!")

# LLM – local supertrader model
llm = OllamaLLM(model="supertrader:latest", temperature=0.7)

rant_prompt = PromptTemplate.from_template(
    "Respond like a Red Bank degenerate to this: {details}"
)
rant_chain = rant_prompt | llm

def get_rant(details: str) -> str:
    return rant_chain.invoke({"details": details})

# Webhook sender for alerts
def send_webhook(message: str):
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
    except Exception as e:
        print(f"Webhook failed: {e}")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Snoogans logged in as {bot.user} – ready to bank theta in Discord")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Respond to @Snoogans mentions
    if bot.user in message.mentions:
        user_prompt = message.content.replace(f"<@{bot.user.id}>", "").strip()
        prompt = user_prompt or "Yo, what's good?"
        with message.channel.typing():
            rant = get_rant(prompt)
        await message.reply(f"{rant}\n\nsnoochie boochies!")

    await bot.process_commands(message)

# === LLM-Powered Alerts (call these from your trading bot) ===
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def calculate_pivots(high, low, close):
    pp = (high + low + close) / 3
    return {k: round(v, 2) for k, v in {
        "PP": pp, "R1": 2*pp-low, "S1": 2*pp-high,
        "R2": pp + (high-low), "S2": pp - (high-low),
        "R3": pp + 2*(high-low), "S3": pp - 2*(high-low)
    }.items()}

def get_yesterdays_spx():
    est = ZoneInfo("America/New_York")
    end = datetime.datetime.now(est).date() - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=20)
    for sym in ["SPX", "^GSPC"]:
        try:
            bars = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                start=start, end=end, adjustment="all"
            )).df
            if not bars.empty:
                latest = bars.iloc[-1]
                return {"high": latest["high"], "low": latest["low"], "close": latest["close"]}
        except: pass
    raise ValueError("No SPX data")

def send_greeting():
    pivots = calculate_pivots(**get_yesterdays_spx())
    pivot_str = f"PP {pivots['PP']} | R1 {pivots['R1']} S1 {pivots['S1']} | R2 {pivots['R2']} S2 {pivots['S2']} | R3 {pivots['R3']} S3 {pivots['S3']}"
    rant = get_rant(f"Morning pivots: {pivot_str}. Tiny vertical credit spreads incoming.")
    msg = f"🌅 **MORNING GREETING** 🌅\n{rant}\n\nsnoochie boochies!"
    send_webhook(msg)

def send_entry(is_put=True, short=6820, long=6810, credit=2.50, underlying="SPX"):
    typ = "put" if is_put else "call"
    rant = get_rant(f"Sold tiny {typ} credit spread {short}/{long} for {credit} credit on {underlying}")
    msg = f"🚀 **ENTRY** 🚀\n{rant}\n\nlunch money secured, snoochie boochies!"
    send_webhook(msg)

def send_exit(is_put=True, short=6820, long=6810, credit=2.50, pnl=1.25, underlying="SPX"):
    typ = "put" if is_put else "call"
    result = "WIN" if pnl > 0 else "LOSS"
    rant = get_rant(f"Closed {typ} spread {short}/{long}, original {credit}, P/L {pnl:+.2f}")
    msg = f"💰 **EXIT – {result}** 💰\n{rant}\n\nSnoogans out, snoochie boochies!"
    send_webhook(msg)

# Run bot
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)