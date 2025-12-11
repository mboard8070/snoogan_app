# code/utils/discord_bot.py — clean, final version
import requests
import os
from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL")

def berserker_yell(message: str):
    if not DISCORD_WEBHOOK:
        print("No webhook — yelling locally:")
        print(message)
        return

    payload = {
        "username": "Snoogans",
        "content": message
    }

    try:
        requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        print("Snoogans spoke — face already set in Discord")
    except Exception as e:
        print(f"Discord failed: {e}")

# Test
if __name__ == "__main__":
    berserker_yell(
        "**SNOOGANS IS FULLY FORMED** 😤\n\n"
        "Face: Kevin Smith energy\n"
        "Voice: pure Jersey\n"
        "Strategy: tiny vertical credit spreads\n"
        "Risk: defined\n"
        "Goal: compound forever\n\n"
        "We are now complete.\n"
        "Snoochie boochies.\n"
        "37."
    )
    
# Add this to your existing discord_bot.py
from utils.pivot_calculator import get_pivots

def post_daily_pivots():
    message = get_pivots()
    post_to_discord(message)