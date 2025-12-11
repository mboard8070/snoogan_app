# /project/code/trading_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime
from polygon import RESTClient
import os
from dotenv import load_dotenv

load_dotenv()

class TradingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Action: 0 = do nothing, 1 = sell 16-delta put credit spread
        self.action_space = spaces.Discrete(2)
        
        # Observation: [SPX price, avg IV, hours until close]
        low = np.array([0, 0, 0], dtype=np.float32)
        high = np.array([10000, 2.0, 24], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        self.client = RESTClient(api_key=os.getenv("POLYGON_API_KEY"))
        self.current_step = 0
        self.max_steps = 390  # trading day in minutes

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Get SPX price (fallback safe)
        try:
            aggs = self.client.get_aggs("SPX", 1, "day", limit=1)
            price = aggs[0].close if aggs else 6000.0
        except:
            price = 6000.0

        # Get 0DTE puts
        try:
            contracts = self.client.list_options_contracts(
                underlying_ticker="SPX",
                expiration_date=datetime.now().strftime("%Y-%m-%d"),
                contract_type="put",
                limit=500
            )
            df = pd.DataFrame([{
                "strike": c.strike_price,
                "delta": getattr(c, "delta", 0.16),
                "iv": getattr(c, "implied_volatility", 0.20)
            } for c in contracts])
            df = df.dropna()
            avg_iv = df['iv'].mean() if not df.empty else 0.20
        except:
            avg_iv = 0.20

        # Hours until 4pm ET
        now = datetime.now()
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        hours_left = max(0, (market_close - now).total_seconds() / 3600)

        return np.array([price, avg_iv, hours_left], dtype=np.float32)

    def step(self, action):
        reward = 0
        if action == 1:
            reward = 50  # fake credit received

        self.current_step += 1
        done = self.current_step >= self.max_steps

        obs = self._get_obs()
        return obs, reward, done, False, {}

    def render(self):
        obs = self._get_obs()
        print(f"SPX: ${obs[0]:.0f} | IV: {obs[1]:.1%} | Time left: {obs[2]:.1f}h")