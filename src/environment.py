import gymnasium as gym
import numpy as np
from gymnasium import spaces

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class PairsTradingEnv(gym.Env):
    """
    Advanced Pairs Trading Environment with Sharpe Ratio Reward Shaping.
    """
    
    def __init__(self, z_scores, prices_a, prices_b, initial_balance=10000):
        super(PairsTradingEnv, self).__init__()
        
        self.z_scores = z_scores.values
        self.prices_a = prices_a.values
        self.prices_b = prices_b.values
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.position = 0 # 0: None, 1: Long, -1: Short
        self.entry_price_diff = 0
        self.portfolio_history = [] # Sharpe hesabı için geçmişi tutacağız
        
        # Action space: 0=Hold, 1=Buy Spread, 2=Sell Spread
        self.action_space = spaces.Discrete(3)
        
        # Observation: [Z-Score, Position, Norm. Balance, Volatility Proxy]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.balance = self.initial_balance
        self.entry_price_diff = 0
        self.portfolio_history = [self.initial_balance]
        return self._next_observation(), {}

    def _next_observation(self):
        # Volatiliteyi (son 5 adımın z-score değişimi) de ajana gösterelim
        volatility = 0
        if self.current_step > 5:
            volatility = np.std(self.z_scores[self.current_step-5:self.current_step])

        obs = np.array([
            self.z_scores[self.current_step],
            self.position,
            self.balance / self.initial_balance,
            volatility
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.z_scores) - 1
        
        current_spread_price = self.prices_a[self.current_step] - self.prices_b[self.current_step]
        step_reward = 0

        # --- Trading Logic ---
        if action == 1: # Long Spread
            if self.position == 0:
                self.position = 1
                self.entry_price_diff = current_spread_price
            elif self.position == -1: # Close Short
                profit = self.entry_price_diff - current_spread_price
                self.balance += profit
                step_reward = profit
                self.position = 0
                
        elif action == 2: # Short Spread
            if self.position == 0:
                self.position = -1
                self.entry_price_diff = current_spread_price
            elif self.position == 1: # Close Long
                profit = current_spread_price - self.entry_price_diff
                self.balance += profit
                step_reward = profit
                self.position = 0
        
        # Pozisyonu kapattıysa küçük bir işlem maliyeti (Slippage/Commission) düşelim
        if step_reward != 0:
            step_reward -= 0.5 # İşlem başına ceza
        
        # Pozisyonda bekliyorsa çok küçük ceza (Parayı bağlama maliyeti)
        if self.position != 0:
            step_reward -= 0.01

        # --- REWARD SHAPING (Sihirli Dokunuş) ---
        # Sadece kâra bakma, portföyün ne kadar stabil olduğuna bak.
        self.portfolio_history.append(self.balance)
        
        # Eğer portföy başlangıçtan aşağı düştüyse ekstra ceza ver (Drawdown penalty)
        if self.balance < self.initial_balance:
            step_reward -= 0.1

        # Sharpe Ratio Reward (Basitleştirilmiş):
        # Getiri / Standart Sapma
        returns = np.diff(self.portfolio_history[-20:]) # Son 20 adımın getirisi
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns)
            step_reward += sharpe * 0.1 # Sharpe oranını ödüle ekle

        return self._next_observation(), step_reward, done, False, {}