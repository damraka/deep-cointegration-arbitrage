import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.data_loader import DataLoader
from src.statistics import StatEngine
from src.environment import PairsTradingEnv

def backtest():
    # 1. Test Data Configuration
    TICKER_A = "GLD"
    TICKER_B = "GDX"
    START = "2024-01-01"  # Training was up to 2024-01-01
    END = "2024-12-01"

    # 2. Pull Data
    loader = DataLoader(TICKER_A, TICKER_B, START, END)
    df = loader.fetch_data()
    
    # 3. Calculate Statistics
    stats_df, beta = StatEngine.calculate_spread_and_zscore(df[TICKER_A], df[TICKER_B])

    # 4. Ready Environment
    env = PairsTradingEnv(
        z_scores=stats_df['z_score'],
        prices_a=df[TICKER_A],
        prices_b=df[TICKER_B]
    )

    # 5. Load Trained Model
    model = PPO.load("ppo_pairs_trader")

    # 6. Backtest Loop
    obs, _ = env.reset()
    done = False
    portfolio_values = []
    
    print("Backtest is starting...")
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Save current portfolio value
        portfolio_values.append(env.balance)

    # 7. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='RL Agent Portfolio', color='purple')
    plt.axhline(y=10000, color='r', linestyle='--', label='Initial Capital')
    plt.title(f'Pairs Trading RL Agent Performance ({START} - {END})')
    plt.xlabel('Time Steps (Days)')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    backtest()