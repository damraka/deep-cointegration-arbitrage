from src.data_loader import DataLoader
from src.statistics import StatEngine
from src.environment import PairsTradingEnv
from stable_baselines3 import PPO

def main():
    # 1. Configuration
    TICKER_A = "GLD" # Gold ETF
    TICKER_B = "GDX" # Gold Miners ETF
    START = "2020-01-01"
    END = "2024-01-01"

    # 2. Data Pipeline
    loader = DataLoader(TICKER_A, TICKER_B, START, END)
    df = loader.fetch_data()

    # 3. Statistical Check
    p_val, _, is_coint = StatEngine.check_cointegration(df[TICKER_A], df[TICKER_B])
    print(f"Cointegration p-value: {p_val:.4f} | Cointegrated: {is_coint}")

    if not is_coint:
        print("Warning: Pairs are not cointegrated. Statistical arbitrage might fail.")

    # 4. Feature Engineering (Z-Scores)
    stats_df, beta = StatEngine.calculate_spread_and_zscore(df[TICKER_A], df[TICKER_B])
    
    # 5. Initialize RL Environment
    env = PairsTradingEnv(
        z_scores=stats_df['z_score'],
        prices_a=df[TICKER_A],
        prices_b=df[TICKER_B]
    )

    # 6. Train Agent (PPO Algorithm)
    print("Starting training with PPO...")
    model = PPO("MlpPolicy", env, verbose=1)

    # Train the agent for 50,000 timesteps (adjust as needed)
    model.learn(total_timesteps=50000)
    
    print("Training finished. Saving model...")
    model.save("ppo_pairs_trader")

if __name__ == "__main__":
    main()