import yfinance as yf
import pandas as pd
from typing import Tuple

class DataLoader:
    """
    Handles data fetching and preprocessing for financial time series.
    """
    def __init__(self, ticker_a: str, ticker_b: str, start_date: str, end_date: str):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.start = start_date
        self.end = end_date

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches adjusted close prices and calculates log returns.
        NOTE: yfinance auto_adjust=True by default now, so 'Close' is the adjusted close.
        """
        print(f"Fetching data for {self.ticker_a} and {self.ticker_b}...")
        
        # Using 'close' instead of 'adj close' as yfinance now returns adjusted close by default
        raw_data = yf.download([self.ticker_a, self.ticker_b], start=self.start, end=self.end)
        
        # Column check
        if 'Adj Close' in raw_data.columns:
            data = raw_data['Adj Close']
        else:
            data = raw_data['Close']
        
        # Drop missing values
        data.dropna(inplace=True)
        
        # Rename columns for clarity
        data.columns = [self.ticker_a, self.ticker_b]
        
        return data