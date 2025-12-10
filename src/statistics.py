import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from typing import Tuple

class StatEngine:
    """
    Performs statistical tests like Cointegration and Z-Score calculation.
    """
    
    @staticmethod
    def check_cointegration(series_a: pd.Series, series_b: pd.Series) -> Tuple[float, float, bool]:
        """
        Performs the Engle-Granger two-step cointegration test.
        Returns: (p-value, t-score, is_cointegrated)
        """
        score, pvalue, _ = coint(series_a, series_b)
        is_coint = pvalue < 0.05
        return pvalue, score, is_coint

    @staticmethod
    def calculate_spread_and_zscore(series_a: pd.Series, series_b: pd.Series) -> pd.DataFrame:
        """
        Calculates the spread using OLS regression (Hedge Ratio) and normalizes it to a Z-Score.
        Spread = Y - (beta * X)
        """
        # Calculate Hedge Ratio using OLS
        X = sm.add_constant(series_b)
        result = sm.OLS(series_a, X).fit()
        beta = result.params.iloc[1]
        
        spread = series_a - beta * series_b
        z_score = (spread - spread.mean()) / spread.std()
        
        df = pd.DataFrame({'spread': spread, 'z_score': z_score})
        return df, beta