import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    Technical Indicators Calculator for Stock Time Series Data
    """

    @staticmethod
    def rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI) using Simple Moving Average
        """
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def stochastic_rsi(close_prices: pd.Series, period: int = 14, stoch_period: int = 14) -> pd.Series:
        """
        Calculate Stochastic RSI
        """
        rsi_values = TechnicalIndicators.rsi(close_prices, period)
        
        # Calculate Stochastic of RSI
        rsi_min = rsi_values.rolling(window=stoch_period).min()
        rsi_max = rsi_values.rolling(window=stoch_period).max()
        
        stoch_rsi = (rsi_values - rsi_min) / (rsi_max - rsi_min) * 100
        return stoch_rsi

    @staticmethod
    def macd(close_prices: pd.Series, fast_period: int = 12, slow_period: int = 26, 
             signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        """
        # Calculate EMAs
        ema_fast = close_prices.ewm(span=fast_period).mean()
        ema_slow = close_prices.ewm(span=slow_period).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=signal_period).mean()

        # MACD histogram
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_signal': signal_line,
            'MACD_histogram': histogram
        })

    @staticmethod
    def moving_average(close_prices: pd.Series, period: int = 16) -> pd.Series:
        """
        Calculate Simple Moving Average
        """
        return close_prices.rolling(window=period).mean()

    @staticmethod
    def rate_of_change(close_prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change (ROC)
        """
        roc = ((close_prices - close_prices.shift(period)) / close_prices.shift(period)) * 100
        return roc

    @staticmethod
    def williams_r(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series, 
                   period: int = 14) -> pd.Series:
        """
        Calculate Williams %R
        """
        highest_high = high_prices.rolling(window=period).max()
        lowest_low = low_prices.rolling(window=period).min()

        williams_r = ((highest_high - close_prices) / (highest_high - lowest_low)) * -100
        return williams_r

    @staticmethod
    def vwma(close_prices: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average (VWMA)
        """
        volume_price = close_prices * volume
        volume_sum = volume.rolling(window=period).sum()
        volume_price_sum = volume_price.rolling(window=period).sum()

        vwma = volume_price_sum / volume_sum
        return vwma

    @staticmethod
    def linear_regression_ma(close_prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Linear Regression Moving Average
        """
        def calculate_lr_value(y):
            # y is a numpy array (because raw=True)
            if len(y) < period or np.any(np.isnan(y)):
                return np.nan
            
            # Create x values (time index)
            x = np.arange(len(y))
            
            # Calculate linear regression
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            # Linear regression formula
            denominator = (n * sum_x2 - sum_x * sum_x)
            if denominator == 0:
                return np.nan
                
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            # Return the predicted value for the last point
            return slope * (n - 1) + intercept

        # Use raw=True for better performance and to pass numpy arrays
        lr_ma = close_prices.rolling(window=period).apply(calculate_lr_value, raw=True)
        return lr_ma

    @staticmethod
    def volatility(close_prices: pd.Series, period: int = 21) -> pd.Series:
        """
        Calculate Historical Volatility (Standard Deviation of Log Returns)
        """
        log_returns = np.log(close_prices / close_prices.shift(1))
        volatility = log_returns.rolling(window=period).std()
        return volatility

