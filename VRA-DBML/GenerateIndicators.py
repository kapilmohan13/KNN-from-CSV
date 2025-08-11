import os

import pandas as pd
import numpy as np
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    Technical Indicators Calculator for Stock Time Series Data
    """

    @staticmethod
    def rsi(close_prices: pd.Series, period: int = 6) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Parameters:
        close_prices (pd.Series): Close prices
        period (int): RSI period (default 6)

        Returns:
        pd.Series: RSI values
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

        Parameters:
        close_prices (pd.Series): Close prices
        period (int): RSI period
        stoch_period (int): Stochastic period

        Returns:
        pd.Series: Stochastic RSI values
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

        Parameters:
        close_prices (pd.Series): Close prices
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line EMA period

        Returns:
        pd.DataFrame: MACD line, Signal line, and Histogram
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

        Parameters:
        close_prices (pd.Series): Close prices
        period (int): Moving average period

        Returns:
        pd.Series: Moving average values
        """
        return close_prices.rolling(window=period).mean()

    @staticmethod
    def rate_of_change(close_prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change (ROC)

        Parameters:
        close_prices (pd.Series): Close prices
        period (int): ROC period

        Returns:
        pd.Series: ROC values as percentage
        """
        roc = ((close_prices - close_prices.shift(period)) / close_prices.shift(period)) * 100
        return roc

    @staticmethod
    def williams_r(high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series,
                   period: int = 14) -> pd.Series:
        """
        Calculate Williams %R (Williams Overbought/Oversold Index)

        Parameters:
        high_prices (pd.Series): High prices
        low_prices (pd.Series): Low prices
        close_prices (pd.Series): Close prices
        period (int): Williams %R period

        Returns:
        pd.Series: Williams %R values
        """
        highest_high = high_prices.rolling(window=period).max()
        lowest_low = low_prices.rolling(window=period).min()

        williams_r = ((highest_high - close_prices) / (highest_high - lowest_low)) * -100
        return williams_r

    @staticmethod
    def vwma(close_prices: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Volume Weighted Moving Average (VWMA)

        Parameters:
        close_prices (pd.Series): Close prices
        volume (pd.Series): Volume data (using volatility as proxy)
        period (int): VWMA period

        Returns:
        pd.Series: VWMA values
        """
        # Use volatility as volume weight (higher volatility = higher weight)
        volume_price = close_prices * volume
        volume_sum = volume.rolling(window=period).sum()
        volume_price_sum = volume_price.rolling(window=period).sum()

        vwma = volume_price_sum / volume_sum
        return vwma

    @staticmethod
    def linear_regression_ma(close_prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Linear Regression Moving Average

        Parameters:
        close_prices (pd.Series): Close prices
        period (int): Linear regression period

        Returns:
        pd.Series: Linear Regression MA values
        """

        def calculate_lr_value(prices_window):
            if len(prices_window) < period:
                return np.nan

            # Create x values (time index)
            x = np.arange(len(prices_window))
            y = prices_window.values

            # Calculate linear regression
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)

            # Linear regression formula
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n

            # Return the predicted value for the last point
            return slope * (n - 1) + intercept

        lr_ma = close_prices.rolling(window=period).apply(calculate_lr_value, raw=False)
        return lr_ma


class StockDataProcessor:
    """
    Main processor for stock data with technical indicators
    """

    def __init__(self):
        self.required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'v']

    def load_and_validate_data(self, input_file: str) -> Optional[pd.DataFrame]:
        """
        Load and validate the input CSV file

        Parameters:
        input_file (str): Path to input CSV file

        Returns:
        pd.DataFrame or None: Validated dataframe
        """
        try:
            df = pd.read_csv(input_file)
            print(f"Loaded data from {input_file}")
            print(f"Initial data shape: {df.shape}")

            # Check required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            print(f"Required columns found: {self.required_columns}")
            return df

        except FileNotFoundError:
            print(f"Error: File {input_file} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def sort_by_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort dataframe by timestamp in ascending order

        Parameters:
        df (pd.DataFrame): Input dataframe

        Returns:
        pd.DataFrame: Sorted dataframe
        """
        try:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Sort by timestamp in ascending order
            df_sorted = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

            print(f"Data sorted by timestamp in ascending order")
            print(f"Date range: {df_sorted['timestamp'].min()} to {df_sorted['timestamp'].max()}")
            print(f"Total records: {len(df_sorted)}")

            return df_sorted

        except Exception as e:
            print(f"Error sorting data: {e}")
            return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and add to dataframe

        Parameters:
        df (pd.DataFrame): Input dataframe with OHLCV data

        Returns:
        pd.DataFrame: Dataframe with technical indicators added
        """
        print("\n" + "=" * 50)
        print("CALCULATING TECHNICAL INDICATORS")
        print("=" * 50)

        df_result = df.copy()

        # Note: RSI6 and StochRSI removed as requested

        # 1. MACD
        print("Calculating MACD...")
        macd_data = TechnicalIndicators.macd(df['close'])
        df_result['MACD'] = macd_data['MACD']
        df_result['MACD_signal'] = macd_data['MACD_signal']
        df_result['MACD_histogram'] = macd_data['MACD_histogram']

        # 2. Moving Average 16
        print("Calculating Moving Average (16 periods)...")
        df_result['ma16'] = TechnicalIndicators.moving_average(df['close'], period=16)

        # 3. Rate of Change (ROC)
        print("Calculating Rate of Change...")
        df_result['ROC'] = TechnicalIndicators.rate_of_change(df['close'])

        # 4. Williams %R (Overbought/Oversold Index)
        print("Calculating Williams %R...")
        df_result['williams_r'] = TechnicalIndicators.williams_r(
            df['high'], df['low'], df['close']
        )

        # 5. Volume Weighted Moving Average (using volatility as volume proxy)
        print("Calculating VWMA (using volatility as volume proxy)...")
        df_result['VWMA'] = TechnicalIndicators.vwma(df['close'], df['v'])

        # 6. Linear Regression Moving Average
        print("Calculating Linear Regression Moving Average...")
        df_result['LRMA'] = TechnicalIndicators.linear_regression_ma(df['close'])

        print("All technical indicators calculated successfully!")
        print("Note: RSI6 and StochRSI excluded as requested")

        return df_result

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove zeros and noise from generated indicator data

        Parameters:
        df (pd.DataFrame): Dataframe with technical indicators

        Returns:
        pd.DataFrame: Cleaned dataframe with no empty or zero indicator values
        """
        print("\n" + "=" * 40)
        print("CLEANING INDICATOR DATA")
        print("=" * 40)

        df_cleaned = df.copy()

        # List of indicator columns (RSI6 and StochRSI removed)
        indicator_columns = [
            'MACD', 'MACD_signal', 'MACD_histogram',
            'ma16', 'ROC', 'williams_r', 'VWMA', 'LRMA'
        ]

        initial_rows = len(df_cleaned)

        # Replace infinite values with NaN
        df_cleaned[indicator_columns] = df_cleaned[indicator_columns].replace([np.inf, -np.inf], np.nan)

        # Replace zeros with NaN in indicators (zeros can cause problems in further analysis)
        print("Replacing zero values with NaN in indicator columns...")
        df_cleaned[indicator_columns] = df_cleaned[indicator_columns].replace(0, np.nan)

        # Remove rows where ANY indicator is NaN or zero (more aggressive cleaning)
        print("Removing rows with any empty or zero indicator values...")
        df_cleaned = df_cleaned.dropna(subset=indicator_columns, how='any')

        # Additional cleaning: Remove rows with extreme outliers that could cause issues
        print("Removing extreme outliers...")
        for col in indicator_columns:
            if col in df_cleaned.columns:
                # Calculate IQR and remove extreme outliers
                Q1 = df_cleaned[col].quantile(0.01)
                Q3 = df_cleaned[col].quantile(0.99)
                IQR = Q3 - Q1

                # Define outlier bounds (more conservative)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                # Remove outliers
                outlier_mask = (df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)
                df_cleaned = df_cleaned[outlier_mask]

        # Final validation: Ensure no problematic values remain
        print("Final validation of indicator values...")
        for col in indicator_columns:
            if col in df_cleaned.columns:
                # Check for any remaining problematic values
                problematic_mask = (
                        df_cleaned[col].isna() |
                        df_cleaned[col].isin([0, np.inf, -np.inf]) |
                        (df_cleaned[col] == 0)
                )

                if problematic_mask.any():
                    df_cleaned = df_cleaned[~problematic_mask]

        # For analysis, show rows removed
        rows_removed = initial_rows - len(df_cleaned)
        print(f"\nData cleaning summary:")
        print(f"Initial rows: {initial_rows}")
        print(f"Rows removed: {rows_removed}")
        print(f"Final rows: {len(df_cleaned)}")
        print(f"Data retention: {(len(df_cleaned) / initial_rows) * 100:.1f}%")

        # Show final data quality summary
        print(f"\nFinal data quality:")
        print(f"Dataset shape: {df_cleaned.shape}")

        for col in indicator_columns:
            if col in df_cleaned.columns:
                non_null_count = df_cleaned[col].notna().sum()
                null_count = df_cleaned[col].isna().sum()
                zero_count = (df_cleaned[col] == 0).sum()
                min_val = df_cleaned[col].min()
                max_val = df_cleaned[col].max()

                print(
                    f"{col}: {non_null_count} valid, {null_count} null, {zero_count} zeros, range=[{min_val:.4f}, {max_val:.4f}]")

        # Ensure we have enough data left
        if len(df_cleaned) < 50:
            print(f"\n⚠️  Warning: Only {len(df_cleaned)} rows remaining after cleaning.")
            print("This might not be sufficient for reliable analysis.")
        else:
            print(f"\n✅ Data cleaning completed successfully!")
            print(f"Clean dataset ready for further analysis with {len(df_cleaned)} valid records.")

        return df_cleaned

    def process_stock_data(self, input_file: str, output_file: str) -> bool:
        """
        Complete pipeline to process stock data with technical indicators

        Parameters:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file

        Returns:
        bool: Success status
        """
        print("STOCK DATA TECHNICAL INDICATORS PIPELINE")
        print("=" * 70)

        # Step 1: Load and validate data
        print("\nStep 1: Loading and validating data...")
        df = self.load_and_validate_data(input_file)
        if df is None:
            return False

        # Step 2: Sort by timestamp
        print("\nStep 2: Sorting data by timestamp...")
        df_sorted = self.sort_by_timestamp(df)

        # Step 3: Calculate technical indicators
        print("\nStep 3: Calculating technical indicators...")
        df_with_indicators = self.calculate_technical_indicators(df_sorted)

        # Step 4: Clean data
        print("\nStep 4: Cleaning indicator data...")
        df_cleaned = self.clean_data(df_with_indicators)

        # Step 5: Save output (only clean data with no empty/zero indicators)
        print("\nStep 5: Saving processed data (excluding rows with empty/zero indicators)...")
        try:
            # Final check before saving - ensure absolutely no problematic values
            final_indicator_columns = [
                'MACD', 'MACD_signal', 'MACD_histogram',
                'ma16', 'ROC', 'williams_r', 'VWMA', 'LRMA'
            ]

            # Double-check: Remove any remaining rows with empty or zero indicator values
            rows_before_final_check = len(df_cleaned)

            for col in final_indicator_columns:
                if col in df_cleaned.columns:
                    # Remove rows where indicator is NaN, zero, or infinite
                    df_cleaned = df_cleaned[
                        df_cleaned[col].notna() &
                        (df_cleaned[col] != 0) &
                        np.isfinite(df_cleaned[col])
                        ]

            rows_after_final_check = len(df_cleaned)

            if rows_before_final_check != rows_after_final_check:
                print(f"Final cleanup: Removed {rows_before_final_check - rows_after_final_check} additional rows")

            # Save the cleaned data
            df_cleaned.to_csv(output_file, index=False)
            print(f"Processed data saved to {output_file}")

            # Verify the saved data has no problematic values
            print(f"\nFinal verification of saved data:")
            for col in final_indicator_columns:
                if col in df_cleaned.columns:
                    has_na = df_cleaned[col].isna().sum()
                    has_zero = (df_cleaned[col] == 0).sum()
                    has_inf = (~np.isfinite(df_cleaned[col])).sum()

                    if has_na > 0 or has_zero > 0 or has_inf > 0:
                        print(f"⚠️  {col}: {has_na} NaN, {has_zero} zeros, {has_inf} infinite values")
                    else:
                        print(f"✅ {col}: Clean (no empty/zero/infinite values)")

            # Display summary
            print(f"\nPROCESSING SUMMARY:")
            print(f"Input file: {input_file}")
            print(f"Output file: {output_file}")
            print(f"Original rows: {len(df)}")
            print(f"Final rows: {len(df_cleaned)}")
            print(f"Columns added: {len(df_cleaned.columns) - len(df)}")
            print(f"Data retention: {(len(df_cleaned) / len(df)) * 100:.1f}%")

            # Show sample of final data
            if len(df_cleaned) > 0:
                print(f"\nSample of final clean data:")
                sample_cols = ['timestamp', 'close'] + final_indicator_columns[:4]  # Show subset for readability
                print(df_cleaned[sample_cols].head().to_string())

                # Show column names
                print(f"\nFinal columns:")
                print(list(df_cleaned.columns))
            else:
                print(f"\n⚠️  Warning: No valid data remains after cleaning!")

            return True

        except Exception as e:
            print(f"Error saving output file: {e}")
            return False


def create_sample_stock_data(filename: str, n_samples: int = 500):
    """
    Create sample stock data for testing

    Parameters:
    filename (str): Output filename
    n_samples (int): Number of samples to generate
    """
    print(f"Creating sample stock data: {filename}")

    np.random.seed(42)

    # Generate timestamps
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')

    # Generate realistic stock price data with trends and volatility changes
    base_price = 100
    prices = [base_price]
    volatilities = []

    for i in range(1, n_samples):
        # Add different market regimes
        if i < n_samples // 4:  # Uptrend with low volatility
            trend = 0.0008
            vol = 0.015
        elif i < n_samples // 2:  # Sideways with medium volatility
            trend = 0.0002
            vol = 0.025
        elif i < 3 * n_samples // 4:  # Downtrend with high volatility
            trend = -0.0006
            vol = 0.035
        else:  # Recovery with decreasing volatility
            trend = 0.0012
            vol = 0.020

        # Generate daily return
        daily_return = np.random.normal(trend, vol)
        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)

        # Store volatility (with some noise)
        volatilities.append(abs(vol + np.random.normal(0, 0.005)))

    # Generate OHLC data
    opens = [prices[0]] + prices[:-1]  # Open is previous close
    highs = []
    lows = []

    for i, close in enumerate(prices):
        # Generate realistic high and low
        daily_range = close * volatilities[i] * 0.5
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)

        # Ensure high >= open, close and low <= open, close
        high = max(high, opens[i], close)
        low = min(low, opens[i], close)

        highs.append(high)
        lows.append(low)

    # Create sample data
    sample_data = {
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'v': volatilities
    }

    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv(filename, index=False)
    print(f"Sample stock data created with {len(df_sample)} records")


def main():
    """
    Main function to run the technical indicators pipeline
    """
    print("Technical Indicators Pipeline for Stock Data Analysis")
    print("=" * 70)

    # File paths
    input_file = "icici_model_validation_input1.csv"  # Input CSV file
    output_file = "stock_data_with_indicators_model_validation.csv"  # Output CSV file

    # Create processor instance
    processor = StockDataProcessor()

    # Check if input file exists, create sample if not
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found. Creating sample data...")
        create_sample_stock_data(input_file)

    # Process the data
    success = processor.process_stock_data(input_file, output_file)

    if success:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"\nYour processed data with technical indicators is ready: {output_file}")
        print(f"\nTechnical indicators added:")
        print(f"• MACD (with signal and histogram)")
        print(f"• Moving Average (16 periods)")
        print(f"• Rate of Change (ROC)")
        print(f"• Williams %R (Overbought/Oversold)")
        print(f"• Volume Weighted MA (using volatility)")
        print(f"• Linear Regression Moving Average")
        print(f"\nNote: RSI6 and StochRSI excluded as requested")
        print(f"All rows with empty or zero indicator values removed")
        print(f"Data is now ready for further analysis or trading strategies!")
    else:
        print(f"\n❌ Pipeline failed. Please check your input data and try again.")


if __name__ == "__main__":
    main()