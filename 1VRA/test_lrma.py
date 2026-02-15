
import pandas as pd
import numpy as np
from backend.indicators import TechnicalIndicators

# Create dummy data
# Use more realistic prices to avoid log(0)
np.random.seed(42)
data = pd.Series(100 + np.random.randn(100).cumsum())

print("--- Testing Technical Indicators ---")

# Test LRMA
try:
    print("\nTesting LRMA...")
    lrma = TechnicalIndicators.linear_regression_ma(data, period=14)
    print(f"LRMA Sample: {lrma.iloc[20]:.4f}")
    assert not lrma.iloc[20:].isna().any(), "LRMA has unexpected NaNs"
    print("LRMA Passed.")
except Exception as e:
    print(f"LRMA Failed: {e}")

# Test Volatility
try:
    print("\nTesting Volatility...")
    vol = TechnicalIndicators.volatility(data, period=21)
    print(f"Volatility Sample: {vol.iloc[-1]:.6f}")
    assert not vol.iloc[21:].isna().any(), "Volatility has unexpected NaNs"
    
    # Check if calculation makes sense
    log_ret = np.log(data / data.shift(1))
    expected = log_ret.rolling(21).std()
    pd.testing.assert_series_equal(vol, expected, obj="Volatility mismatch")
    print("Volatility Passed.")
except Exception as e:
    print(f"Volatility Failed: {e}")
