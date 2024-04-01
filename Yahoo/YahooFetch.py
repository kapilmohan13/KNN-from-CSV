# Load the required modules and packages
import numpy as np
import pandas as pd
import yfinance as yf
import Mystock_info as si

ticker = "NVDA"
# Pull NIFTY data from Yahoo finance
NIFTY = yf.download(ticker, '2024-01-01', '2024-03-31')

# Compute the logarithmic returns using the closing price
returns = np.log(NIFTY['Close'] / NIFTY['Close'].shift(1))
volatility = returns.std() * np.sqrt(252)
sharpe_ratio = ((returns.mean()*252) - 0.05) / volatility
print("sharpe_ratio:", end=" ")
print(sharpe_ratio)


# Pull EPS and calclate P/E ratio
stock_info = yf.Ticker(ticker).info
price = stock_info["currentPrice"]
EPS = stock_info["trailingEps"]
PE = round(price / EPS, 2)
print("PE ratio", end=" ")
print(PE)

# Pull EPS and calclate EPS ratio
# from yahoo_fin.stock_info import get_analysts_info
# info = get_analysts_info(ticker)
# print("Stock info:", end=" ")
# print(info)

from yahoo_fin.stock_info import get_stats_valuation
info = get_stats_valuation(ticker)
print("stats valuation info:", end=" ")
print(info)
toDict = pd.Series(info.loc[:, 1].values, index=info.loc[:, 0]).to_dict()
PS_ratio = toDict.get("Price/Sales (ttm)")
PB_ratio = toDict.get("Price/Book (mrq)")
print("PS_ratio:", end=" ")
print(PS_ratio)
print("PB_ratio:", end=" ")
print(PB_ratio)


info = si.get_stats(ticker)
print("stats info:", end=" ")
print(info)
toDict = pd.Series(info.Value.values,index=info.Attribute).to_dict()
EPS_ratio = toDict.get("Diluted EPS (ttm)")
print("EPS_ratio:", end=" ")
print(EPS_ratio)

# Get ticker from quote
quote = si.get_quote_table(ticker)
print("stats info:", end=" ")
print(quote)
