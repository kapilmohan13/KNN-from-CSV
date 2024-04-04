# Load the required modules and packages
import numpy as np
import pandas as pd
import yfinance as yf

import GaussianNB
import Mystock_info as si
import MyYesg

ticker = "GOOG"
# Pull NIFTY data from Yahoo finance
NIFTY = yf.download(ticker, '2024-01-01', '2024-03-31')
stock_financials = {}

# Compute the logarithmic returns using the closing price
returns = np.log(NIFTY['Close'] / NIFTY['Close'].shift(1))
volatility = returns.std() * np.sqrt(252)
sharpe_ratio = ((returns.mean()*252) - 0.05) / volatility
print("sharpe_ratio:", end=" ")
print(sharpe_ratio)
stock_financials.update({"sharpe_ratio": sharpe_ratio})


# Pull EPS and calclate P/E ratio
stock_info = yf.Ticker(ticker).info
price = stock_info["currentPrice"]
EPS = stock_info["trailingEps"]
PE_ratio = round(price / EPS, 2)
print("PE_ratio", end=" ")
print(PE_ratio)

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

stock_financials.update({"PE_ratio": PE_ratio})
stock_financials.update({"EPS_ratio": float(EPS_ratio)})
stock_financials.update({"PS_ratio": float(PS_ratio)})
stock_financials.update({"PB_ratio":  float(PB_ratio)})


NetMargin_ratio = float(toDict.get("Profit Margin").rstrip("%")) / 100
print("NetMargin_ratio:", end=" ")
print(NetMargin_ratio)

current_ratio = float(toDict.get("Current Ratio (mrq)"))
print("current_ratio:", end=" ")
print(current_ratio)

roa_ratio = float(toDict.get("Return on Assets (ttm)").rstrip("%")) / 100
print("roa_ratio:", end=" ")
print(roa_ratio)

roe_ratio = float(toDict.get("Return on Equity (ttm)").rstrip("%")) / 100
print("roe_ratio:", end=" ")
print(roe_ratio)


stock_financials.update({"NetMargin_ratio": NetMargin_ratio})
stock_financials.update({"current_ratio": current_ratio})
stock_financials.update({"roa_ratio": roa_ratio})
stock_financials.update({"roe_ratio": roe_ratio})


# Get ticker from quote
# quote = si.get_quote_table(ticker)
# print("stats info:", end=" ")
# print(quote)

try :
    df = MyYesg.get_historic_esg(ticker)

    last_row = df.iloc[-1:]
    esg_dict = {"ESG_score": last_row.iloc[0, 0]}
    print(esg_dict)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print(f"Exception type: {type(e)}")

# stock_financials.update(esg_dict)
print(stock_financials)

stock_financialsList = list(stock_financials.values())
print(stock_financialsList)
predictioninput = [stock_financialsList]
y_pred = GaussianNB.predictorGaussianNB(predictioninput)
print(y_pred)