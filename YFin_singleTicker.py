import csv

import yfinance as yahooFinance
import yesg
import MyYesg
import pandas as pd
import datetime
from time import sleep



# READ  TICKERS TO SCRAP FROM YAHOO FINANCE
sp_500 = pd.read_csv("Tickers//temp.csv")
newList = []  # this will contain the out fields with ESG rating post iteration on tickers
loop = 1
for symbol in sp_500["SYMBOL"]:
    if loop % 250 == 0:
        #sleep to slow down
        print("SLEEPING . .")
        sleep(605)
    loop = loop + 1
    try:
        # Here We are getting Facebook financial information
        # We need to pass FB as argument for that
        print("Processing symbol: "+symbol)
        GetFacebookInformation = yahooFinance.Ticker(symbol)
        obj = GetFacebookInformation.info
        # newTickerdf = pd.DataFrame(obj)
        # print(newTickerdf)
        # print(obj.get("esgScores"))
        df = MyYesg.get_historic_esg(symbol)
        if df is None:
            continue
        last_row = df.iloc[-1:]
        esg_dict = {"ESG_score": last_row.iloc[0, 0]}
        print(esg_dict)
        obj.update(esg_dict)

        newObjDict = {"trailingEps": obj.get("trailingEps"), "forwardEps": obj.get("forwardEps"),
                      "heldPercentInsiders": obj.get("heldPercentInsiders"),
                      "heldPercentInstitutions": obj.get("heldPercentInstitutions"),
                      "profitMargins": obj.get("profitMargins"),
                      "priceToSalesTrailing12Months": obj.get("priceToSalesTrailing12Months"),
                      "payOutRatio": obj.get("payOutRatio"),
                      "fiveYearAvgDividendYield": obj.get("fiveYearAvgDividendYield"), "beta": obj.get("beta"),
                      "overallRisk": obj.get("overallRisk"), "boardRisk": obj.get("boardRisk"),
                      "ESG_score": obj.get("ESG_score")}
        #heldPercentInsiders, heldPercentInstitutions, profitMargins, priceToSalesTrailing12Months, payoutRatio,
        #fiveYearAvgDividendYield, beta, overallRisk, boardRisk

        print(newObjDict)

        newList.append(newObjDict)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Exception type: {type(e)}")

# Write to file
header_info = ["trailingEps", "forwardEps", "heldPercentInsiders", "heldPercentInstitutions", "profitMargins",
               "priceToSalesTrailing12Months", "payOutRatio", "fiveYearAvgDividendYield", "beta", "overallRisk", "boardRisk", "ESG_score"]
with open('data//SingleTicker.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header_info)
    writer.writeheader()
    writer.writerows(newList)

