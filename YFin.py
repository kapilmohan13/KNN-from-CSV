import csv

import yfinance as yahooFinance
import yesg
import MyYesg
import pandas as pd
import datetime



# READ  TICKERS TO SCRAP FROM YAHOO FINANCE
sp_500 = pd.read_csv("data//yahoo.csv")
newList = []  # this will contain the out fields with ESG rating post iteration on tickers

for symbol in sp_500["Symbol"]:
    # Here We are getting Facebook financial information
    # We need to pass FB as argument for that
    GetFacebookInformation = yahooFinance.Ticker(symbol)
    obj = GetFacebookInformation.info
    # newTickerdf = pd.DataFrame(obj)
    # print(newTickerdf)

    df = yesg.get_historic_esg(symbol)
    if df is None:
        continue
    last_row = df.iloc[-1:]
    esg_dict = {"ESG_score": last_row.iloc[0, 0]}
    print(esg_dict)
    obj.update(esg_dict)

    newObjDict = {"trailingEps": obj.get("trailingEps"), "forwardEps": obj.get("forwardEps"),
                  "ESG_score": obj.get("ESG_score")}


    print(newObjDict)

    newList.append(newObjDict)

# Write to file
header_info = ["trailingEps", "forwardEps", "ESG_score"]
with open('data//ESG_yfin.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header_info)
    writer.writeheader()
    writer.writerows(newList)

