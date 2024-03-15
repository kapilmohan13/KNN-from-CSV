import yfinance as yahooFinance
import pandas as pd
import datetime

# Here We are getting Facebook financial information
# We need to pass FB as argument for that
GetFacebookInformation = yahooFinance.Ticker("META")

# whole python dictionary is printed here
print(GetFacebookInformation.info)
# display Company Sector
print("Company Sector : ", GetFacebookInformation.info['sector'])

# display Price Earnings Ratio
print("Price Earnings Ratio : ", GetFacebookInformation.info['trailingPE'])

# display Company Beta
print(" Company Beta : ", GetFacebookInformation.info['beta'])

# get all key value pairs that are available
for key, value in GetFacebookInformation.info.items():
    print(key, ":", value)

# Let us  get historical stock prices for Facebook
# covering the past few years.
# max->maximum number of daily prices available
# for Facebook.
# Valid options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y,
# 5y, 10y and ytd.
print(GetFacebookInformation.history(period="max"))

pd.set_option('display.max_rows', None)
# Let us  get historical stock prices for Facebook
# covering the past few years.
# max->maximum number of daily prices available
# for Facebook.
# Valid options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y,
# 5y, 10y and ytd.
print(GetFacebookInformation.history(period="max"))

##################################
# startDate , as per our convenience we can modify
startDate = datetime.datetime(2019, 5, 31)

# endDate , as per our convenience we can modify
endDate = datetime.datetime(2021, 1, 30)
GetFacebookInformation = yahooFinance.Ticker("META")

# pass the parameters as the taken dates for start and end
print(GetFacebookInformation.history(start=startDate,
                                     end=endDate))