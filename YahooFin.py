# import yfinance, pandas and os
import requests
import yfinance as yf
import pandas as pd
import os


##################
# def _get_crumbs_and_cookies(stock):
#     """
#     get crumb and cookies for historical data csv download from yahoo finance
#
#     parameters: stock - short-handle identifier of the company
#
#     returns a tuple of header, crumb and cookie
#     """
#
#     url = "https://query2.finance.yahoo.com/v1/finance/esgChart"
#     with requests.session():
#         header = {'Connection': 'keep-alive',
#                   'Expires': '-1',
#                   'Upgrade-Insecure-Requests': '1',
#                   'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
#                    AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
#                   }
#         payload = {"symbol": "AAPL"}
#         website = requests.get(url, params=payload, headers=header)
#         if website.ok:
#             print("OOOKKKKK1111")
#         # soup = BeautifulSoup(website.text, 'lxml')
#         # crumb = re.findall('"CrumbStore":{"crumb":"(.+?)"}', str(soup))
#         #
#         # return (header, crumb[0], website.cookies)


##################


# url = "https://query2.finance.yahoo.com/v1/finance/esgChart"
# with requests.session():
#     header = {'Connection': 'keep-alive',
#               'Expires': '-1',
#               'Upgrade-Insecure-Requests': '1',
#               'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
#                AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
#               }
#     payload = {"symbol": "AAPL"}
#     website = requests.get(url, params=payload, headers=header)
#     if website.ok:
#         print("OOOKKKKK1111")

# Read in your symbols
sp_500 = pd.read_csv("data//NSI.csv")
url = "https://query2.finance.yahoo.com/v1/finance/esgChart"
# # url2 = "https://query2.finance.yahoo.com/v1/finance/esgChart?symbol=AAPL"
# dataframes = []
# # payload = {"symbol": "AAPL"}
# response = requests.get(url, params=payload)
# print(response.json)


# if response.ok:
#     print("OOOKKKKK")
#     df = pd.DataFrame(response.json()["esgChart"]["result"][0]["symbolSeries"])
#     df["symbol"] = "AAPL"
#     dataframes.append(df)


# df = pd.concat(dataframes)
# df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
# df.head()
# List of dataframes
dataframes = []

for symbol in sp_500["ticker_code"]:
    try:
        with requests.session():
            header = {'Connection': 'keep-alive',
                      'Expires': '-1',
                      'Upgrade-Insecure-Requests': '1',
                      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                       AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'
                      }
        response = requests.get(url, params={"symbol": symbol}, headers=header)
        if response.ok:
            print(symbol)
            df = pd.DataFrame(response.json()["esgChart"]["result"][0]["symbolSeries"])
            df["symbol"] = symbol
            dataframes.append(df)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Exception type: {type(e)}")
#

df = pd.concat(dataframes)
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
print(df.head())

# msft = "MSFT"
# msft_y = yf.Ticker(msft)
# esg_data = pd.DataFrame.transpose(msft_y.sustainability)
# esg_data['company_ticker'] = str(msft_y.ticker)

# print(esg_data)

# Import list of tickers from file
# os.chdir("C:\...")
# sp_500 = pd.read_csv('SP_500_tickers.csv')
# # Retrieve Yahoo! Finance Sustainability Scores for each ticker
# for i in sp_500['ticker_code']:
#     # print(i)
#     i_y = yf.Ticker(i)
#     try:
#         if i_y.sustainability is not None:
#             temp = pd.DataFrame.transpose(i_y.sustainability)
#             temp['company_ticker'] = str(i_y.ticker)
#             # print(temp)
#             esg_data = esg_data.append(temp)
#     except IndexError:
#         pass
