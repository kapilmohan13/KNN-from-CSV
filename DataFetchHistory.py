# %%
import csv
import datetime
import math

import dateutil.utils

import globalAttribs as ga
import credential as cr
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s -  %(funcName)25s()] %(message)s", datefmt='%m/%d/%Y-%I:%M:%S %p')
logger.setLevel(logging.DEBUG)

history_data_dict = {}
# Define the symbol and resolution HDFCBANK.NS  NSE:NIFTY50-INDEX
symbol = "NSE:HDFCBANK-EQ"
resolution = "1"
date_format = "1"
cont_flag = "1"
histdata = pd.DataFrame()
#  %%
def main():
    # Initialize Fyers API client
    ga.init(False)
    print(ga.fyers.positions())



    start_date = datetime.date(2018, 1, 1)
    end_Date = datetime.date(2018, 4, 10)
    iteration = math.ceil(((datetime.date.today() - start_date).days) / 100)
    getHistoryData(start_date,end_Date,5)
    print(histdata)

    for x in range(iteration):
        getHistoryData(start_date,end_Date,15)
        start_date = start_date + datetime.timedelta(days=100)
        end_Date = end_Date + datetime.timedelta(days=100)




    # all_data = pd.DataFrame()
    # history_data_dict = {}
    # Define the symbol and resolution
    # symbol = "NSE:NIFTY50-INDEX"
    # resolution = "5"
    # date_format = "1"
    # cont_flag = "1"
    # # Loop over the last  1 years
    # for year in range(1):
    #     # range_from = (datetime.date.today() - datetime.timedelta(days=(year + 1) * 365)).strftime('%Y-%m-%d')
    #     # range_to = (datetime.date.today() - datetime.timedelta(days=year * 365)).strftime('%Y-%m-%d')
    #     range_from = (datetime.date.today() - datetime.timedelta(days=(year + 1) * 99)).strftime('%Y-%m-%d')
    #     range_to = (datetime.date.today() - datetime.timedelta(days=year)).strftime('%Y-%m-%d')
    #     data = {
    #         "symbol": symbol,
    #         "resolution": resolution,
    #         "date_format": date_format,
    #         "range_from": range_from,
    #         "range_to": range_to,
    #         "cont_flag": cont_flag
    #     }
    #
    #
    #     response = ga.fyers.history(data=data)
    #     # print(response)
    #     if not history_data_dict:
    #         print("The dictionary is empty")
    #         history_data_dict = response
    #     else:
    #         print("The dictionary is not empty")
    #         history_data_dict.get("candles").extend(response.get("candles") )
    #
    # Open a CSV file in write mode
    histdata.to_csv('..//data//new.csv')
    # with open('..//data//new.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #
    #         # Write each list to the CSV file
    #     for row in history_data_dict.get("candles"):
    #         transformed_row = transform(row)
    #         writer.writerow(transformed_row)

    print("CSV file has been created successfully.")

    # # Load the JSON data
    # # with open('data.json', 'r') as json_file:
    # data = pd.read_json(response)
    #
    # # Convert the DataFrame to a CSV file
    # data.to_csv('data.csv', index=False)


def getHistoryData(startDate, endDate, res):
    data = {
        "symbol": symbol,
        "resolution": res,
        "date_format": date_format,
        "range_from": startDate,
        "range_to": endDate,
        "cont_flag": cont_flag
    }
    global histdata
    response = ga.fyers.history(data=data)
    print(response)
    data = pd.DataFrame.from_dict(response['candles'])
    cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    data.columns = cols
    # df['date'] = pd.to_datetime(df['date'], unit='s')
    data['datetime'] = pd.to_datetime(data['datetime'], unit='s')
    # data['datetime'] = data['datetime'].dt.tz_localize('utc').dt.tz_convert('Asia/Kolkata')
    # data['datetime'] = data['datetime'].dt.tz_localize(None)
    # data.data.setindex('datetime')
    # datetime.datetime.fromtimestamp(0)
    histdata = pd.concat([histdata,data], axis=0)




# Function to transform the "Age" field
def transform(row):
    if row[0] != "Name":  # Skip the header
        dt_object = datetime.datetime.utcfromtimestamp(row[0])

        # Format datetime object to 'yyyy-mm-dd'
        row[0] = dt_object.strftime('%Y-%m-%d')

        # row[0] =  1  # Add 1 to the age
    return row

# %%
if __name__ =="__main__":
    main()

# %%
