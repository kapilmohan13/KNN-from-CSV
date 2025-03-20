# %%
import csv
import datetime

import globalAttribs as ga
import credential as cr
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(asctime)s -  %(funcName)25s()] %(message)s", datefmt='%m/%d/%Y-%I:%M:%S %p')
logger.setLevel(logging.DEBUG)

#  %%
def main():
    # Initialize Fyers API client
    ga.init(False)
    print(ga.fyers.positions())

    all_data = pd.DataFrame()
    history_data_dict = {}
    # Define the symbol and resolution
    symbol = "NSE:NIFTY24D0523700PE" #NIFTY24D0523700PE"  #"NSE:NIFTY50-INDEX"
    filename = '..//data//options.csv'
    resolution = "5S"
    date_format = "1"
    cont_flag = "1"
    # Loop over the last  1 years
    for year in range(1):
        range_from = (datetime.date.today() - datetime.timedelta(days=(year + 1) * 10)).strftime('%Y-%m-%d')
        range_to = (datetime.date.today() - datetime.timedelta(days=year * 10)).strftime('%Y-%m-%d')
        # range_from = (datetime.date.today() - datetime.timedelta(days=(year + 1) * 99)).strftime('%Y-%m-%d')
        # range_to = (datetime.date.today() - datetime.timedelta(days=year)).strftime('%Y-%m-%d')
        print("range_from:", range_from, "  range_to:", range_to)
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": date_format,
            "range_from": range_from,
            "range_to": range_to,
            "cont_flag": cont_flag
        }


        response = ga.fyers.history(data=data)
        # print(response)
        if not history_data_dict:
            print("The dictionary is empty")
            history_data_dict = response
        else:
            print("The dictionary is not empty")
            history_data_dict.get("candles").extend(response.get("candles") )

    # Open a CSV file in write mode
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

            # Write each list to the CSV file
        for row in history_data_dict.get("candles"):
            transformed_row = transform(row)
            writer.writerow(transformed_row)

    print("CSV file has been created successfully.")

    # # Load the JSON data
    # # with open('data.json', 'r') as json_file:
    # data = pd.read_json(response)
    #
    # # Convert the DataFrame to a CSV file
    # data.to_csv('data.csv', index=False)

# Function to transform the "Age" field
def transform(row):
    if row[0] != "Name":  # Skip the header
        dt_object = datetime.datetime.utcfromtimestamp(row[0])
        # datetime.datetime.fromtimestamp(row[0], tz=datetime.timezone.)

        # Format datetime object to 'yyyy-mm-dd'
        row[0] = dt_object.strftime('%Y-%m-%d: %H:%M:%S')

        # row[0] =  1  # Add 1 to the age
    return row

# %%
if __name__ =="__main__":
    main()

# %%
