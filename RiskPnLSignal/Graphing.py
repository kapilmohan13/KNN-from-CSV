import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np


df = pd.read_csv("C:\source\KNN-from-CSV\data\GOLDBEES_2025.csv")
# df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')
# data = df[['time', 'intc']]

data = df['v']

import pandas as pd

# Example DataFrame
# data = {'column_name': [10, 15, 20, 25, 40]}
# df = pd.DataFrame(data)

# Calculate the average jump
# average_jump = df['v'].diff().abs().mean()  # Use .abs() if only positive jumps matter
# print(f"The average jump is: {average_jump}")

diff = df['v'].diff().abs()
df['diff'] = diff
# print(df)

selected_rows_loc = df.loc[df['time'] == '17-02-2025 15:24:00']
print(selected_rows_loc)
