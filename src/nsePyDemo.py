import nsepy as ns
import pandas as pd
from datetime import date

# Fetch historical data for NIFTY 50 options
start_date = date(2023,1,1)
end_date = date(2023,1,2)
options_data = ns.get_history(symbol='SBIN', start=start_date, end=end_date, futures=False)

# Convert to DataFrame
options_df = pd.DataFrame(options_data)

# Display the first few rows
print(options_df.head())
