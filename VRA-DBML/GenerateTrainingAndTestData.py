import pandas as pd

# --- CONFIGURATION ---
input_file = 'stock_data_with_indicators1.csv'
timestamp_column = 'timestamp'
train_file = 'train.csv'
valtest_file = 'valtest.csv'

# --- READ AND SORT ---
df = pd.read_csv(input_file)
df[timestamp_column] = pd.to_datetime(df[timestamp_column], dayfirst=True)  # Ensure correct parsing
df_sorted = df.sort_values(by=timestamp_column, ascending=True)

# --- SPLIT ---
total_rows = len(df_sorted)
split_index = int(total_rows * 0.90)

train_df = df_sorted.iloc[:split_index]
valtest_df = df_sorted.iloc[split_index:]

# --- OUTPUT ---
train_df.to_csv(train_file, index=False)
valtest_df.to_csv(valtest_file, index=False)

print(f"Training rows: {len(train_df)}")
print(f"Validation/Test rows: {len(valtest_df)}")