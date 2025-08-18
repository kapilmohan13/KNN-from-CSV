import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
lstm_file = 'lstm_volatility_predictions.csv'
dbscan_file = 'risk_assessment_results.csv'  # Now using CSV
output_csv = 'merged_DBSCAN_LSTM.csv'
output_txt = 'LSTM_output_ML.txt'

# --- LOAD DATA ---
lstm_df = pd.read_csv(lstm_file)
dbscan_df = pd.read_csv(dbscan_file)  # Changed to read_csv

# --- MERGE ---
# Ensure both files have the same order or a common key (e.g., timestamp)
if 'timestamp' in lstm_df.columns and 'timestamp' in dbscan_df.columns:
    merged_df = pd.merge(lstm_df, dbscan_df[['timestamp', 'cluster', 'risk']], on='timestamp', how='left')
else:
    # If no timestamp, just concatenate columns (assuming same order)
    merged_df = lstm_df.copy()
    merged_df['cluster'] = dbscan_df['cluster']
    merged_df['risk'] = dbscan_df['risk']

# --- SAVE MERGED CSV ---
merged_df.to_csv(output_csv, index=False)

# --- METRICS CALCULATION ---
y_true = merged_df['actual_volatility_risk']
y_pred = merged_df['predicted_volatility_risk']

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label='high', zero_division=0)
recall = recall_score(y_true, y_pred, pos_label='high', zero_division=0)
f1 = f1_score(y_true, y_pred, pos_label='high', zero_division=0)

# --- WRITE METRICS TO TXT ---
with open(output_txt, 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")

print(f"Merged file saved as: {output_csv}")