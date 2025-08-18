import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
lr_file = 'lr_volatility_predictions.csv'
dbscan_file = 'risk_assessment_results.csv'
output_csv = 'merged_DBSCAN_LR.csv'
output_txt = 'LR_metrics_ML.txt'

# --- LOAD DATA ---
lr_df = pd.read_csv(lr_file)
dbscan_df = pd.read_csv(dbscan_file)

# --- MERGE ---
if 'timestamp' in lr_df.columns and 'timestamp' in dbscan_df.columns:
    merged_df = pd.merge(lr_df, dbscan_df[['timestamp', 'cluster', 'risk']], on='timestamp', how='left')
else:
    merged_df = lr_df.copy()
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
print(f"Metrics written to: {output_txt}")