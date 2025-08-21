import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
merged_file = 'merged_DBSCAN_RF.csv'
output_txt = 'rf_DBSCAN_prediction.txt'

# --- LOAD MERGED DATA ---
df = pd.read_csv(merged_file)

# --- HARD VOTING STRATEGY ---
def hard_vote(row):
    if row['predicted_volatility_risk'] == row['risk']:
        return row['predicted_volatility_risk']
    else:
        return 'low'

df['combined_prediction'] = df.apply(hard_vote, axis=1)

# --- METRICS CALCULATION ---
y_true = df['actual_volatility_risk']
y_pred = df['combined_prediction']

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

print(f"Metrics written to: {output_txt}")