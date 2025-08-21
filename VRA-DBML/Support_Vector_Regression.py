import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'v', 'MACD', 'MACD_signal',
    'MACD_histogram', 'ma16', 'ROC', 'williams_r', 'VWMA', 'LRMA'
]

def train_svm(train_csv, model_path='svm_volatility_model.pkl', scaler_path='svm_scaler.pkl'):
    df = pd.read_csv(train_csv)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Reduce weightage of 'v' in input features
    X_features = df[FEATURE_COLUMNS].copy()
    X_features['v'] = X_features['v'] * 1  # Scale down volatility

    # Prepare features and target for next time step prediction
    X = X_features.values[:-1]  # All but last row
    y = df['v'].values[1:]      # All but first row (next time step)

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = SVR(kernel='rbf')
    model.fit(X_scaled, y)

    # Save model and scaler
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Calculate top 5% volatility threshold
    vol_threshold = np.percentile(df['v'], 95)
    print(f"Top 5% volatility threshold: {vol_threshold:.2f}")

    # Save threshold for later use
    joblib.dump(vol_threshold, 'svm_volatility_threshold.pkl')

    return model, scaler, vol_threshold

def test_svm(test_csv, model_path='svm_volatility_model.pkl', scaler_path='svm_scaler.pkl', threshold_path='svm_volatility_threshold.pkl', output_csv='svm_volatility_predictions.csv'):
    df = pd.read_csv(test_csv)
    df = df.sort_values('timestamp').reset_index(drop=True)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    vol_threshold = joblib.load(threshold_path)

    # Prepare features for prediction (all but last row)
    X = df[FEATURE_COLUMNS].values[:-1]
    X_scaled = scaler.transform(X)
    predicted_vol = model.predict(X_scaled)

    # Align predictions to next time step
    df['predicted_volatility'] = np.nan
    df['predicted_volatility_risk'] = 'low'
    for i in range(1, len(df)):
        df.loc[i, 'predicted_volatility'] = predicted_vol[i-1]
        df.loc[i, 'predicted_volatility_risk'] = 'high' if predicted_vol[i-1] >= vol_threshold else 'low'

    # Actual risk classification
    df['actual_volatility_risk'] = df['v'].apply(lambda x: 'high' if x >= vol_threshold else 'low')

    # Save output
    df.to_csv(output_csv, index=False)
    print(f"Test predictions saved to: {output_csv}")

if __name__ == "__main__":
    # Train
    train_svm('train.csv')
    # Test
    test_svm('valtest.csv')