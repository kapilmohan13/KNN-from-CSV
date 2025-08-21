import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'v', 'MACD', 'MACD_signal',
    'MACD_histogram', 'ma16', 'ROC', 'williams_r', 'VWMA', 'LRMA'
]

def train_rnn(train_csv, model_path='rnn_volatility_model.keras', scaler_path='rnn_scaler.pkl', seq_len=5):
    df = pd.read_csv(train_csv)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Reduce weightage of 'v' in input features
    X_features = df[FEATURE_COLUMNS].copy()
    X_features['v'] = X_features['v'] * 0.4  # Scale down volatility

    # Prepare features and target for next time step prediction
    X = X_features.values
    y = df['v'].values

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences for RNN
    X_seq = []
    y_seq = []
    for i in range(seq_len, len(X_scaled)):
        X_seq.append(X_scaled[i-seq_len:i])
        y_seq.append(y[i])  # Predict current volatility

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Build RNN model
    model = Sequential([
        SimpleRNN(32, input_shape=(seq_len, len(FEATURE_COLUMNS)), activation='tanh'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_seq, y_seq, epochs=30, batch_size=32, verbose=1)

    # Save model and scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    # Calculate top 5% volatility threshold
    vol_threshold = np.percentile(df['v'], 95)
    print(f"Top 5% volatility threshold: {vol_threshold:.2f}")
    joblib.dump(vol_threshold, 'rnn_volatility_threshold.pkl')

    return model, scaler, vol_threshold, seq_len

def test_rnn(test_csv, model_path='rnn_volatility_model.keras', scaler_path='rnn_scaler.pkl', threshold_path='rnn_volatility_threshold.pkl', output_csv='rnn_volatility_predictions.csv', seq_len=5):
    df = pd.read_csv(test_csv)
    df = df.sort_values('timestamp').reset_index(drop=True)

    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    vol_threshold = joblib.load(threshold_path)

    X = df[FEATURE_COLUMNS].values
    X_scaled = scaler.transform(X)

    # Prepare sequences for prediction
    predicted_vol = [np.nan] * len(df)
    predicted_risk = ['low'] * len(df)
    for i in range(seq_len, len(X_scaled)):
        seq = X_scaled[i-seq_len:i].reshape(1, seq_len, len(FEATURE_COLUMNS))
        pred = model.predict(seq, verbose=0)[0][0]
        predicted_vol[i] = pred
        predicted_risk[i] = 'high' if pred >= vol_threshold else 'low'

    # Actual risk classification
    df['actual_volatility_risk'] = df['v'].apply(lambda x: 'high' if x >= vol_threshold else 'low')
    df['predicted_volatility'] = predicted_vol
    df['predicted_volatility_risk'] = predicted_risk

    df.to_csv(output_csv, index=False)
    print(f"Test predictions saved to: {output_csv}")

if __name__ == "__main__":
    # Train
    model, scaler, vol_threshold, seq_len = train_rnn('train.csv')
    # Test
    test_rnn('valtest.csv', seq_len=seq_len)