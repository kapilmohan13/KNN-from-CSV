import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
from typing import Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Please install: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False


class LSTMVolatilityPredictor:
    """
    LSTM-based volatility prediction system with risk assessment.
    Predicts future volatility and classifies risk based on top 5% threshold.
    """

    def __init__(self, sequence_length: int = 10, lstm_units: int = 50, epochs: int = 100):
        """
        Initialize the LSTM volatility predictor.

        Args:
            sequence_length: Number of time steps to look back for prediction
            lstm_units: Number of LSTM units in the model
            epochs: Number of training epochs
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM functionality")

        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.epochs = epochs

        # Feature columns (excluding timestamp)
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'v', 'MACD', 'MACD_signal',
            'MACD_histogram', 'ma16', 'ROC', 'williams_r', 'VWMA', 'LRMA'
        ]

        # Model components
        self.model = None
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.volatility_threshold = None  # Top 5% threshold
        self.is_fitted = False

    def calculate_volatility_threshold(self, df: pd.DataFrame) -> float:
        """
        Calculate the 95th percentile threshold for volatility classification.

        Args:
            df: DataFrame containing 'v' column

        Returns:
            95th percentile threshold value
        """
        if 'v' not in df.columns:
            raise ValueError("Column 'v' (volatility) not found in DataFrame")

        threshold = df['v'].quantile(0.95)

        # Statistics
        v_min, v_max = df['v'].min(), df['v'].max()
        high_vol_count = (df['v'] >= threshold).sum()
        total_count = len(df)
        high_vol_percentage = (high_vol_count / total_count) * 100

        print(f"Volatility Threshold Analysis:")
        print(f"  Range: {v_min:.4f} to {v_max:.4f}")
        print(f"  95th Percentile Threshold: {threshold:.4f}")
        print(f"  High volatility records: {high_vol_count} ({high_vol_percentage:.2f}%)")
        print(f"  Low volatility records: {total_count - high_vol_count} ({100 - high_vol_percentage:.2f}%)")

        return threshold

    def classify_volatility_risk(self, volatility_values: np.ndarray) -> np.ndarray:
        """
        Classify volatility as high/low risk based on saved threshold.

        Args:
            volatility_values: Array of volatility values

        Returns:
            Array of risk classifications ('high' or 'low')
        """
        if self.volatility_threshold is None:
            raise ValueError("Volatility threshold not set. Train the model first.")

        return np.where(volatility_values >= self.volatility_threshold, 'high', 'low')

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.

        Args:
            df: Input DataFrame with time series data

        Returns:
            Tuple of (X_sequences, y_targets) for LSTM training
        """
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # Extract features and target
        features = df[self.feature_columns].values
        target = df['v'].values.reshape(-1, 1)

        # Handle missing values
        if np.isnan(features).any() or np.isnan(target).any():
            print("Warning: Found NaN values. Forward filling...")
            df_clean = df[self.feature_columns + ['v']].fillna(method='ffill').fillna(method='bfill')
            features = df_clean[self.feature_columns].values
            target = df_clean['v'].values.reshape(-1, 1)

        # Scale features and target
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target)

        # Create sequences
        X_sequences = []
        y_targets = []

        for i in range(self.sequence_length, len(features_scaled)):
            # Use past sequence_length time steps as input
            X_sequences.append(features_scaled[i - self.sequence_length:i])
            # Predict next time step volatility
            y_targets.append(target_scaled[i, 0])

        X_sequences = np.array(X_sequences)
        y_targets = np.array(y_targets)

        print(f"Sequence preparation completed:")
        print(f"  Input sequences shape: {X_sequences.shape}")
        print(f"  Target shape: {y_targets.shape}")
        print(f"  Features per timestep: {X_sequences.shape[2]}")

        return X_sequences, y_targets

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture.

        Args:
            input_shape: Shape of input sequences (timesteps, features)

        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='linear')  # Regression output
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, df: pd.DataFrame, validation_split: float = 0.2) -> 'LSTMVolatilityPredictor':
        """
        Train the LSTM model on time series data.

        Args:
            df: Training DataFrame
            validation_split: Fraction of data to use for validation

        Returns:
            Self for method chaining
        """
        print("=== Training LSTM Volatility Predictor ===")

        # Calculate volatility threshold
        print("\n1. Calculating volatility threshold...")
        self.volatility_threshold = self.calculate_volatility_threshold(df)

        # Prepare sequences
        print("\n2. Preparing LSTM sequences...")
        X_train, y_train = self.prepare_sequences(df)

        if len(X_train) == 0:
            raise ValueError("Not enough data to create sequences. Increase dataset size or reduce sequence_length.")

        # Build model
        print("\n3. Building LSTM model...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_lstm_model(input_shape)

        print("Model architecture:")
        self.model.summary()

        # Set up callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Train model
        print(f"\n4. Training LSTM model for {self.epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=32,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )

        # Training summary
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        print(f"\nTraining completed:")
        print(f"  Final training loss: {final_loss:.6f}")
        print(f"  Final validation loss: {final_val_loss:.6f}")
        print(f"  Epochs trained: {len(history.history['loss'])}")

        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict volatility for new data.

        Args:
            df: DataFrame with time series data

        Returns:
            Array of predicted volatility values
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # Prepare features
        features = df[self.feature_columns].values

        # Handle missing values
        if np.isnan(features).any():
            print("Warning: Found NaN values in prediction data. Forward filling...")
            df_clean = df[self.feature_columns].fillna(method='ffill').fillna(method='bfill')
            features = df_clean[self.feature_columns].values

        # Scale features
        features_scaled = self.feature_scaler.transform(features)

        # Create sequences for prediction
        predictions = []

        # For the first few records where we don't have enough history
        for i in range(min(self.sequence_length, len(features_scaled))):
            predictions.append(np.nan)  # Can't predict without enough history

        # Predict for records with sufficient history
        for i in range(self.sequence_length, len(features_scaled)):
            sequence = features_scaled[i - self.sequence_length:i].reshape(1, self.sequence_length, -1)
            pred_scaled = self.model.predict(sequence, verbose=0)
            pred_actual = self.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            predictions.append(pred_actual)

        return np.array(predictions)

    def save_model(self, model_dir: str = "lstm_model"):
        """
        Save the trained LSTM model and components.

        Args:
            model_dir: Directory to save model files
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")

        os.makedirs(model_dir, exist_ok=True)

        # Save LSTM model
        model_path = os.path.join(model_dir, "lstm_volatility_model.h5")
        self.model.save(model_path)

        # Save scalers
        feature_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")
        joblib.dump(self.feature_scaler, feature_scaler_path)
        joblib.dump(self.target_scaler, target_scaler_path)

        # Save configuration
        config_path = os.path.join(model_dir, "model_config.pkl")
        config_data = {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'epochs': self.epochs,
            'feature_columns': self.feature_columns,
            'volatility_threshold': self.volatility_threshold
        }
        joblib.dump(config_data, config_path)

        print(f"LSTM model saved to: {model_dir}")
        print(f"  - Model: {model_path}")
        print(f"  - Feature scaler: {feature_scaler_path}")
        print(f"  - Target scaler: {target_scaler_path}")
        print(f"  - Configuration: {config_path}")

    @classmethod
    def load_model(cls, model_dir: str = "lstm_model") -> 'LSTMVolatilityPredictor':
        """
        Load a trained LSTM model.

        Args:
            model_dir: Directory containing model files

        Returns:
            Loaded LSTMVolatilityPredictor instance
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required to load LSTM models")

        # Load configuration
        config_path = os.path.join(model_dir, "model_config.pkl")
        config_data = joblib.load(config_path)

        # Create instance
        instance = cls(
            sequence_length=config_data['sequence_length'],
            lstm_units=config_data['lstm_units'],
            epochs=config_data['epochs']
        )

        instance.feature_columns = config_data['feature_columns']
        instance.volatility_threshold = config_data['volatility_threshold']

        # Load model and scalers
        model_path = os.path.join(model_dir, "lstm_volatility_model.h5")
        feature_scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")

        instance.model = load_model(model_path)
        instance.feature_scaler = joblib.load(feature_scaler_path)
        instance.target_scaler = joblib.load(target_scaler_path)
        instance.is_fitted = True

        print(f"LSTM model loaded from: {model_dir}")
        return instance


def train_lstm_volatility_model(csv_path: str, sequence_length: int = 10,
                                lstm_units: int = 50, epochs: int = 100) -> LSTMVolatilityPredictor:
    """
    Train LSTM model for volatility prediction.

    Args:
        csv_path: Path to training CSV file
        sequence_length: Number of time steps for LSTM input
        lstm_units: Number of LSTM units
        epochs: Number of training epochs

    Returns:
        Trained LSTMVolatilityPredictor instance
    """
    print(f"Loading training data from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        print(f"Training data loaded. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Validate required columns
        predictor = LSTMVolatilityPredictor(sequence_length, lstm_units, epochs)
        missing_cols = [col for col in predictor.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Train model
        predictor.fit(df)

        # Save model
        predictor.save_model()

        return predictor

    except Exception as e:
        print(f"Error during training: {e}")
        raise


def test_lstm_predictions(test_csv_path: str, output_csv_path: str, model_dir: str = "lstm_model") -> pd.DataFrame:
    """
    Test LSTM model and generate predictions with risk assessment.

    Args:
        test_csv_path: Path to test CSV file
        output_csv_path: Path to save results
        model_dir: Directory containing trained model

    Returns:
        DataFrame with predictions and risk assessments
    """
    print("=== LSTM Volatility Prediction and Risk Assessment ===")
    print(f"Loading test data from: {test_csv_path}")

    try:
        # Load test data
        test_df = pd.read_csv(test_csv_path)

        # Sort by timestamp
        if 'timestamp' in test_df.columns:
            test_df = test_df.sort_values('timestamp').reset_index(drop=True)

        print(f"Test data loaded. Shape: {test_df.shape}")

        # Load trained model
        print(f"Loading LSTM model from: {model_dir}")
        predictor = LSTMVolatilityPredictor.load_model(model_dir)

        # Make predictions
        print("Generating volatility predictions...")
        predicted_volatility = predictor.predict(test_df)

        # Create results DataFrame
        results_df = test_df.copy()

        # Add actual volatility risk classification
        actual_risk = predictor.classify_volatility_risk(test_df['v'].values)
        results_df['actual_volatility_risk'] = actual_risk

        # Add predicted volatility (aligned to next row for future prediction)
        results_df['predicted_volatility'] = np.nan
        results_df['predicted_volatility_risk'] = 'low'

        # Align predictions to next time step
        for i in range(len(predicted_volatility) - 1):
            if not np.isnan(predicted_volatility[i]):
                results_df.loc[i + 1, 'predicted_volatility'] = predicted_volatility[i]
                # Classify predicted volatility risk
                pred_risk = predictor.classify_volatility_risk(np.array([predicted_volatility[i]]))[0]
                results_df.loc[i + 1, 'predicted_volatility_risk'] = pred_risk

        # Save results
        results_df.to_csv(output_csv_path, index=False)
        print(f"Results saved to: {output_csv_path}")

        # Generate analysis summary
        print("\n=== PREDICTION ANALYSIS ===")

        # Valid predictions (excluding NaN values)
        valid_predictions = ~np.isnan(results_df['predicted_volatility'])
        valid_count = valid_predictions.sum()

        print(f"Valid predictions: {valid_count} out of {len(results_df)} records")

        if valid_count > 0:
            # Actual risk distribution
            actual_risk_counts = results_df['actual_volatility_risk'].value_counts()
            print(f"\nActual Volatility Risk Distribution:")
            for risk, count in actual_risk_counts.items():
                percentage = (count / len(results_df)) * 100
                print(f"  {risk.capitalize()}: {count} ({percentage:.1f}%)")

            # Predicted risk distribution (for valid predictions)
            pred_risk_counts = results_df[valid_predictions]['predicted_volatility_risk'].value_counts()
            print(f"\nPredicted Volatility Risk Distribution:")
            for risk, count in pred_risk_counts.items():
                percentage = (count / valid_count) * 100
                print(f"  {risk.capitalize()}: {count} ({percentage:.1f}%)")

            # Prediction accuracy metrics
            valid_results = results_df[valid_predictions].copy()

            if len(valid_results) > 0:
                # Calculate RMSE and MAE for volatility predictions
                actual_vol = valid_results['v'].values
                pred_vol = valid_results['predicted_volatility'].values

                rmse = np.sqrt(np.mean((actual_vol - pred_vol) ** 2))
                mae = np.mean(np.abs(actual_vol - pred_vol))

                print(f"\nVolatility Prediction Metrics:")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  MAE: {mae:.4f}")

                # Risk prediction accuracy
                actual_risk_valid = valid_results['actual_volatility_risk'].values
                pred_risk_valid = valid_results['predicted_volatility_risk'].values

                risk_accuracy = np.mean(actual_risk_valid == pred_risk_valid)
                print(f"  Risk Classification Accuracy: {risk_accuracy:.3f} ({risk_accuracy * 100:.1f}%)")

                # Confusion matrix for risk prediction
                from collections import Counter
                risk_combinations = Counter(zip(actual_risk_valid, pred_risk_valid))

                print(f"\nRisk Prediction Confusion Matrix:")
                print(f"  (Actual, Predicted) -> Count")
                for (actual, predicted), count in risk_combinations.items():
                    print(f"  ({actual}, {predicted}) -> {count}")

        # Sample results display
        print(f"\nSample Results (first 10 rows with predictions):")
        display_cols = ['timestamp', 'v', 'actual_volatility_risk', 'predicted_volatility', 'predicted_volatility_risk']
        sample_with_predictions = results_df[valid_predictions][display_cols].head(10)

        if len(sample_with_predictions) > 0:
            print(sample_with_predictions.to_string(index=False))
        else:
            print("No valid predictions to display")

        return results_df

    except Exception as e:
        print(f"Error during testing: {e}")
        raise


# Utility functions
def generate_sample_financial_data(filename: str, n_samples: int = 2000, seed: int = 42):
    """
    Generate sample financial time series data with realistic patterns.

    Args:
        filename: Output CSV filename
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    print(f"Generating {n_samples} sample financial records...")

    # Generate timestamps
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='H')

    # Generate realistic financial data with trends and volatility patterns
    base_price = 100.0
    data = []

    for i in range(n_samples):
        # Create market cycles and trends
        cycle_component = 10 * np.sin(2 * np.pi * i / 100)  # 100-period cycle
        trend_component = 0.01 * i  # Slight upward trend
        noise_component = np.random.normal(0, 2)

        # Price evolution
        price_change = cycle_component + trend_component + noise_component
        base_price += price_change * 0.1  # Dampen changes

        # OHLC data
        open_price = base_price + np.random.normal(0, 0.5)
        high_price = open_price + abs(np.random.normal(2, 1))
        low_price = open_price - abs(np.random.normal(1.5, 0.8))
        close_price = open_price + np.random.normal(0, 1)

        # Volume with patterns (volatility proxy)
        # Create volatility clusters - periods of high and low volatility
        if i > 0:
            volatility_persistence = 0.7  # High volatility tends to cluster
            prev_vol = data[-1][5] if data else 10000

            if prev_vol > 20000:  # If previous volume was high
                # Tend to stay high with some probability
                if np.random.random() < volatility_persistence:
                    volume = abs(np.random.normal(25000, 8000))
                else:
                    volume = abs(np.random.normal(10000, 3000))
            else:
                # Normal volume with occasional spikes
                if np.random.random() < 0.05:  # 5% chance of spike
                    volume = abs(np.random.normal(40000, 10000))
                else:
                    volume = abs(np.random.normal(10000, 3000))
        else:
            volume = abs(np.random.normal(10000, 3000))

        # Technical indicators (with some correlation to price and volume)
        macd = np.random.normal(0.02 * price_change, 0.15)
        macd_signal = macd + np.random.normal(0, 0.05)
        macd_histogram = macd - macd_signal
        ma16 = base_price + np.random.normal(0, 1)
        roc = price_change + np.random.normal(0, 3)
        williams_r = np.random.uniform(-100, 0)
        vwma = base_price + np.random.normal(0, 0.5)
        lrma = base_price + np.random.normal(0, 0.3)

        data.append([
            timestamps[i], open_price, high_price, low_price, close_price, volume,
            macd, macd_signal, macd_histogram, ma16, roc, williams_r, vwma, lrma
        ])

    # Create DataFrame
    columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'v', 'MACD',
        'MACD_signal', 'MACD_histogram', 'ma16', 'ROC', 'williams_r', 'VWMA', 'LRMA'
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)

    print(f"Sample data saved to: {filename}")
    print(f"Volume range: {df['v'].min():.0f} to {df['v'].max():.0f}")
    print(f"Price range: {df['close'].min():.2f} to {df['close'].max():.2f}")

    # Show volatility distribution
    vol_95th = df['v'].quantile(0.95)
    high_vol_count = (df['v'] >= vol_95th).sum()
    print(f"Top 5% volatility threshold: {vol_95th:.0f}")
    print(f"High volatility records: {high_vol_count} ({high_vol_count / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    """
    Main execution example for LSTM volatility prediction
    """
    print("=== LSTM Volatility Prediction System ===\n")

    if not TENSORFLOW_AVAILABLE:
        print("ERROR: TensorFlow is not available. Please install it first:")
        print("pip install tensorflow")
        exit(1)

    # Example 1: Generate sample data
    # print("Generating sample financial data...")
    # generate_sample_financial_data('lstm_training_data.csv', n_samples=3000, seed=42)
    # generate_sample_financial_data('lstm_test_data.csv', n_samples=1000, seed=123)

    # Example 2: Train LSTM model
    try:
        print("\n" + "=" * 60)
        print("TRAINING LSTM MODEL")
        print("=" * 60)

        trained_model = train_lstm_volatility_model(
            csv_path='stock_data_with_indicators.csv',
            sequence_length=15,  # Look back 15 time steps
            lstm_units=64,  # 64 LSTM units
            epochs=50  # 50 training epochs
        )
        print("✓ LSTM training completed successfully!")

    except Exception as e:
        print(f"Training failed: {e}")
        exit(1)

    # Example 3: Test model and generate predictions
    try:
        print("\n" + "=" * 60)
        print("TESTING LSTM MODEL & GENERATING PREDICTIONS")
        print("=" * 60)

        results_df = test_lstm_predictions(
            test_csv_path='stock_data_with_indicators_model_validation.csv',
            output_csv_path='lstm_volatility_predictions.csv'
        )
        print("✓ LSTM testing and prediction completed successfully!")

    except Exception as e:
        print(f"Testing failed: {e}")
        exit(1)

    print(f"\n{'=' * 60}")
    print("LSTM VOLATILITY PREDICTION COMPLETE!")
    print("=" * 60)
    print("Output files generated:")
    print("📁 lstm_training_data.csv - Training data")
    print("📁 lstm_test_data.csv - Test data")
    print("📁 lstm_volatility_predictions.csv - Results with predictions")
    print("📁 lstm_model/ - Saved LSTM model directory")
    print("\nThe results file contains:")
    print("• Original time series data")
    print("• actual_volatility_risk (high/low based on top 5%)")
    print("• predicted_volatility (LSTM predictions)")
    print("• predicted_volatility_risk (risk classification of predictions)")
    print("• Predictions are aligned to next time step for future forecasting")