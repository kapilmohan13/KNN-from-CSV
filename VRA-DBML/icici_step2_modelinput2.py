import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesDBSCANClustering:
    """
    A class for performing DBSCAN clustering on time series financial data
    with risk assessment based on cluster transitions.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize the clustering model.

        Args:
            eps: The maximum distance between two samples for them to be considered neighbors
            min_samples: The number of samples in a neighborhood for a point to be considered core
        """
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'v', 'MACD_signal',
            'MACD_histogram', 'ma16', 'ROC', 'williams_r', 'VWMA', 'LRMA'
        ]
        self.is_fitted = False

    def prepare_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare and validate the input data.

        Args:
            df: DataFrame with time series data

        Returns:
            Numpy array of features ready for clustering
        """
        # Validate required columns
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Extract features
        features = df[self.feature_columns].copy()

        # Handle missing values
        if features.isnull().any().any():
            print(
                f"Warning: Found {features.isnull().sum().sum()} missing values. Filling with forward fill and backward fill.")
            features = features.fillna(method='ffill').fillna(method='bfill')

        return features.values

    def fit(self, df: pd.DataFrame) -> 'TimeSeriesDBSCANClustering':
        """
        Fit the DBSCAN model on the training data.

        Args:
            df: Training DataFrame with time series data

        Returns:
            Self for method chaining
        """
        print("Preparing training data...")
        X = self.prepare_data(df)

        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        print("Fitting DBSCAN model...")
        self.dbscan.fit(X_scaled)

        # Get cluster labels
        labels = self.dbscan.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        print(f"Clustering completed:")
        print(f"  - Number of clusters: {n_clusters}")
        print(f"  - Number of noise points: {n_noise}")
        print(f"  - Total points: {len(labels)}")

        self.is_fitted = True
        return self

    def predict_single(self, features: np.ndarray) -> int:
        """
        Predict cluster for a single sample.

        Args:
            features: Single sample features (1D array)

        Returns:
            Cluster label (-1 for noise/outlier)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Reshape for single sample prediction
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Find the closest core sample
        core_samples = self.dbscan.core_sample_indices_
        if len(core_samples) == 0:
            return -1  # No core samples found

        # Get training data (this is a limitation - in practice, you'd save this)
        # For now, we'll use a simple approach based on the fitted model
        labels = self.dbscan.labels_

        # Calculate distances to all core samples
        X_train_scaled = self.scaler.transform(self.X_train) if hasattr(self, 'X_train') else None

        if X_train_scaled is None:
            # Fallback: return -1 for outlier
            return -1

        distances = np.linalg.norm(X_train_scaled[core_samples] - features_scaled, axis=1)

        # Find closest core sample within eps distance
        min_dist_idx = np.argmin(distances)
        if distances[min_dist_idx] <= self.eps:
            closest_core_idx = core_samples[min_dist_idx]
            return labels[closest_core_idx]
        else:
            return -1  # Noise/outlier

    def save_model(self, scaler_path: str = 'scaler.pkl', model_path: str = 'dbscan_model.pkl'):
        """
        Save the fitted scaler and model.

        Args:
            scaler_path: Path to save the scaler
            model_path: Path to save the DBSCAN model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.dbscan, model_path)

        # Save model parameters and training data for prediction
        model_data = {
            'eps': self.eps,
            'min_samples': self.min_samples,
            'feature_columns': self.feature_columns,
            'X_train': getattr(self, 'X_train', None)
        }
        joblib.dump(model_data, model_path.replace('.pkl', '_data.pkl'))

        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

    @classmethod
    def load_model(cls, scaler_path: str = 'scaler.pkl',
                   model_path: str = 'dbscan_model.pkl') -> 'TimeSeriesDBSCANClustering':
        """
        Load a previously saved model and scaler.

        Args:
            scaler_path: Path to the saved scaler
            model_path: Path to the saved DBSCAN model

        Returns:
            Loaded TimeSeriesDBSCANClustering instance
        """
        # Load model data
        model_data = joblib.load(model_path.replace('.pkl', '_data.pkl'))

        # Create instance with saved parameters
        instance = cls(eps=model_data['eps'], min_samples=model_data['min_samples'])
        instance.feature_columns = model_data['feature_columns']
        instance.scaler = joblib.load(scaler_path)
        instance.dbscan = joblib.load(model_path)
        instance.X_train = model_data.get('X_train')
        instance.is_fitted = True

        print("Model and scaler loaded successfully")
        return instance


def train_model_from_csv(csv_path: str, eps: float = 0.5, min_samples: int = 5) -> TimeSeriesDBSCANClustering:
    """
    Train DBSCAN model from CSV file.

    Args:
        csv_path: Path to training CSV file
        eps: DBSCAN eps parameter
        min_samples: DBSCAN min_samples parameter

    Returns:
        Trained TimeSeriesDBSCANClustering instance
    """
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Sort by timestamp to ensure ascending order
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Training data shape: {df.shape}")
    print(f"Available columns: {list(df.columns)}")

    # Initialize and fit model
    model = TimeSeriesDBSCANClustering(eps=eps, min_samples=min_samples)

    # Store training data for prediction (workaround for DBSCAN limitation)
    X_train = model.prepare_data(df)
    model.X_train = X_train

    # Fit the model
    model.fit(df)

    # Save the model
    model.save_model()

    return model


def test_and_assess_risk(test_csv_path: str, output_csv_path: str,
                         scaler_path: str = 'scaler.pkl',
                         model_path: str = 'dbscan_model.pkl') -> pd.DataFrame:
    """
    Test the model on new data and assess risk based on cluster changes.

    Args:
        test_csv_path: Path to test CSV file
        output_csv_path: Path to save results with risk assessment
        scaler_path: Path to saved scaler
        model_path: Path to saved DBSCAN model

    Returns:
        DataFrame with risk assessments
    """
    print(f"Loading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    # Sort by timestamp
    if 'timestamp' in test_df.columns:
        test_df = test_df.sort_values('timestamp').reset_index(drop=True)

    print(f"Test data shape: {test_df.shape}")

    # Load model
    model = TimeSeriesDBSCANClustering.load_model(scaler_path, model_path)

    # Prepare results dataframe
    results_df = test_df.copy()
    results_df['cluster'] = -1
    results_df['risk'] = 'low'

    previous_cluster = None

    print("Processing records and assessing risk...")

    for idx, row in test_df.iterrows():
        # Extract features for current row
        features = np.array([row[col] for col in model.feature_columns])

        # Handle missing values
        if np.isnan(features).any():
            print(f"Warning: Missing values in row {idx}, filling with mean")
            features = np.nan_to_num(features, nan=0.0)

        # Predict cluster
        try:
            current_cluster = model.predict_single(features)
            results_df.loc[idx, 'cluster'] = current_cluster

            # Assess risk based on cluster change
            if previous_cluster is not None and current_cluster != previous_cluster:
                results_df.loc[idx, 'risk'] = 'high'
                print(f"Row {idx}: Cluster change detected ({previous_cluster} -> {current_cluster}) - HIGH RISK")
            else:
                results_df.loc[idx, 'risk'] = 'low'

            previous_cluster = current_cluster

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            results_df.loc[idx, 'cluster'] = -1
            results_df.loc[idx, 'risk'] = 'high'  # Unknown = high risk

    # Save results
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")

    # Print summary
    risk_summary = results_df['risk'].value_counts()
    cluster_summary = results_df['cluster'].value_counts()

    print("\nRisk Assessment Summary:")
    print(risk_summary)
    print("\nCluster Distribution:")
    print(cluster_summary)

    return results_df


# Example usage and main execution
if __name__ == "__main__":
    # Example 1: Train model from CSV
    try:
        print("=== Training DBSCAN Model ===")
        # Adjust these parameters based on your data characteristics
        trained_model = train_model_from_csv(
            csv_path='train.csv',  # Replace with your training file path
            eps=0.3,  # Adjust based on your data scale and clustering needs
            min_samples=5  # Minimum samples to form a cluster
        )
        print("Training completed successfully!")

    except FileNotFoundError:
        print("Training file not found. Please ensure 'stock_data_with_indicators.csv' exists.")
    except Exception as e:
        print(f"Training error: {e}")

    # Example 2: Test model and assess risk
    try:
        print("\n=== Testing Model and Assessing Risk ===")
        results = test_and_assess_risk(
            test_csv_path='valtest.csv',  # Replace with your test file path
            output_csv_path='risk_assessment_results.csv'
        )
        print("Risk assessment completed successfully!")

    except FileNotFoundError:
        print("Test file not found or model files missing.")
    except Exception as e:
        print(f"Testing error: {e}")


    # Example 3: Sample data generation (for testing purposes)
    def generate_sample_data(filename: str, n_samples: int = 1000):
        """Generate sample data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')

        # Generate correlated financial data
        base_price = 100
        data = []

        for i in range(n_samples):
            price_change = np.random.normal(0, 2)
            base_price += price_change

            open_price = base_price + np.random.normal(0, 0.5)
            high_price = open_price + abs(np.random.normal(0, 1))
            low_price = open_price - abs(np.random.normal(0, 1))
            close_price = open_price + np.random.normal(0, 0.8)
            volume = abs(np.random.normal(10000, 2000))

            # Technical indicators (simplified)
            macd_signal = np.random.normal(0, 0.1)
            macd_hist = np.random.normal(0, 0.2)
            ma16 = base_price + np.random.normal(0, 1)
            roc = np.random.normal(0, 5)
            williams_r = np.random.uniform(-100, 0)
            vwma = base_price + np.random.normal(0, 0.5)
            lrma = base_price + np.random.normal(0, 0.3)

            data.append([
                dates[i], open_price, high_price, low_price, close_price, volume,
                np.random.normal(0, 0.15),  # MACD
                macd_signal, macd_hist, ma16, roc, williams_r, vwma, lrma
            ])

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'v', 'MACD',
            'MACD_signal', 'MACD_histogram', 'ma16', 'ROC', 'williams_r', 'VWMA', 'LRMA'
        ])

        df.to_csv(filename, index=False)
        print(f"Sample data saved to: {filename}")
        return df

    # Uncomment to generate sample data for testing
    # generate_sample_data('stock_data_with_indicators_model_validation.csv', 2000)
    # generate_sample_data('test_data.csv', 500)