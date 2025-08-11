import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


class VolatilityDBSCANAnalyzer:
    """
    DBSCAN clustering analyzer for stock volatility prediction
    """

    def __init__(self, eps=0.0003, min_samples=5):
        """
        Initialize the DBSCAN analyzer

        Parameters:
        eps (float): Maximum distance between two samples for them to be in the same neighborhood
        min_samples (int): Number of samples in a neighborhood for a point to be core point
        """
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        self.dbscan = None
        self.feature_columns = ['open', 'high', 'low', 'close', 'v']
        self.is_trained = False

    def load_and_prepare_data(self, input_file):
        """
        Load and prepare data for DBSCAN clustering

        Parameters:
        input_file (str): Path to the classified CSV file from Step 1

        Returns:
        pd.DataFrame: Prepared dataframe
        """
        try:
            df = pd.read_csv(input_file)
            print(f"Loaded data from {input_file}")
            print(f"Data shape: {df.shape}")

            # Check if all required columns exist
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Remove rows with missing values in feature columns
            initial_rows = len(df)
            df = df.dropna(subset=self.feature_columns)
            if len(df) < initial_rows:
                print(f"Removed {initial_rows - len(df)} rows with missing values")

            return df

        except FileNotFoundError:
            print(f"Error: File {input_file} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def train_dbscan_model(self, df, save_models=True):
        """
        Train DBSCAN model on the prepared data

        Parameters:
        df (pd.DataFrame): Input dataframe with features
        save_models (bool): Whether to save scaler and model

        Returns:
        dict: Training results including cluster labels and metrics
        """
        print("\n" + "=" * 50)
        print("TRAINING DBSCAN MODEL")
        print("=" * 50)

        # Extract features for clustering
        X = df[self.feature_columns].values
        print(f"Features used: {self.feature_columns}")
        print(f"Feature matrix shape: {X.shape}")

        # Normalize/Scale the features
        print("\nScaling features...")
        X_scaled = self.scaler.fit_transform(X)

        print("Feature scaling statistics:")
        for i, col in enumerate(self.feature_columns):
            print(f"{col}: mean={X_scaled[:, i].mean():.4f}, std={X_scaled[:, i].std():.4f}")

        # Apply DBSCAN clustering
        print(f"\nApplying DBSCAN (eps={self.eps}, min_samples={self.min_samples})...")
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean')
        cluster_labels = self.dbscan.fit_predict(X_scaled)

        # Add cluster labels to dataframe
        df_result = df.copy()
        df_result['cluster'] = cluster_labels

        # Calculate clustering metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f"\nClustering Results:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Percentage of noise: {(n_noise / len(cluster_labels)) * 100:.2f}%")

        # Cluster distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"\nCluster distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(cluster_labels)) * 100
            cluster_name = "Noise" if label == -1 else f"Cluster {label}"
            print(f"{cluster_name}: {count} points ({percentage:.2f}%)")

        # Calculate silhouette score (only if we have more than 1 cluster)
        if n_clusters > 1:
            # Remove noise points for silhouette calculation
            mask = cluster_labels != -1
            if np.sum(mask) > 1:
                silhouette_avg = silhouette_score(X_scaled[mask], cluster_labels[mask])
                print(f"Average silhouette score: {silhouette_avg:.4f}")

        # Analyze clusters by volatility risk
        print(f"\nCluster analysis by volatility risk (vl):")
        if 'vl' in df_result.columns:
            cluster_risk_analysis = df_result.groupby(['cluster', 'vl']).size().unstack(fill_value=0)
            print(cluster_risk_analysis)

        # Save models
        if save_models:
            self.save_models()

        self.is_trained = True

        results = {
            'dataframe': df_result,
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'X_scaled': X_scaled
        }

        return results

    def save_models(self, scaler_path='volatility_scaler.pkl', dbscan_path='volatility_dbscan.pkl'):
        """
        Save the trained scaler and DBSCAN model

        Parameters:
        scaler_path (str): Path to save the scaler
        dbscan_path (str): Path to save the DBSCAN model
        """
        try:
            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Scaler saved to {scaler_path}")

            # Save DBSCAN model
            with open(dbscan_path, 'wb') as f:
                pickle.dump(self.dbscan, f)
            print(f"DBSCAN model saved to {dbscan_path}")

        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self, scaler_path='volatility_scaler.pkl', dbscan_path='volatility_dbscan.pkl'):
        """
        Load the trained scaler and DBSCAN model

        Parameters:
        scaler_path (str): Path to the saved scaler
        dbscan_path (str): Path to the saved DBSCAN model
        """
        try:
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Scaler loaded from {scaler_path}")

            # Load DBSCAN model
            with open(dbscan_path, 'rb') as f:
                self.dbscan = pickle.load(f)
            print(f"DBSCAN model loaded from {dbscan_path}")

            self.is_trained = True
            return True

        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            return False
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def predict_cluster(self, row_data):
        """
        Predict cluster for a single row of data

        Parameters:
        row_data (dict or pd.Series): Single row with feature values

        Returns:
        int: Cluster label (-1 for noise)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Train the model first or load saved models.")

        # Extract features
        features = np.array([[row_data[col] for col in self.feature_columns]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict cluster using the trained model's core samples
        # Since DBSCAN doesn't have a direct predict method, we'll find the nearest core sample
        if hasattr(self.dbscan, 'components_'):
            from sklearn.metrics.pairwise import euclidean_distances

            # Calculate distances to all core samples
            distances = euclidean_distances(features_scaled, self.dbscan.components_)
            min_distance_idx = np.argmin(distances)
            min_distance = distances[0][min_distance_idx]

            # If within eps distance of a core sample, assign its cluster
            if min_distance <= self.eps:
                # Find which cluster this core sample belongs to
                core_sample_labels = self.dbscan.labels_[self.dbscan.core_sample_indices_]
                return core_sample_labels[min_distance_idx]
            else:
                return -1  # Noise point
        else:
            return -1  # No core samples found


def test_cluster_changes(test_file, output_file, analyzer=None):
    """
    Test method to identify cluster changes in new time series data

    Parameters:
    test_file (str): Path to test CSV file with similar structure
    output_file (str): Path to save records where cluster changes occurred
    analyzer (VolatilityDBSCANAnalyzer): Trained analyzer instance

    Returns:
    pd.DataFrame: DataFrame with cluster change records
    """
    print("\n" + "=" * 50)
    print("TESTING CLUSTER CHANGES")
    print("=" * 50)

    # Load or create analyzer
    if analyzer is None:
        analyzer = VolatilityDBSCANAnalyzer()
        if not analyzer.load_models():
            print("Error: Could not load saved models. Please train the model first.")
            return None

    try:
        # Load test data
        df_test = pd.read_csv(test_file)
        print(f"Loaded test data from {test_file}")
        print(f"Test data shape: {df_test.shape}")

        # Sort by timestamp in ascending order
        if 'timestamp' in df_test.columns:
            df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
            df_test = df_test.sort_values('timestamp', ascending=True).reset_index(drop=True)
            print("Data sorted by timestamp (ascending order)")

        # Check required columns
        required_cols = analyzer.feature_columns + ['vl']
        missing_cols = [col for col in required_cols if col not in df_test.columns]
        if missing_cols:
            print(f"Warning: Missing columns in test data: {missing_cols}")

        # Process each row and predict cluster
        cluster_changes = []
        previous_cluster = None

        print(f"\nProcessing {len(df_test)} records...")

        for idx, row in df_test.iterrows():
            try:
                # Predict cluster for current row
                current_cluster = analyzer.predict_cluster(row)

                # Check if cluster changed from previous row
                if previous_cluster is not None and current_cluster != previous_cluster:
                    # Add both previous and current row to changes
                    if idx > 0:
                        prev_row = df_test.iloc[idx - 1].copy()
                        prev_row['cluster'] = previous_cluster
                        prev_row['change_type'] = 'previous'
                        cluster_changes.append(prev_row)

                    curr_row = row.copy()
                    curr_row['cluster'] = current_cluster
                    curr_row['change_type'] = 'current'
                    cluster_changes.append(curr_row)

                    print(f"Cluster change detected at row {idx}: {previous_cluster} -> {current_cluster}")

                previous_cluster = current_cluster

                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(df_test)} records...")

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        # Create DataFrame with cluster changes
        if cluster_changes:
            df_changes = pd.DataFrame(cluster_changes)

            # Add additional information
            df_changes['detection_timestamp'] = datetime.now()

            print(f"\nFound {len(df_changes)} records with cluster changes")
            print(f"Number of cluster transitions: {len(df_changes) // 2}")

            # Display sample of changes
            print("\nSample of cluster changes:")
            print(df_changes[['timestamp', 'vl', 'cluster', 'change_type']].head(10))

            # Analyze volatility risk in cluster changes
            if 'vl' in df_changes.columns:
                print(f"\nVolatility risk distribution in cluster changes:")
                vl_counts = df_changes['vl'].value_counts()
                for risk, count in vl_counts.items():
                    percentage = (count / len(df_changes)) * 100
                    print(f"{risk.capitalize()} risk: {count} ({percentage:.1f}%)")

            # Save results
            df_changes.to_csv(output_file, index=False)
            print(f"\nCluster change records saved to {output_file}")

            return df_changes

        else:
            print("No cluster changes detected in the test data")
            return pd.DataFrame()

    except FileNotFoundError:
        print(f"Error: Test file {test_file} not found.")
        return None
    except Exception as e:
        print(f"Error in cluster change detection: {e}")
        return None


def main():
    """
    Main function to run DBSCAN training and testing
    """
    print("Stock Volatility DBSCAN Clustering - Step 2")
    print("=" * 60)

    # File paths
    input_file = "icici_step2_modelinput.csv"  # Output from Step 1
    test_file = "icici_model_validation_input.csv"  # Test data file
    cluster_changes_file = "cluster_changes.csv"  # Output file for cluster changes

    # Initialize analyzer
    analyzer = VolatilityDBSCANAnalyzer(eps=0.5, min_samples=5)

    # Step 2A: Train DBSCAN model
    if os.path.exists(input_file):
        print(f"\nStep 2A: Training DBSCAN model...")

        # Load and prepare data
        df = analyzer.load_and_prepare_data(input_file)

        if df is not None:
            # Train model
            results = analyzer.train_dbscan_model(df, save_models=True)

            if results:
                # Save training results
                results['dataframe'].to_csv('stock_data_with_clusters.csv', index=False)
                print(f"\nTraining completed! Results saved to 'stock_data_with_clusters.csv'")
        else:
            print("Could not load training data.")
            return
    else:
        print(f"Training file {input_file} not found. Please run Step 1 first.")
        return

    # Step 2B: Test cluster changes (create sample test data if not exists)
    if not os.path.exists(test_file):
        print(f"\nTest file {test_file} not found. Creating sample test data...")
        create_sample_test_data(test_file)

    print(f"\nStep 2B: Testing cluster changes...")
    cluster_changes_df = test_cluster_changes(test_file, cluster_changes_file, analyzer)

    if cluster_changes_df is not None and not cluster_changes_df.empty:
        print(f"\nTesting completed! Cluster changes saved to {cluster_changes_file}")
    else:
        print("No cluster changes detected or testing failed.")


def create_sample_test_data(filename, n_samples=200):
    """
    Create sample test data for demonstration
    """
    np.random.seed(123)

    # Generate sample data similar to training data but with some variations
    dates = pd.date_range(start='2024-02-01', periods=n_samples, freq='D')

    base_price = 105  # Slightly different base price
    returns = np.random.normal(0, 0.025, n_samples)  # Slightly higher volatility
    prices = [base_price]

    for i in range(1, n_samples):
        prices.append(prices[-1] * (1 + returns[i]))

    # Create test data with some regime changes
    volatilities = []
    risk_levels = []

    for i in range(n_samples):
        if i < n_samples // 3:
            vol = abs(np.random.normal(0.015, 0.005))  # Low volatility period
            risk = 'low' if vol < 0.02 else 'medium'
        elif i < 2 * n_samples // 3:
            vol = abs(np.random.normal(0.035, 0.01))  # High volatility period
            risk = 'high' if vol > 0.03 else 'medium'
        else:
            vol = abs(np.random.normal(0.02, 0.008))  # Medium volatility period
            risk = 'medium' if 0.015 < vol < 0.025 else ('high' if vol > 0.025 else 'low')

        volatilities.append(vol)
        risk_levels.append(risk)

    test_data = {
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.012))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.012))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.008)) for p in prices],
        'v': volatilities,
        'vl': risk_levels
    }

    df_test = pd.DataFrame(test_data)
    df_test.to_csv(filename, index=False)
    print(f"Sample test data created: {filename}")


if __name__ == "__main__":
    main()