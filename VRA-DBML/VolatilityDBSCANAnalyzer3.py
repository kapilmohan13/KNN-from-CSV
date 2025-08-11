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


class TimeSeriesDBSCANAnalyzer:
    """
    DBSCAN clustering analyzer for time series stock data to detect risk regime changes
    """

    def __init__(self, eps=0.5, min_samples=5):
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
        Load and prepare time series data for DBSCAN clustering

        Parameters:
        input_file (str): Path to the CSV file

        Returns:
        pd.DataFrame: Prepared and sorted dataframe
        """
        try:
            df = pd.read_csv(input_file)
            print(f"Loaded data from {input_file}")
            print(f"Initial data shape: {df.shape}")

            # Check if all required columns exist
            missing_cols = [col for col in self.feature_columns + ['timestamp'] if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Convert timestamp to datetime if it's not already
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print("Timestamp column converted to datetime")
            except Exception as e:
                print(f"Warning: Could not convert timestamp: {e}")

            # Sort by timestamp in ascending order (oldest first)
            df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
            print(f"Data sorted by timestamp in ascending order")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            # Remove rows with missing values in feature columns
            initial_rows = len(df)
            df = df.dropna(subset=self.feature_columns)
            if len(df) < initial_rows:
                print(f"Removed {initial_rows - len(df)} rows with missing values")

            # Display basic statistics
            print(f"\nFinal data shape: {df.shape}")
            print(f"\nFeature statistics:")
            print(df[self.feature_columns].describe())

            return df

        except FileNotFoundError:
            print(f"Error: File {input_file} not found.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def train_dbscan_model(self, df, save_models=True):
        """
        Train DBSCAN model on the time series data

        Parameters:
        df (pd.DataFrame): Input dataframe with features
        save_models (bool): Whether to save scaler and model

        Returns:
        dict: Training results including cluster labels and metrics
        """
        print("\n" + "=" * 60)
        print("TRAINING DBSCAN MODEL FOR TIME SERIES RISK DETECTION")
        print("=" * 60)

        # Extract features for clustering
        X = df[self.feature_columns].values
        print(f"Features used: {self.feature_columns}")
        print(f"Feature matrix shape: {X.shape}")

        # Normalize/Scale the features (essential for DBSCAN)
        print("\nScaling features using StandardScaler...")
        X_scaled = self.scaler.fit_transform(X)

        print("Feature scaling statistics:")
        for i, col in enumerate(self.feature_columns):
            original_mean = X[:, i].mean()
            original_std = X[:, i].std()
            scaled_mean = X_scaled[:, i].mean()
            scaled_std = X_scaled[:, i].std()
            print(
                f"{col}: Original(μ={original_mean:.4f}, σ={original_std:.4f}) -> Scaled(μ={scaled_mean:.4f}, σ={scaled_std:.4f})")

        # Apply DBSCAN clustering with Euclidean distance
        print(f"\nApplying DBSCAN clustering...")
        print(f"Parameters: eps={self.eps}, min_samples={self.min_samples}, metric='euclidean'")
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='euclidean')
        cluster_labels = self.dbscan.fit_predict(X_scaled)

        # Add cluster labels to dataframe
        df_result = df.copy()
        df_result['cluster'] = cluster_labels

        # Calculate clustering metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f"\nClustering Results:")
        print(f"Number of clusters found: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Percentage of noise: {(n_noise / len(cluster_labels)) * 100:.2f}%")

        # Cluster distribution
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print(f"\nCluster size distribution:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(cluster_labels)) * 100
            cluster_name = "Noise" if label == -1 else f"Cluster {label}"
            print(f"{cluster_name}: {count} points ({percentage:.2f}%)")

        # Calculate silhouette score (only if we have more than 1 cluster and not all noise)
        if n_clusters > 1:
            mask = cluster_labels != -1
            if np.sum(mask) > 1:
                try:
                    silhouette_avg = silhouette_score(X_scaled[mask], cluster_labels[mask])
                    print(f"\nAverage silhouette score: {silhouette_avg:.4f}")
                    print("(Silhouette score ranges from -1 to 1, higher is better)")
                except Exception as e:
                    print(f"Could not calculate silhouette score: {e}")

        # Analyze cluster characteristics
        self.analyze_cluster_characteristics(df_result)

        # Detect cluster transitions in training data
        self.analyze_cluster_transitions(df_result)

        # Save models
        if save_models:
            self.save_models()

        self.is_trained = True

        results = {
            'dataframe': df_result,
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'X_scaled': X_scaled,
            'silhouette_score': silhouette_avg if n_clusters > 1 and np.sum(mask) > 1 else None
        }

        return results

    def analyze_cluster_characteristics(self, df_result):
        """
        Analyze the characteristics of each cluster

        Parameters:
        df_result (pd.DataFrame): DataFrame with cluster assignments
        """
        print(f"\n" + "=" * 40)
        print("CLUSTER CHARACTERISTICS ANALYSIS")
        print("=" * 40)

        for cluster_id in sorted(df_result['cluster'].unique()):
            cluster_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
            cluster_data = df_result[df_result['cluster'] == cluster_id]

            print(f"\n{cluster_name} ({len(cluster_data)} points):")

            # Calculate feature statistics for this cluster
            feature_stats = cluster_data[self.feature_columns].agg(['mean', 'std', 'min', 'max'])

            for feature in self.feature_columns:
                mean_val = feature_stats.loc['mean', feature]
                std_val = feature_stats.loc['std', feature]
                min_val = feature_stats.loc['min', feature]
                max_val = feature_stats.loc['max', feature]
                print(f"  {feature}: μ={mean_val:.4f}, σ={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")

    def analyze_cluster_transitions(self, df_result):
        """
        Analyze cluster transitions in the training data

        Parameters:
        df_result (pd.DataFrame): DataFrame with cluster assignments
        """
        print(f"\n" + "=" * 40)
        print("CLUSTER TRANSITION ANALYSIS")
        print("=" * 40)

        transitions = []
        prev_cluster = None

        for idx, row in df_result.iterrows():
            current_cluster = row['cluster']
            if prev_cluster is not None and current_cluster != prev_cluster:
                transitions.append({
                    'timestamp': row['timestamp'],
                    'from_cluster': prev_cluster,
                    'to_cluster': current_cluster,
                    'index': idx
                })
            prev_cluster = current_cluster

        print(f"Total cluster transitions detected: {len(transitions)}")

        if transitions:
            print(f"\nSample of cluster transitions:")
            for i, trans in enumerate(transitions[:10]):  # Show first 10 transitions
                from_name = "Noise" if trans['from_cluster'] == -1 else f"Cluster {trans['from_cluster']}"
                to_name = "Noise" if trans['to_cluster'] == -1 else f"Cluster {trans['to_cluster']}"
                print(f"  {trans['timestamp']}: {from_name} -> {to_name}")

        return transitions

    def save_models(self, scaler_path='timeseries_scaler.pkl', dbscan_path='timeseries_dbscan.pkl'):
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
            print(f"\nScaler saved to {scaler_path}")

            # Save DBSCAN model
            with open(dbscan_path, 'wb') as f:
                pickle.dump(self.dbscan, f)
            print(f"DBSCAN model saved to {dbscan_path}")

            # Save additional model info
            model_info = {
                'eps': self.eps,
                'min_samples': self.min_samples,
                'feature_columns': self.feature_columns,
                'n_features': len(self.feature_columns)
            }

            with open('model_info.pkl', 'wb') as f:
                pickle.dump(model_info, f)
            print(f"Model info saved to model_info.pkl")

        except Exception as e:
            print(f"Error saving models: {e}")

    def load_models(self, scaler_path='timeseries_scaler.pkl', dbscan_path='timeseries_dbscan.pkl'):
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

            # Load model info if available
            try:
                with open('model_info.pkl', 'rb') as f:
                    model_info = pickle.load(f)
                    self.eps = model_info['eps']
                    self.min_samples = model_info['min_samples']
                    self.feature_columns = model_info['feature_columns']
                print(f"Model info loaded: eps={self.eps}, min_samples={self.min_samples}")
            except FileNotFoundError:
                print("Model info file not found, using default parameters")

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

        # Scale features using the trained scaler
        features_scaled = self.scaler.transform(features)

        # Predict cluster using the trained model's core samples
        # Since DBSCAN doesn't have a direct predict method, we'll find the nearest core sample
        if hasattr(self.dbscan, 'components_') and len(self.dbscan.components_) > 0:
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


def detect_cluster_changes(test_file, output_file, analyzer=None):
    """
    Detect cluster changes in new time series data to identify risk regime transitions

    Parameters:
    test_file (str): Path to test CSV file with time series data
    output_file (str): Path to save records where cluster changes occurred
    analyzer (TimeSeriesDBSCANAnalyzer): Trained analyzer instance

    Returns:
    pd.DataFrame: DataFrame with cluster change records
    """
    print("\n" + "=" * 60)
    print("DETECTING CLUSTER CHANGES FOR TRADING DECISIONS")
    print("=" * 60)

    # Load or create analyzer
    if analyzer is None:
        analyzer = TimeSeriesDBSCANAnalyzer()
        if not analyzer.load_models():
            print("Error: Could not load saved models. Please train the model first.")
            return None

    try:
        # Load test data
        df_test = pd.read_csv(test_file)
        print(f"Loaded test data from {test_file}")
        print(f"Test data shape: {df_test.shape}")

        # Convert timestamp and sort by timestamp in ascending order
        if 'timestamp' in df_test.columns:
            df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
            df_test = df_test.sort_values('timestamp', ascending=True).reset_index(drop=True)
            print("Test data sorted by timestamp in ascending order")
            print(f"Test date range: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")

        # Check required columns
        required_cols = analyzer.feature_columns
        missing_cols = [col for col in required_cols if col not in df_test.columns]
        if missing_cols:
            print(f"Error: Missing columns in test data: {missing_cols}")
            return None

        # Process each row and predict cluster
        cluster_changes = []
        all_predictions = []
        previous_cluster = None

        print(f"\nProcessing {len(df_test)} records to detect cluster changes...")

        for idx, row in df_test.iterrows():
            try:
                # Predict cluster for current row
                current_cluster = analyzer.predict_cluster(row)

                # Store prediction for analysis
                all_predictions.append({
                    'index': idx,
                    'timestamp': row['timestamp'],
                    'cluster': current_cluster
                })

                # Check if cluster changed from previous row
                if previous_cluster is not None and current_cluster != previous_cluster:
                    # Add transition record
                    transition_record = row.copy()
                    transition_record['previous_cluster'] = previous_cluster
                    transition_record['current_cluster'] = current_cluster
                    transition_record['transition_type'] = determine_risk_transition(previous_cluster, current_cluster)
                    transition_record['detection_timestamp'] = datetime.now()
                    transition_record['row_index'] = idx

                    cluster_changes.append(transition_record)

                    prev_name = "Noise" if previous_cluster == -1 else f"Cluster {previous_cluster}"
                    curr_name = "Noise" if current_cluster == -1 else f"Cluster {current_cluster}"
                    print(
                        f"  Transition at {row['timestamp']}: {prev_name} -> {curr_name} ({transition_record['transition_type']})")

                previous_cluster = current_cluster

                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(df_test)} records...")

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

        # Create DataFrame with cluster changes
        if cluster_changes:
            df_changes = pd.DataFrame(cluster_changes)

            print(f"\nCLUSTER CHANGE DETECTION RESULTS:")
            print(f"Total transitions detected: {len(df_changes)}")

            # Analyze transition types
            if 'transition_type' in df_changes.columns:
                transition_counts = df_changes['transition_type'].value_counts()
                print(f"\nTransition type distribution:")
                for trans_type, count in transition_counts.items():
                    percentage = (count / len(df_changes)) * 100
                    print(f"  {trans_type}: {count} ({percentage:.1f}%)")

            # Display sample of changes
            print(f"\nSample of detected transitions:")
            display_cols = ['timestamp', 'previous_cluster', 'current_cluster', 'transition_type', 'v']
            if all(col in df_changes.columns for col in display_cols):
                print(df_changes[display_cols].head(10).to_string(index=False))

            # Save results
            df_changes.to_csv(output_file, index=False)
            print(f"\nCluster transition records saved to {output_file}")
            print(f"These transitions can help traders identify risk regime changes!")

            return df_changes

        else:
            print("No cluster changes detected in the test data")
            # Still save the predictions for analysis
            df_predictions = pd.DataFrame(all_predictions)
            predictions_file = output_file.replace('.csv', '_predictions.csv')
            df_predictions.to_csv(predictions_file, index=False)
            print(f"All predictions saved to {predictions_file}")
            return pd.DataFrame()

    except FileNotFoundError:
        print(f"Error: Test file {test_file} not found.")
        return None
    except Exception as e:
        print(f"Error in cluster change detection: {e}")
        return None


def determine_risk_transition(prev_cluster, curr_cluster):
    """
    Determine the type of risk transition based on cluster change

    Parameters:
    prev_cluster (int): Previous cluster ID
    curr_cluster (int): Current cluster ID

    Returns:
    str: Type of transition
    """
    if prev_cluster == -1 and curr_cluster != -1:
        return "Noise_to_Regime"
    elif prev_cluster != -1 and curr_cluster == -1:
        return "Regime_to_Noise"
    elif prev_cluster == -1 and curr_cluster == -1:
        return "Noise_to_Noise"
    else:
        return f"Regime_Change_{prev_cluster}_to_{curr_cluster}"


def main():
    """
    Main function to run time series DBSCAN training and testing
    """
    print("Time Series DBSCAN Clustering for Trading Risk Detection")
    print("=" * 70)

    # File paths
    input_file = "icici_step2_modelinput1.csv"  # Training data
    test_file = "icici_model_validation_input1.csv"  # Test data
    cluster_changes_file = "cluster_transitions.csv"  # Output file for transitions

    # Initialize analyzer with parameters suitable for more sensitive detection
    analyzer = TimeSeriesDBSCANAnalyzer(eps=0.8, min_samples=3)

    # Step 1: Train DBSCAN model
    if os.path.exists(input_file):
        print(f"\nStep 1: Training DBSCAN model on time series data...")

        # Load and prepare data
        df = analyzer.load_and_prepare_data(input_file)

        if df is not None:
            # Train model
            results = analyzer.train_dbscan_model(df, save_models=True)

            if results:
                # Save training results
                results['dataframe'].to_csv('timeseries_with_clusters.csv', index=False)
                print(f"\nTraining completed! Results saved to 'timeseries_with_clusters.csv'")
        else:
            print("Could not load training data.")
            return
    else:
        print(f"Training file {input_file} not found. Creating sample data...")
        create_sample_timeseries_data(input_file)

        # Load and train on sample data
        df = analyzer.load_and_prepare_data(input_file)
        if df is not None:
            results = analyzer.train_dbscan_model(df, save_models=True)
            results['dataframe'].to_csv('timeseries_with_clusters.csv', index=False)

    # Step 2: Test cluster change detection
    if not os.path.exists(test_file):
        print(f"\nTest file {test_file} not found. Creating sample test data...")
        create_sample_test_timeseries(test_file)

    print(f"\nStep 2: Detecting cluster changes for trading decisions...")
    cluster_changes_df = detect_cluster_changes(test_file, cluster_changes_file, analyzer)

    if cluster_changes_df is not None and not cluster_changes_df.empty:
        print(f"\nDetection completed! Cluster transitions saved to {cluster_changes_file}")
        print("\nUse these transitions to make trading decisions:")
        print("- Regime changes may indicate volatility shifts")
        print("- Transitions from noise to regime suggest emerging patterns")
        print("- Transitions to noise may indicate increased uncertainty")
    else:
        print("No cluster changes detected or detection failed.")


def create_sample_timeseries_data(filename, n_samples=1000):
    """
    Create sample time series data for training
    """
    np.random.seed(42)

    # Generate timestamps
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')

    # Create regime-based time series with different volatility periods
    base_price = 100
    prices = [base_price]
    volatilities = []

    for i in range(1, n_samples):
        # Create different regimes
        if i < n_samples // 4:  # Low volatility regime
            vol = 0.01
            daily_return = np.random.normal(0.0005, vol)
        elif i < n_samples // 2:  # Medium volatility regime
            vol = 0.02
            daily_return = np.random.normal(0.001, vol)
        elif i < 3 * n_samples // 4:  # High volatility regime
            vol = 0.035
            daily_return = np.random.normal(-0.0005, vol)
        else:  # Return to low volatility
            vol = 0.015
            daily_return = np.random.normal(0.0008, vol)

        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
        volatilities.append(vol + np.random.normal(0, 0.005))  # Add noise to volatility

    # Generate OHLC data based on close prices
    opens = [prices[0]] + prices[:-1]  # Open is previous close
    highs = [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices]

    # Ensure high >= close >= low and high >= open >= low
    for i in range(len(prices)):
        if highs[i] < prices[i]:
            highs[i] = prices[i] * 1.001
        if lows[i] > prices[i]:
            lows[i] = prices[i] * 0.999
        if highs[i] < opens[i]:
            highs[i] = opens[i] * 1.001
        if lows[i] > opens[i]:
            lows[i] = opens[i] * 0.999

    sample_data = {
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'v': volatilities
    }

    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv(filename, index=False)
    print(f"Sample time series data created: {filename}")


def create_sample_test_timeseries(filename, n_samples=300):
    """
    Create sample test time series data with regime changes
    """
    np.random.seed(123)

    # Generate test timestamps (continuation from training data)
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')

    base_price = 110  # Different starting price
    prices = [base_price]
    volatilities = []

    for i in range(1, n_samples):
        # Create different regimes with transitions
        if i < n_samples // 5:  # Start with medium volatility
            vol = 0.025
            daily_return = np.random.normal(0.001, vol)
        elif i < 2 * n_samples // 5:  # Transition to high volatility
            vol = 0.04
            daily_return = np.random.normal(-0.001, vol)
        elif i < 3 * n_samples // 5:  # Stay in high volatility
            vol = 0.045
            daily_return = np.random.normal(-0.002, vol)
        elif i < 4 * n_samples // 5:  # Transition to low volatility
            vol = 0.012
            daily_return = np.random.normal(0.0015, vol)
        else:  # Return to medium volatility
            vol = 0.022
            daily_return = np.random.normal(0.0008, vol)

        new_price = prices[-1] * (1 + daily_return)
        prices.append(new_price)
        volatilities.append(vol + np.random.normal(0, 0.003))

    # Generate OHLC data
    opens = [prices[0]] + prices[:-1]
    highs = [p * (1 + abs(np.random.normal(0, 0.006))) for p in prices]
    lows = [p * (1 - abs(np.random.normal(0, 0.006))) for p in prices]

    # Ensure OHLC consistency
    for i in range(len(prices)):
        if highs[i] < prices[i]:
            highs[i] = prices[i] * 1.001
        if lows[i] > prices[i]:
            lows[i] = prices[i] * 0.999
        if highs[i] < opens[i]:
            highs[i] = opens[i] * 1.001
        if lows[i] > opens[i]:
            lows[i] = opens[i] * 0.999

    test_data = {
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'v': volatilities
    }

    df_test = pd.DataFrame(test_data)
    df_test.to_csv(filename, index=False)
    print(f"Sample test time series data created: {filename}")


if __name__ == "__main__":
    main()