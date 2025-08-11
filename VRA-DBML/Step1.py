import pandas as pd
import numpy as np
import os


def classify_volatility_risk(input_file, output_file):
    """
    Classify stock volatility into risk categories based on percentiles.

    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file

    Returns:
    pd.DataFrame: DataFrame with volatility risk classification
    """

    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded data from {input_file}")
        print(f"Data shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

    # Display basic info about the dataset
    print("\nDataset Info:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")

    # Check if required columns exist
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'v']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
        print("Available columns:", list(df.columns))
        # Try to identify volatility column if 'v' is not found
        volatility_candidates = [col for col in df.columns if 'vol' in col.lower() or col.lower() == 'v']
        if volatility_candidates:
            print(f"Possible volatility columns: {volatility_candidates}")

    # Ensure the data is sorted by timestamp (latest first)
    if 'timestamp' in df.columns:
        # Convert timestamp to datetime if it's not already
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
            print(f"\nData sorted by timestamp (latest first)")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        except Exception as e:
            print(f"Warning: Could not process timestamp column: {e}")

    # Check if volatility column exists
    if 'v' not in df.columns:
        print("Error: Volatility column 'v' not found in the dataset")
        return None

    # Remove any rows with missing volatility values
    initial_rows = len(df)
    df = df.dropna(subset=['v'])
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} rows with missing volatility values")

    # Calculate volatility percentiles
    volatility_values = df['v'].values

    # Calculate the 10th and 90th percentiles
    low_threshold = np.percentile(volatility_values, 10)  # Bottom 10%
    high_threshold = np.percentile(volatility_values, 90)  # Top 10%

    print(f"\nVolatility Statistics:")
    print(f"Min volatility: {volatility_values.min():.6f}")
    print(f"Max volatility: {volatility_values.max():.6f}")
    print(f"Mean volatility: {volatility_values.mean():.6f}")
    print(f"Std volatility: {volatility_values.std():.6f}")
    print(f"\nThresholds:")
    print(f"Low threshold (10th percentile): {low_threshold:.6f}")
    print(f"High threshold (90th percentile): {high_threshold:.6f}")

    # Classify volatility risk
    def classify_risk(volatility):
        if volatility >= high_threshold:
            return 'high'
        elif volatility <= low_threshold:
            return 'low'
        else:
            return 'medium'

    # Apply classification
    df['vl'] = df['v'].apply(classify_risk)

    # Count classifications
    risk_counts = df['vl'].value_counts()
    print(f"\nVolatility Risk Classification Results:")
    for risk_level in ['high', 'medium', 'low']:
        count = risk_counts.get(risk_level, 0)
        percentage = (count / len(df)) * 100
        print(f"{risk_level.capitalize()} risk: {count} rows ({percentage:.1f}%)")

    # Identify the latest 3% of rows for performance calculation
    latest_3_percent_count = int(np.ceil(len(df) * 0.03))
    df['is_test_set'] = False
    df.loc[:latest_3_percent_count - 1, 'is_test_set'] = True

    print(f"\nTest Set Info:")
    print(f"Latest 3% of data: {latest_3_percent_count} rows marked for performance calculation")

    # Display sample of classified data
    print(f"\nSample of classified data:")
    print(df[['timestamp', 'open', 'high', 'low', 'close', 'v', 'vl', 'is_test_set']].head(10))

    # Save the results to CSV
    try:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

    return df


def main():
    """
    Main function to run the volatility risk classification
    """
    # File paths - modify these according to your file locations
    input_file = "ICICI_step1_input.csv"  # Change this to your input file path
    output_file = "ICICI_step1_output.csv"  # Output file path

    print("Stock Volatility Risk Classification - Step 1")
    print("=" * 50)

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found.")
        print("Please ensure your CSV file is in the same directory or update the file path.")

        # Create a sample dataset for demonstration
        print("\nCreating sample dataset for demonstration...")
        create_sample_data(input_file)

    # Run the classification
    result_df = classify_volatility_risk(input_file, output_file)

    if result_df is not None:
        print(f"\nStep 1 completed successfully!")
        print(f"Classified dataset saved as: {output_file}")
        print(f"Ready for DBSCAN analysis in the next step.")
    else:
        print("Classification failed. Please check your input data.")


def create_sample_data(filename):
    """
    Create sample stock data for demonstration purposes
    """
    np.random.seed(42)
    n_samples = 1000

    # Generate sample timestamps (last 1000 trading days)
    dates = pd.date_range(end='2024-01-31', periods=n_samples, freq='D')

    # Generate sample stock data
    base_price = 100
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]

    for i in range(1, n_samples):
        prices.append(prices[-1] * (1 + returns[i]))

    # Create OHLC data
    sample_data = {
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'v': np.abs(np.random.normal(0.02, 0.015, n_samples))  # Volatility column
    }

    df_sample = pd.DataFrame(sample_data)
    df_sample = df_sample.sort_values('timestamp', ascending=False)  # Latest first
    df_sample.to_csv(filename, index=False)
    print(f"Sample dataset created: {filename}")


if __name__ == "__main__":
    main()